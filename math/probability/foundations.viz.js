/* foundations.viz.js — the eight visualizations on math/probability/foundations.html.
   Loaded after ../../data.js → ../../notes.js → ../statistics/stats-viz.js (which provides
   the SC palette and the ST numeric/drawing toolbox — distribution functions come from ST,
   never re-implemented here).

   Everything page-local lives under the single namespace PF. Each figure is an IIFE that
   exits quietly if its <svg> is not on the page.

     1  #freq-svg      long-run relative frequency of heads (log-x), reseed, count-gap view
     2  #venn-svg      inclusion–exclusion on a draggable two/three-event Venn diagram
     3  #bday-svg      the birthday problem: exact product vs the exponential approximation
     4  #mosaic-svg    conditioning as renormalisation + the law of total probability
     5  #bayes-svg     diagnostic-test tree per 10,000 people: prevalence / sensitivity / specificity
     6  #indep-svg     pairwise vs mutual independence: two coins and the XOR event
     7  #rel-svg       reliability: series, parallel and k-of-n systems against component p
     8  #monty-svg     Monty Hall: 1,000 simulated games, switch vs stay                     */

const PF = {
  // Default seeds were chosen by the caption audit: each one shows the phenomenon its
  // caption claims (see the note at each figure).
  seeds: { freq: 21, monty: 6 },
  fmt: (x, d) => ST.fmt(x, d),
  // running relative frequency of successes in n Bernoulli(p) trials from a seeded stream
  runFreq(n, p, seed) {
    const r = ST.rng(seed), out = new Float64Array(n);
    let k = 0;
    for (let i = 0; i < n; i++) { if (r() < p) k++; out[i] = k; }
    return out;                                   // cumulative success counts
  },
  // birthday problem, exact: P(at least one shared) among n people over d equally likely days
  bdayExact(n, d) {
    let q = 1;
    for (let i = 0; i < n; i++) q *= (d - i) / d;
    return 1 - Math.max(q, 0);
  },
  bdayApprox: (n, d) => 1 - Math.exp(-n * (n - 1) / (2 * d)),
  // probability that at least k of n independent components (each working w.p. p) work
  kOfN(k, n, p) {
    if (k <= 0) return 1;
    return 1 - ST.binomCdf(k - 1, n, p);         // binomial tail from the shared library
  }
};

/* ─────────────────── 1 · long-run relative frequency ───────────────────
   Caption audit (60 seeds scanned): seed 21 at p = 0.5 opens with five heads in a row
   (f = 1.000 at n = 5), is 0.570 at n = 100, 0.494 at n = 1,000, and ends at 5,006 heads,
   f = 0.5006 — inside the ½ ± 2 sd band for every n ≥ 30 and on both sides of ½ late on.
   It converges to ½ and visibly to nothing else.                                        */
(function () {
  const svg = d3.select("#freq-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 680, H = 340, N = 10000;
  const F = ST.frame(svg, W, H, { l: 54, r: 20, t: 18, b: 40 });
  const x = d3.scaleLog().domain([1, N]).range([0, F.iw]);
  const gAx = F.g.append("g"), gBand = F.g.append("g"), gRef = F.g.append("g"),
    gPath = F.g.append("g"), gHover = F.g.append("g");

  const pIn = document.getElementById("fq-p"), pOut = document.getElementById("fq-pv");
  const band = document.getElementById("fq-band"), gapCb = document.getElementById("fq-gap");
  const out = document.getElementById("freq-readout");
  let seed = PF.seeds.freq, counts = null, p = 0.5, y = null;

  function simulate() { counts = PF.runFreq(N, p, seed); }

  function draw() {
    const gap = gapCb.checked;
    gAx.selectAll("*").remove(); gBand.selectAll("*").remove(); gRef.selectAll("*").remove();
    gPath.selectAll("*").remove();
    if (gap) {
      let mx = 0;
      for (let i = 0; i < N; i++) mx = Math.max(mx, Math.abs(2 * counts[i] - (i + 1) - (2 * p - 1) * (i + 1)));
      const lim = Math.max(mx, 2 * Math.sqrt(N * 4 * p * (1 - p))) * 1.08;
      y = d3.scaleLinear().domain([-lim, lim]).range([F.ih, 0]).nice();
    } else {
      y = d3.scaleLinear().domain([0, 1]).range([F.ih, 0]);
    }
    ST.gridY(gAx, y, F.iw, 5);
    gAx.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`)
      .call(d3.axisBottom(x).tickValues([1, 10, 100, 1000, 10000]).tickFormat(d3.format(",")));
    gAx.append("text").attr("x", F.iw).attr("y", F.ih + 32).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.muted).text("number of tosses n (log scale)");
    ST.axisL(gAx, y, 5, gap ? "excess heads: #H − #T − (expected)" : "proportion of heads so far");

    const ns = d3.range(1, N + 1);
    const sd1 = n => Math.sqrt(p * (1 - p) / n);           // sd of the proportion after n tosses
    if (band.checked) {
      const area = gap
        ? d3.area().x(n => x(n)).y0(n => y(-2 * 2 * Math.sqrt(n * p * (1 - p)))).y1(n => y(2 * 2 * Math.sqrt(n * p * (1 - p))))
        : d3.area().x(n => x(n)).y0(n => y(Math.max(0, p - 2 * sd1(n)))).y1(n => y(Math.min(1, p + 2 * sd1(n))));
      gBand.append("path").datum(ns.filter(n => n === 1 || n % 5 === 0 || n < 60))
        .attr("d", area).attr("fill", SC.accent).attr("opacity", 0.12);
    }
    const ref = gap ? 0 : p;
    gRef.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(ref)).attr("y2", y(ref))
      .attr("stroke", SC.good).attr("stroke-width", 1.5).attr("stroke-dasharray", "6 4");
    gRef.append("text").attr("x", F.iw - 4).attr("y", y(ref) - 6).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", SC.good)
      .text(gap ? "0 = exactly the expected number of heads" : `true P(heads) = ${ST.fmt(p, 2)}`);

    const val = i => gap ? (2 * counts[i] - (i + 1)) - (2 * p - 1) * (i + 1) : counts[i] / (i + 1);
    const line = d3.line().x((d, i) => x(i + 1)).y((d, i) => y(val(i)));
    gPath.append("path").datum(Array.from(counts)).attr("d", line)
      .attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 1.6);

    const at = [10, 100, 1000, 10000].map(n => `n = ${d3.format(",")(n)}: <b>${ST.fmt(counts[n - 1] / n, 4)}</b>`).join(" · ");
    const h = counts[N - 1];
    let gmx = 0, gat = 1, gsg = 0;
    for (let i = 0; i < N; i++) {
      const g = (2 * counts[i] - (i + 1)) - (2 * p - 1) * (i + 1);
      if (Math.abs(g) > gmx) { gmx = Math.abs(g); gat = i + 1; gsg = g; }
    }
    out.innerHTML = `seed ${seed} · proportion of heads at ${at}` +
      `<br>after 10,000 tosses: <b>${d3.format(",")(h)}</b> heads, ` +
      `#H − #T = <b>${d3.format("+,")(2 * h - N)}</b> · ` +
      `distance from p in standard deviations: <b>${ST.fmt((h / N - p) / sd1(N), 2)}</b>` +
      `<br>largest departure of the count gap from its expected value: <b>${d3.format("+,.0f")(gsg)}</b> at n = ${d3.format(",")(gat)}` +
      (gap ? " · the gap wanders on a √n scale instead of closing in on 0; the band is ±2 sd of the gap" : "");
  }

  // hover: a vertical rule with the value at that n
  const rule = gHover.append("line").attr("y1", 0).attr("y2", F.ih).attr("stroke", SC.muted)
    .attr("stroke-dasharray", "2 3").attr("opacity", 0);
  const tip = gHover.append("text").attr("font-size", 11).attr("fill", SC.ink).attr("opacity", 0);
  F.g.append("rect").attr("width", F.iw).attr("height", F.ih).attr("fill", "transparent")
    .on("mousemove", function (ev) {
      const mx = d3.pointer(ev)[0];
      const n = Math.max(1, Math.min(N, Math.round(x.invert(mx))));
      const v = gapCb.checked ? (2 * counts[n - 1] - n) - (2 * p - 1) * n : counts[n - 1] / n;
      rule.attr("x1", x(n)).attr("x2", x(n)).attr("opacity", 1);
      tip.attr("x", Math.min(x(n) + 6, F.iw - 150)).attr("y", 14).attr("opacity", 1)
        .text(`n = ${d3.format(",")(n)} · ${gapCb.checked ? "gap" : "proportion"} = ${gapCb.checked ? ST.fmt(v, 1) : ST.fmt(v, 4)}`);
    })
    .on("mouseleave", () => { rule.attr("opacity", 0); tip.attr("opacity", 0); });

  document.getElementById("fq-reseed").addEventListener("click", () => { seed = (seed * 48271 + 101) % 2147483647; simulate(); draw(); });
  document.getElementById("fq-reset").addEventListener("click", () => { seed = PF.seeds.freq; simulate(); draw(); });
  pIn.addEventListener("input", () => { p = +pIn.value; pOut.textContent = ST.fmt(p, 2); simulate(); draw(); });
  [band, gapCb].forEach(c => c.addEventListener("change", draw));

  simulate(); draw();
})();

/* ─────────────────── 2 · inclusion–exclusion as area ───────────────────
   Ω is a 640 × 270 rectangle whose area is defined to be 1. Single and pairwise
   probabilities come from the closed-form circle and lens areas; the union (and the
   triple overlap, which has no simple closed form) are measured on a 2-px grid. The
   readout compares the inclusion–exclusion sum with the grid-measured union, so the two
   numbers are computed independently and agree to grid precision.
   Caption audit: the default separation of A and B is solved for so that the lens area
   equals P(A)·P(B) exactly — the "independent-looking" layout the caption promises.      */
PF.lens = function (r1, r2, d) {
  if (d >= r1 + r2) return 0;
  if (d <= Math.abs(r1 - r2)) return Math.PI * Math.min(r1, r2) ** 2;
  const a = r1 * r1 * Math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1));
  const b = r2 * r2 * Math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2));
  const c = 0.5 * Math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2));
  return a + b - c;
};
(function () {
  const svg = d3.select("#venn-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 680, H = 380, X0 = 20, Y0 = 16, RW = 640, RH = 270, AREA = RW * RH;
  const raIn = document.getElementById("vn-ra"), rbIn = document.getElementById("vn-rb");
  const cCb = document.getElementById("vn-c"), out = document.getElementById("venn-readout");
  const col = { A: SC.accent, B: SC.a2, C: SC.good };
  let circ;

  function defaults() {
    const ra = 100, rb = 85;
    const target = (Math.PI * ra * ra / AREA) * (Math.PI * rb * rb / AREA) * AREA;   // lens = P(A)P(B)
    let lo = Math.abs(ra - rb), hi = ra + rb;
    for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (PF.lens(ra, rb, m) > target) lo = m; else hi = m; }
    const d = (lo + hi) / 2, cx = X0 + RW / 2, cy = Y0 + RH / 2;
    circ = {
      A: { x: cx - d / 2 - 40, y: cy, r: ra },
      B: { x: cx + d / 2 - 40, y: cy, r: rb },
      C: { x: cx - 40, y: cy + 60, r: 75 }        // under the A–B overlap, so all seven regions exist
    };
    raIn.value = ra; rbIn.value = rb;
  }

  svg.selectAll("*").remove();
  svg.append("rect").attr("x", X0).attr("y", Y0).attr("width", RW).attr("height", RH)
    .attr("fill", SC.panel2).attr("stroke", SC.line);
  svg.append("text").attr("x", X0 + RW - 8).attr("y", Y0 + 18).attr("text-anchor", "end")
    .attr("font-size", 13).attr("fill", SC.muted).text("Ω  (area = 1)");
  const gC = svg.append("g"), gBar = svg.append("g");

  function clampC(c) {
    c.x = ST.clamp(c.x, X0 + c.r, X0 + RW - c.r);
    c.y = ST.clamp(c.y, Y0 + c.r, Y0 + RH - c.r);
  }

  function measure(useC) {                      // grid areas: union and triple overlap
    let uni = 0, tri = 0;
    const st = 2, cell = st * st, A = circ.A, B = circ.B, C = circ.C;
    for (let px = X0 + 1; px < X0 + RW; px += st) {
      for (let py = Y0 + 1; py < Y0 + RH; py += st) {
        const inA = (px - A.x) ** 2 + (py - A.y) ** 2 <= A.r * A.r;
        const inB = (px - B.x) ** 2 + (py - B.y) ** 2 <= B.r * B.r;
        const inC = useC && (px - C.x) ** 2 + (py - C.y) ** 2 <= C.r * C.r;
        if (inA || inB || inC) uni += cell;
        if (inA && inB && inC) tri += cell;
      }
    }
    return { uni: uni / AREA, tri: tri / AREA };
  }

  function draw() {
    const useC = cCb.checked;
    const keys = useC ? ["A", "B", "C"] : ["A", "B"];
    keys.forEach(k => clampC(circ[k]));
    const sel = gC.selectAll("g.ev").data(keys, k => k);
    sel.exit().remove();
    const en = sel.enter().append("g").attr("class", "ev").style("cursor", "grab");
    en.append("circle").attr("fill-opacity", 0.22).attr("stroke-width", 2);
    en.append("text").attr("font-size", 16).attr("font-weight", 700);
    en.call(d3.drag().on("drag", function (ev, k) { circ[k].x += ev.dx; circ[k].y += ev.dy; draw(); }));
    const all = en.merge(sel);
    all.select("circle").attr("cx", k => circ[k].x).attr("cy", k => circ[k].y).attr("r", k => circ[k].r)
      .attr("fill", k => col[k]).attr("stroke", k => col[k]);
    all.select("text").attr("x", k => circ[k].x - circ[k].r * 0.55).attr("y", k => circ[k].y - circ[k].r * 0.55)
      .attr("fill", k => col[k]).text(k => k);

    const P = k => Math.PI * circ[k].r ** 2 / AREA;
    const I = (a, b) => PF.lens(circ[a].r, circ[b].r, Math.hypot(circ[a].x - circ[b].x, circ[a].y - circ[b].y)) / AREA;
    const m = measure(useC);
    let ie, txt;
    if (!useC) {
      ie = P("A") + P("B") - I("A", "B");
      txt = `P(A) = <b>${ST.fmt(P("A"), 4)}</b> · P(B) = <b>${ST.fmt(P("B"), 4)}</b> · P(A∩B) = <b>${ST.fmt(I("A", "B"), 4)}</b>` +
        `<br>P(A) + P(B) − P(A∩B) = <b>${ST.fmt(ie, 4)}</b> · P(A∪B) measured on the grid = <b>${ST.fmt(m.uni, 4)}</b>` +
        `<br>P(A)·P(B) = <b>${ST.fmt(P("A") * P("B"), 4)}</b> vs P(A∩B) = <b>${ST.fmt(I("A", "B"), 4)}</b> → ` +
        (I("A", "B") === 0 ? "disjoint: P(A∩B) = 0 although both events are possible, so they are strongly <b>dependent</b>"
          : Math.abs(I("A", "B") - P("A") * P("B")) < 0.0015 ? "equal (to 3 dp): A and B behave as <b>independent</b>"
            : "unequal: A and B are <b>dependent</b>");
    } else {
      const s1 = P("A") + P("B") + P("C"), s2 = I("A", "B") + I("A", "C") + I("B", "C");
      ie = s1 - s2 + m.tri;
      txt = `singles Σ = <b>${ST.fmt(s1, 4)}</b> · pairs Σ = <b>${ST.fmt(s2, 4)}</b> (AB ${ST.fmt(I("A", "B"), 4)}, AC ${ST.fmt(I("A", "C"), 4)}, BC ${ST.fmt(I("B", "C"), 4)}) · triple = <b>${ST.fmt(m.tri, 4)}</b>` +
        `<br>singles − pairs + triple = <b>${ST.fmt(ie, 4)}</b> · P(A∪B∪C) measured on the grid = <b>${ST.fmt(m.uni, 4)}</b>` +
        `<br>truncations alternate: singles ${ST.fmt(s1, 4)} ≥ union ≥ singles − pairs ${ST.fmt(s1 - s2, 4)}`;
    }
    out.innerHTML = txt;

    // stacked bars: the sum of singles over-counts the union by the overlaps
    gBar.selectAll("*").remove();
    const sx = d3.scaleLinear().domain([0, 1]).range([0, RW]);
    const by = Y0 + RH + 16;
    let acc = 0;
    keys.forEach(k => {
      gBar.append("rect").attr("x", X0 + sx(acc)).attr("y", by).attr("width", sx(P(k))).attr("height", 14)
        .attr("fill", col[k]).attr("opacity", 0.75);
      acc += P(k);
    });
    gBar.append("text").attr("x", X0 + sx(acc) + 6).attr("y", by + 11).attr("font-size", 11).attr("fill", SC.muted)
      .text(`sum of the ${useC ? "three" : "two"} probabilities = ${ST.fmt(acc, 3)}`);
    gBar.append("rect").attr("x", X0).attr("y", by + 22).attr("width", sx(m.uni)).attr("height", 14)
      .attr("fill", SC.violet).attr("opacity", 0.75);
    gBar.append("rect").attr("x", X0 + sx(m.uni)).attr("y", by + 22).attr("width", Math.max(0, sx(acc - m.uni)))
      .attr("height", 14).attr("fill", "none").attr("stroke", SC.bad).attr("stroke-dasharray", "3 2");
    gBar.append("text").attr("x", X0 + sx(Math.max(acc, m.uni)) + 6).attr("y", by + 33).attr("font-size", 11).attr("fill", SC.muted)
      .text(`union = ${ST.fmt(m.uni, 3)} · dashed = counted more than once`);
  }

  raIn.addEventListener("input", () => { circ.A.r = +raIn.value; draw(); });
  rbIn.addEventListener("input", () => { circ.B.r = +rbIn.value; draw(); });
  cCb.addEventListener("change", draw);
  document.getElementById("vn-disjoint").addEventListener("click", () => {
    circ.B.x = circ.A.x + circ.A.r + circ.B.r + 6;
    if (circ.B.x + circ.B.r > X0 + RW) { circ.A.x = X0 + circ.A.r + 2; circ.B.x = circ.A.x + circ.A.r + circ.B.r + 6; }
    circ.B.y = circ.A.y; draw();
  });
  document.getElementById("vn-reset").addEventListener("click", () => { defaults(); cCb.checked = false; draw(); });
  defaults(); draw();
})();

/* ─────────────────── 3 · the birthday problem ───────────────────
   Exact product vs e^(−n(n−1)/2N). Caption audit: at N = 365 the exact curve first
   passes ½ at n = 23 (0.5073; n = 22 gives 0.4757), matching the marker. Values checked
   against an independent Python computation and a 200,000-group Monte Carlo below.      */
(function () {
  const svg = d3.select("#bday-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 680, H = 330;
  const dSel = document.getElementById("bd-d"), apx = document.getElementById("bd-apx");
  const out = document.getElementById("bday-readout");
  const F = ST.frame(svg, W, H, { l: 52, r: 20, t: 16, b: 40 });
  const gA = F.g.append("g"), gL = F.g.append("g"), gH = F.g.append("g");
  let x, y, d, nMax, first;

  function draw() {
    d = +dSel.value;
    nMax = d === 12 ? 13 : d === 365 ? 80 : 120;
    x = d3.scaleLinear().domain([1, nMax]).range([0, F.iw]);
    y = d3.scaleLinear().domain([0, 1]).range([F.ih, 0]);
    gA.selectAll("*").remove(); gL.selectAll("*").remove();
    ST.gridY(gA, y, F.iw, 5);
    ST.axisB(gA, x, F.ih, 8, "group size n");
    ST.axisL(gA, y, 5, "P(at least two share a value)");
    const ns = d3.range(1, nMax + 1);
    first = ns.find(n => PF.bdayExact(n, d) >= 0.5);
    gL.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(0.5)).attr("y2", y(0.5))
      .attr("stroke", SC.muted).attr("stroke-dasharray", "3 4");
    gL.append("line").attr("x1", x(first)).attr("x2", x(first)).attr("y1", y(0)).attr("y2", y(PF.bdayExact(first, d)))
      .attr("stroke", SC.good).attr("stroke-dasharray", "4 3");
    gL.append("text").attr("x", x(first) + 6).attr("y", y(0.5) + 16).attr("font-size", 11.5).attr("fill", SC.good)
      .text(`n = ${first}: ${ST.fmt(PF.bdayExact(first, d), 4)}  (first n past ½)`);
    if (apx.checked) {
      gL.append("path").datum(ns).attr("fill", "none").attr("stroke", SC.violet).attr("stroke-width", 1.6)
        .attr("stroke-dasharray", "6 4").attr("d", d3.line().x(n => x(n)).y(n => y(PF.bdayApprox(n, d))));
    }
    gL.append("path").datum(ns).attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2.2)
      .attr("d", d3.line().x(n => x(n)).y(n => y(PF.bdayExact(n, d))));
    gL.selectAll("circle").data(ns.length <= 20 ? ns : []).join("circle")
      .attr("cx", n => x(n)).attr("cy", n => y(PF.bdayExact(n, d))).attr("r", 3).attr("fill", SC.a2);
    ST.legend(gL, [{ label: "exact 1 − Π(1 − i/N)", color: SC.a2 }].concat(apx.checked ? [{ label: "e^(−n(n−1)/2N)", color: SC.violet, dash: "6 4" }] : []),
      F.iw - 190, F.ih - 40);
    report(first);
  }
  function report(n) {
    const ex = PF.bdayExact(n, d), ap = PF.bdayApprox(n, d);
    out.innerHTML = `N = ${d3.format(",")(d)} · n = <b>${n}</b>: exact P(shared) = <b>${ST.fmt(ex, 4)}</b> · ` +
      `approximation = <b>${ST.fmt(ap, 4)}</b> · pairs C(n,2) = <b>${d3.format(",")(n * (n - 1) / 2)}</b>` +
      `<br>birthday-bound prediction of the 50% point, 1.18√N = <b>${ST.fmt(Math.sqrt(2 * d * Math.LN2), 1)}</b> · exact first n past ½ = <b>${first}</b>`;
  }
  const rule = gH.append("line").attr("y1", 0).attr("y2", F.ih).attr("stroke", SC.muted).attr("opacity", 0);
  const dot = gH.append("circle").attr("r", 5).attr("fill", "none").attr("stroke", SC.ink).attr("opacity", 0);
  F.g.append("rect").attr("width", F.iw).attr("height", F.ih).attr("fill", "transparent")
    .on("mousemove", function (ev) {
      const n = Math.max(1, Math.min(nMax, Math.round(x.invert(d3.pointer(ev)[0]))));
      rule.attr("x1", x(n)).attr("x2", x(n)).attr("opacity", 0.6);
      dot.attr("cx", x(n)).attr("cy", y(PF.bdayExact(n, d))).attr("opacity", 1);
      report(n);
    })
    .on("mouseleave", () => { rule.attr("opacity", 0); dot.attr("opacity", 0); report(first); });
  dSel.addEventListener("change", draw);
  apx.addEventListener("change", draw);
  draw();
})();

/* ─────────────────── 4 · mosaic: conditioning and total probability ───────────────────
   Four tiles AB, AᶜB, ABᶜ, AᶜBᶜ with areas equal to their joint probabilities. Three
   layouts: the whole of Ω; conditioned on B (the B column stretched to full width);
   conditioned on A (only A's tiles, rescaled to fill the rectangle, so the width of the
   B part is P(B | A)). Caption audit: defaults P(B) = 0.3, P(A|B) = 0.7, P(A|Bᶜ) = 0.2
   give P(A) = 0.35, strictly between 0.2 and 0.7, and P(B|A) = 0.6 ≠ P(A|B) = 0.7.     */
(function () {
  const svg = d3.select("#mosaic-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 680, H = 360, X0 = 40, Y0 = 18, RW = 470, RH = 300;
  const ids = ["ms-b", "ms-ab", "ms-anb"].map(i => document.getElementById(i));
  const vals = ["ms-bv", "ms-abv", "ms-anbv"].map(i => document.getElementById(i));
  const out = document.getElementById("mosaic-readout");
  let mode = "none";

  svg.selectAll("*").remove();
  svg.append("rect").attr("x", X0).attr("y", Y0).attr("width", RW).attr("height", RH)
    .attr("fill", "none").attr("stroke", SC.line);
  const gT = svg.append("g"), gLab = svg.append("g"), gLev = svg.append("g"), gSide = svg.append("g");
  const keys = ["AB", "AcB", "ABc", "AcBc"];
  const fill = { AB: SC.accent, ABc: SC.accent, AcB: SC.panel2, AcBc: SC.panel2 };
  const tiles = gT.selectAll("rect").data(keys).join("rect")
    .attr("stroke", SC.bg).attr("stroke-width", 2)
    .attr("fill", k => fill[k]).attr("fill-opacity", k => k[1] === "c" ? 1 : 0.75);

  function layout(b, ab, anb) {
    const pA = ab * b + anb * (1 - b), bGivenA = pA > 0 ? ab * b / pA : 0;
    // rectangles in unit coordinates: [x, y(top), w, h]
    if (mode === "A") {
      return {
        AB: [0, 0, bGivenA, 1], ABc: [bGivenA, 0, 1 - bGivenA, 1],
        AcB: [0, 0, bGivenA, 0], AcBc: [bGivenA, 0, 1 - bGivenA, 0]
      };
    }
    const wb = mode === "B" ? 1 : b, wn = mode === "B" ? 0 : 1 - b;
    return {
      AB: [0, 1 - ab, wb, ab], AcB: [0, 0, wb, 1 - ab],
      ABc: [wb, 1 - anb, wn, anb], AcBc: [wb, 0, wn, 1 - anb]
    };
  }

  function draw(animate) {
    const [b, ab, anb] = ids.map(e => +e.value);
    vals.forEach((v, i) => v.textContent = ST.fmt(+ids[i].value, 2));
    const pA = ab * b + anb * (1 - b), pAB = ab * b, pABc = anb * (1 - b);
    const bGivenA = pA > 0 ? pAB / pA : NaN;
    const L = layout(b, ab, anb);
    const t = animate ? tiles.transition().duration(750) : tiles;
    t.attr("x", k => X0 + L[k][0] * RW).attr("y", k => Y0 + L[k][1] * RH)
      .attr("width", k => Math.max(0, L[k][2] * RW)).attr("height", k => Math.max(0, L[k][3] * RH))
      .attr("opacity", k => (L[k][2] * L[k][3] > 0 ? 1 : 0));

    gLab.selectAll("*").remove();
    const joint = { AB: pAB, AcB: (1 - ab) * b, ABc: pABc, AcBc: (1 - anb) * (1 - b) };
    const name = { AB: "A∩B", AcB: "Aᶜ∩B", ABc: "A∩Bᶜ", AcBc: "Aᶜ∩Bᶜ" };
    const show = () => {
      gLab.selectAll("*").remove();
      keys.forEach(k => {
        const r = L[k];
        if (r[2] * RW < 58 || r[3] * RH < 30) return;
        const cx = X0 + (r[0] + r[2] / 2) * RW, cy = Y0 + (r[1] + r[3] / 2) * RH;
        const share = mode === "A" ? joint[k] / pA : mode === "B" ? joint[k] / b : joint[k];
        gLab.append("text").attr("x", cx).attr("y", cy - 4).attr("text-anchor", "middle")
          .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", k[1] === "c" ? SC.muted : SC.ink).text(name[k]);
        gLab.append("text").attr("x", cx).attr("y", cy + 13).attr("text-anchor", "middle")
          .attr("font-size", 11.5).attr("fill", k[1] === "c" ? SC.muted : SC.ink)
          .text(mode === "none" ? `area ${ST.fmt(share, 3)}` : `${ST.fmt(share, 3)} of the new Ω`);
      });
    };
    if (animate) setTimeout(show, 760); else show();

    // the P(A) level (whole Ω) or P(A|B) level (conditioned on B)
    gLev.selectAll("*").remove();
    if (mode !== "A") {
      const lev = mode === "B" ? ab : pA;
      gLev.append("line").attr("x1", X0).attr("x2", X0 + RW).attr("y1", Y0 + (1 - lev) * RH).attr("y2", Y0 + (1 - lev) * RH)
        .attr("stroke", SC.a2).attr("stroke-width", 2).attr("stroke-dasharray", "7 5");
      gLev.append("text").attr("x", X0 + RW - 4).attr("y", Y0 + (1 - lev) * RH - 6).attr("text-anchor", "end")
        .attr("font-size", 11.5).attr("fill", SC.a2).text(mode === "B" ? `P(A | B) = ${ST.fmt(ab, 3)}` : `P(A) = ${ST.fmt(pA, 3)}`);
    }
    // column captions under the rectangle
    if (mode === "none") {
      gLev.append("text").attr("x", X0 + b * RW / 2).attr("y", Y0 + RH + 16).attr("text-anchor", "middle")
        .attr("font-size", 11.5).attr("fill", SC.muted).text(`B  (width ${ST.fmt(b, 2)})`);
      gLev.append("text").attr("x", X0 + (b + (1 - b) / 2) * RW).attr("y", Y0 + RH + 16).attr("text-anchor", "middle")
        .attr("font-size", 11.5).attr("fill", SC.muted).text(`Bᶜ  (width ${ST.fmt(1 - b, 2)})`);
    } else {
      gLev.append("text").attr("x", X0 + RW / 2).attr("y", Y0 + RH + 16).attr("text-anchor", "middle")
        .attr("font-size", 11.5).attr("fill", SC.muted)
        .text(mode === "B" ? "only B is left, rescaled by 1/P(B): proportions inside B are unchanged"
          : `only A is left, rescaled by 1/P(A): the B part is now P(B | A) = ${ST.fmt(bGivenA, 3)} wide`);
    }

    // side panel: the weighted-average scale
    gSide.selectAll("*").remove();
    const sx = X0 + RW + 50, sy = d3.scaleLinear().domain([0, 1]).range([Y0 + RH, Y0]);
    gSide.append("line").attr("x1", sx).attr("x2", sx).attr("y1", sy(0)).attr("y2", sy(1)).attr("stroke", SC.line).attr("stroke-width", 2);
    [0, 0.5, 1].forEach(v => gSide.append("text").attr("x", sx - 8).attr("y", sy(v) + 4).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", SC.muted).text(v));
    const marks = [
      { v: ab, lab: "P(A|B)", c: SC.accent }, { v: anb, lab: "P(A|Bᶜ)", c: SC.accent },
      { v: pA, lab: "P(A)", c: SC.a2 }, { v: bGivenA, lab: "P(B|A)", c: SC.good }
    ];
    marks.sort((p, q) => q.v - p.v);
    let lastY = -1e9;
    marks.forEach(m => {
      if (!isFinite(m.v)) return;
      let ty = sy(m.v) + 4; if (ty - lastY < 13) ty = lastY + 13; lastY = ty;
      gSide.append("circle").attr("cx", sx).attr("cy", sy(m.v)).attr("r", 4.5).attr("fill", m.c);
      gSide.append("text").attr("x", sx + 10).attr("y", ty).attr("font-size", 11).attr("fill", m.c)
        .text(`${m.lab} ${ST.fmt(m.v, 3)}`);
    });

    out.innerHTML = `P(A) = P(A|B)·P(B) + P(A|Bᶜ)·P(Bᶜ) = ${ST.fmt(ab, 2)}·${ST.fmt(b, 2)} + ${ST.fmt(anb, 2)}·${ST.fmt(1 - b, 2)} = <b>${ST.fmt(pA, 4)}</b>` +
      ` — a weighted average of ${ST.fmt(ab, 2)} and ${ST.fmt(anb, 2)}` +
      `<br>P(B | A) = P(A∩B)/P(A) = ${ST.fmt(pAB, 4)}/${ST.fmt(pA, 4)} = <b>${ST.fmt(bGivenA, 4)}</b>` +
      ` · compare P(A | B) = <b>${ST.fmt(ab, 4)}</b> — the bar does not commute` +
      (Math.abs(ab - anb) < 1e-9 ? "<br>P(A|B) = P(A|Bᶜ): A is independent of B, and conditioning on B changes nothing" : "");
  }

  ids.forEach(e => e.addEventListener("input", () => draw(false)));
  document.getElementById("ms-none").addEventListener("click", () => { mode = "none"; draw(true); });
  document.getElementById("ms-condb").addEventListener("click", () => { mode = "B"; draw(true); });
  document.getElementById("ms-conda").addEventListener("click", () => { mode = "A"; draw(true); });
  draw(false);
})();

/* ─────────────────── 5 · the diagnostic-test tree ───────────────────
   Expected counts per 10,000 people. Caption audit: the default (prevalence 1%, sens 90%,
   spec 91%) shows leaves 90 / 10 / 891 / 9,009 and PPV = 90/981 = 0.0917, NPV = 0.9989,
   matching the worked example exactly (recomputed in Python). Repeat positives use the
   odds form, assuming conditional independence of the repeats.                          */
PF.PREV = [0.001, 0.002, 0.005, 0.0075, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9];
PF.count = v => Math.abs(v - Math.round(v)) < 1e-6 ? d3.format(",")(Math.round(v)) : d3.format(",.1f")(v);
(function () {
  const svg = d3.select("#bayes-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 680, H = 380, N = 10000;
  const pr = document.getElementById("by-prev"), se = document.getElementById("by-sens"),
    sp = document.getElementById("by-spec"), kSel = document.getElementById("by-k");
  const prv = document.getElementById("by-prevv"), sev = document.getElementById("by-sensv"),
    spv = document.getElementById("by-specv"), out = document.getElementById("bayes-readout");
  const pctS = v => (v < 0.01 ? ST.fmt(100 * v, 2) : ST.fmt(100 * v, 1)) + "%";

  function draw() {
    const prev = PF.PREV[+pr.value], sens = +se.value, spec = +sp.value, k = +kSel.value;
    prv.textContent = pctS(prev); sev.textContent = ST.fmt(100 * sens, 1) + "%"; spv.textContent = ST.fmt(100 * spec, 1) + "%";
    const D = N * prev, Hh = N - D, TP = D * sens, FN = D - TP, FP = Hh * (1 - spec), TN = Hh - FP;
    const ppv = TP / (TP + FP), npv = TN / (TN + FN);

    svg.selectAll("*").remove();
    const wScale = d3.scaleLinear().domain([0, N]).range([1.2, 30]);
    const nodes = {
      root: { x: 74, y: 190, t: "10,000 people", v: N, c: SC.ink },
      D: { x: 250, y: 95, t: "disease", v: D, c: SC.bad },
      Hh: { x: 250, y: 285, t: "healthy", v: Hh, c: SC.good },
      TP: { x: 430, y: 45, t: "test + (true +)", v: TP, c: SC.bad },
      FN: { x: 430, y: 140, t: "test − (false −)", v: FN, c: SC.muted },
      FP: { x: 430, y: 240, t: "test + (false +)", v: FP, c: SC.a2 },
      TN: { x: 430, y: 335, t: "test − (true −)", v: TN, c: SC.muted }
    };
    const edges = [
      ["root", "D", `prevalence ${pctS(prev)}`], ["root", "Hh", `${pctS(1 - prev)}`],
      ["D", "TP", `sens ${ST.fmt(100 * sens, 1)}%`], ["D", "FN", `${ST.fmt(100 * (1 - sens), 1)}%`],
      ["Hh", "FP", `1 − spec = ${ST.fmt(100 * (1 - spec), 1)}%`], ["Hh", "TN", `spec ${ST.fmt(100 * spec, 1)}%`]
    ];
    const gE = svg.append("g"), gN = svg.append("g");
    edges.forEach(([a, b, lab]) => {
      const A = nodes[a], B = nodes[b];
      gE.append("path").attr("d", `M${A.x + 52},${A.y} C${(A.x + B.x) / 2 + 20},${A.y} ${(A.x + B.x) / 2 - 10},${B.y} ${B.x - 58},${B.y}`)
        .attr("fill", "none").attr("stroke", B.c).attr("stroke-opacity", 0.45).attr("stroke-width", wScale(B.v));
      // label beside the parent end of the branch, where the curve is still horizontal
      const half = d3.max(edges.filter(e => e[0] === a), e => wScale(nodes[e[1]].v)) / 2;   // clear the widest sibling band
      gE.append("text").attr("x", A.x + 62).attr("y", A.y + (B.y < A.y ? -(half + 5) : half + 13))
        .attr("text-anchor", "start").attr("font-size", 10).attr("fill", SC.muted).text(lab);
    });
    Object.entries(nodes).forEach(([key, n]) => {
      const g = gN.append("g").attr("transform", `translate(${n.x},${n.y})`);
      g.append("rect").attr("x", -56).attr("y", -19).attr("width", 112).attr("height", 38).attr("rx", 7)
        .attr("fill", SC.panel2).attr("stroke", n.c).attr("stroke-width", key === "TP" || key === "FP" ? 2 : 1);
      g.append("text").attr("y", -3).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", SC.muted).text(n.t);
      g.append("text").attr("y", 13).attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 700)
        .attr("fill", n.c).text(PF.count(n.v));
    });

    // right panel: the positives, split
    const px = 520, gP = svg.append("g");
    gP.append("text").attr("x", px).attr("y", 30).attr("font-size", 11.5).attr("fill", SC.ink).attr("font-weight", 600)
      .text(`all ${PF.count(TP + FP)} positives`);
    const bh = 230, by = 42, fT = TP / (TP + FP);
    gP.append("rect").attr("x", px).attr("y", by).attr("width", 34).attr("height", bh * (1 - fT)).attr("fill", SC.a2).attr("opacity", 0.8);
    gP.append("rect").attr("x", px).attr("y", by + bh * (1 - fT)).attr("width", 34).attr("height", bh * fT).attr("fill", SC.bad).attr("opacity", 0.9);
    gP.append("text").attr("x", px + 42).attr("y", by + 14).attr("font-size", 10.5).attr("fill", SC.a2).text(`${PF.count(FP)} false +`);
    gP.append("text").attr("x", px + 42).attr("y", by + bh - 4).attr("font-size", 10.5).attr("fill", SC.bad).text(`${PF.count(TP)} true +`);
    gP.append("text").attr("x", px).attr("y", by + bh + 26).attr("font-size", 12).attr("fill", SC.ink)
      .text("PPV = P(D | +)");
    gP.append("text").attr("x", px).attr("y", by + bh + 46).attr("font-size", 17).attr("font-weight", 700).attr("fill", SC.bad)
      .text(ST.pct(ppv, 1));
    gP.append("text").attr("x", px).attr("y", by + bh + 66).attr("font-size", 11).attr("fill", SC.muted)
      .text(`NPV = P(healthy | −) = ${ST.pct(npv, 2)}`);

    // repeat positives by the odds form
    const lr = sens / (1 - spec);
    let odds = prev / (1 - prev);
    const chain = [prev];
    for (let i = 0; i < k; i++) { odds *= lr; chain.push(odds / (1 + odds)); }
    out.innerHTML = `P(+) = ${ST.fmt((TP + FP) / N, 4)} · PPV = ${PF.count(TP)}/${PF.count(TP + FP)} = <b>${ST.fmt(ppv, 4)}</b>` +
      ` · NPV = ${PF.count(TN)}/${PF.count(TN + FN)} = <b>${ST.fmt(npv, 4)}</b>` +
      `<br>LR⁺ = sens/(1 − spec) = <b>${ST.fmt(lr, 2)}</b> · LR⁻ = (1 − sens)/spec = <b>${ST.fmt((1 - sens) / spec, 3)}</b>` +
      `<br>P(disease) after 0…${k} positives in a row: ` + chain.map((c, i) => `${i}: <b>${ST.fmt(c, 4)}</b>`).join(" → ") +
      (k > 1 ? " (assumes the repeats err independently)" : "");
  }
  [pr, se, sp].forEach(e => e.addEventListener("input", draw));
  kSel.addEventListener("change", draw);
  draw();
})();

/* ─────────────────── 6 · pairwise vs mutual independence ───────────────────
   Exact probabilities from the 2 × 2 product model, plus a seeded Monte Carlo of 100,000
   coin pairs as an independent check. Caption audit: default p₁ = p₂ = 0.5, C = XOR
   shows three ✓ pairs and a ✗ triple (0 vs 0.125); p₁ = 0.7, p₂ = 0.5 keeps A ⊥ C and
   breaks B ⊥ C (0.15 vs 0.25), exactly as the caption states.                           */
(function () {
  const svg = d3.select("#indep-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 680, H = 330, X0 = 34, Y0 = 34, S = 250, MC = 100000;
  const p1In = document.getElementById("ind-p1"), p2In = document.getElementById("ind-p2");
  const p1v = document.getElementById("ind-p1v"), p2v = document.getElementById("ind-p2v");
  const cSel = document.getElementById("ind-c"), out = document.getElementById("indep-readout");

  function draw() {
    const p1 = +p1In.value, p2 = +p2In.value, mode = cSel.value;
    p1v.textContent = ST.fmt(p1, 2); p2v.textContent = ST.fmt(p2, 2);
    const inC = (a, b) => mode === "xor" ? a !== b : a === b;    // a, b: true = heads
    const cells = [];
    [true, false].forEach(a => [true, false].forEach(b => cells.push({
      a, b, name: (a ? "H" : "T") + (b ? "H" : "T"),
      p: (a ? p1 : 1 - p1) * (b ? p2 : 1 - p2), c: inC(a, b)
    })));
    const P = f => d3.sum(cells.filter(f), d => d.p);
    const pA = P(d => d.a), pB = P(d => d.b), pC = P(d => d.c);
    const rows = [
      { lab: "A, B", j: P(d => d.a && d.b), pr: pA * pB, f: (a, b, c) => a && b },
      { lab: "A, C", j: P(d => d.a && d.c), pr: pA * pC, f: (a, b, c) => a && c },
      { lab: "B, C", j: P(d => d.b && d.c), pr: pB * pC, f: (a, b, c) => b && c },
      { lab: "A, B, C", j: P(d => d.a && d.b && d.c), pr: pA * pB * pC, f: (a, b, c) => a && b && c }
    ];
    // Monte Carlo check, fixed seed so the numbers are stable
    const r = ST.rng(5), hits = [0, 0, 0, 0];
    for (let i = 0; i < MC; i++) {
      const a = r() < p1, b = r() < p2, c = inC(a, b);
      rows.forEach((row, k) => { if (row.f(a, b, c)) hits[k]++; });
    }
    rows.forEach((row, k) => { row.mc = hits[k] / MC; row.ok = Math.abs(row.j - row.pr) < 1e-12; });

    svg.selectAll("*").remove();
    const x = a => a ? X0 : X0 + p1 * S, w = a => (a ? p1 : 1 - p1) * S;
    const y = b => b ? Y0 : Y0 + p2 * S, h = b => (b ? p2 : 1 - p2) * S;
    svg.append("text").attr("x", X0).attr("y", Y0 - 18).attr("font-size", 11).attr("fill", SC.muted)
      .text("toss 1: H | T  →   (columns)");
    svg.append("text").attr("x", X0 - 22).attr("y", Y0 + S / 2).attr("font-size", 11).attr("fill", SC.muted)
      .attr("transform", `rotate(-90 ${X0 - 22} ${Y0 + S / 2})`).attr("text-anchor", "middle").text("toss 2: H (top) | T");
    cells.forEach(d => {
      svg.append("rect").attr("x", x(d.a)).attr("y", y(d.b)).attr("width", w(d.a)).attr("height", h(d.b))
        .attr("fill", d.c ? SC.good : SC.panel2).attr("fill-opacity", d.c ? 0.35 : 1).attr("stroke", SC.bg).attr("stroke-width", 2);
      if (w(d.a) > 34 && h(d.b) > 30) {
        svg.append("text").attr("x", x(d.a) + w(d.a) / 2).attr("y", y(d.b) + h(d.b) / 2 - 3).attr("text-anchor", "middle")
          .attr("font-size", 13).attr("font-weight", 700).attr("fill", SC.ink).text(d.name);
        svg.append("text").attr("x", x(d.a) + w(d.a) / 2).attr("y", y(d.b) + h(d.b) / 2 + 13).attr("text-anchor", "middle")
          .attr("font-size", 11).attr("fill", SC.muted).text(ST.fmt(d.p, 3));
      }
    });
    svg.append("rect").attr("x", X0 + 2).attr("y", Y0 + 2).attr("width", p1 * S - 4).attr("height", S - 4)
      .attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 3);
    svg.append("rect").attr("x", X0 + 6).attr("y", Y0 + 6).attr("width", S - 12).attr("height", p2 * S - 12)
      .attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 3).attr("stroke-dasharray", "8 4");
    ST.legend(svg, [
      { label: "A: toss 1 = H", color: SC.accent }, { label: "B: toss 2 = H", color: SC.a2 },
      { label: mode === "xor" ? "C: exactly one H" : "C: tosses agree", color: SC.good, op: 0.5 }
    ], X0, Y0 + S + 20, { vertical: false, gap: 17 });

    // checklist
    const tx = 330, g = svg.append("g").attr("transform", `translate(${tx},${Y0})`);
    g.append("text").attr("x", 0).attr("y", 0).attr("font-size", 12).attr("fill", SC.ink)
      .text(`P(A) = ${ST.fmt(pA, 3)}   P(B) = ${ST.fmt(pB, 3)}   P(C) = ${ST.fmt(pC, 3)}`);
    const cols = [0, 70, 150, 230, 312];
    ["events", "P(joint)", "product", "simulated", ""].forEach((t, i) =>
      g.append("text").attr("x", cols[i]).attr("y", 30).attr("font-size", 11).attr("fill", SC.muted).text(t));
    rows.forEach((row, k) => {
      const yy = 56 + k * 30;
      g.append("rect").attr("x", -6).attr("y", yy - 17).attr("width", 342).attr("height", 26).attr("rx", 5)
        .attr("fill", row.ok ? SC.good : SC.bad).attr("opacity", 0.1);
      g.append("text").attr("x", cols[0]).attr("y", yy).attr("font-size", 12.5).attr("fill", SC.ink).text(row.lab);
      g.append("text").attr("x", cols[1]).attr("y", yy).attr("font-size", 12.5).attr("fill", SC.ink).text(ST.fmt(row.j, 4));
      g.append("text").attr("x", cols[2]).attr("y", yy).attr("font-size", 12.5).attr("fill", SC.ink).text(ST.fmt(row.pr, 4));
      g.append("text").attr("x", cols[3]).attr("y", yy).attr("font-size", 12.5).attr("fill", SC.muted).text(ST.fmt(row.mc, 4));
      g.append("text").attr("x", cols[4]).attr("y", yy).attr("font-size", 14).attr("font-weight", 700)
        .attr("fill", row.ok ? SC.good : SC.bad).text(row.ok ? "✓" : "✗");
    });
    const pairsOk = rows.slice(0, 3).every(r => r.ok), mutual = rows.every(r => r.ok);
    const verdict = mutual ? "mutually independent" : pairsOk ? "pairwise but NOT mutually independent" : "not even pairwise independent";
    g.append("text").attr("x", 0).attr("y", 186).attr("font-size", 12).attr("font-weight", 600)
      .attr("fill", mutual ? SC.good : pairsOk ? SC.a2 : SC.bad).text(`A, B, C are ${verdict}`);
    g.append("text").attr("x", 0).attr("y", 206).attr("font-size", 10.5).attr("fill", SC.muted)
      .text(`simulated = frequency in ${d3.format(",")(MC)} seeded coin pairs (sd ≤ ${ST.fmt(0.5 / Math.sqrt(MC), 4)})`);

    out.innerHTML = rows.map(r => `${r.lab}: joint <b>${ST.fmt(r.j, 4)}</b> vs product <b>${ST.fmt(r.pr, 4)}</b>`).join(" · ") +
      `<br>verdict: <b>${verdict}</b>`;
  }
  [p1In, p2In].forEach(e => e.addEventListener("input", draw));
  cSel.addEventListener("change", draw);
  draw();
})();

/* ─────────────────── 7 · reliability: series, parallel, k-of-n ───────────────────
   k-of-n uses the binomial tail from the shared library (ST.binomCdf). Caption audit:
   default n = 3, k = 2, p = 0.9 reads series 0.729, parallel 0.999, 2-of-3 0.972, and
   the 2-of-3 curve crosses the diagonal at p = 0.5 (value 0.5000), as the caption says.  */
(function () {
  const svg = d3.select("#rel-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 680, H = 360;
  const pIn = document.getElementById("rl-p"), nIn = document.getElementById("rl-n"), kIn = document.getElementById("rl-k");
  const pv = document.getElementById("rl-pv"), nv = document.getElementById("rl-nv"), kv = document.getElementById("rl-kv");
  const out = document.getElementById("rel-readout");
  const col = { series: SC.bad, parallel: SC.good, kofn: SC.a2, single: SC.muted };

  function draw() {
    const n = +nIn.value;
    kIn.max = n; if (+kIn.value > n) kIn.value = n;
    const p = +pIn.value, k = +kIn.value;
    pv.textContent = ST.fmt(p, 2); nv.textContent = n; kv.textContent = k;
    const f = {
      series: q => Math.pow(q, n),
      parallel: q => 1 - Math.pow(1 - q, n),
      kofn: q => PF.kOfN(k, n, q),
      single: q => q
    };

    svg.selectAll("*").remove();
    const F = { g: svg.append("g").attr("transform", "translate(50,16)"), iw: 340, ih: 300 };
    const x = d3.scaleLinear().domain([0, 1]).range([0, F.iw]), y = d3.scaleLinear().domain([0, 1]).range([F.ih, 0]);
    ST.gridY(F.g, y, F.iw, 5); ST.gridX(F.g, x, F.ih, 5);
    ST.axisB(F.g, x, F.ih, 5, "component reliability p");
    ST.axisL(F.g, y, 5, "system reliability R");
    const qs = d3.range(0, 1.0001, 0.005);
    Object.keys(f).forEach(key => {
      F.g.append("path").datum(qs).attr("fill", "none").attr("stroke", col[key])
        .attr("stroke-width", key === "single" ? 1.2 : 2.2).attr("stroke-dasharray", key === "single" ? "4 4" : null)
        .attr("d", d3.line().x(q => x(q)).y(q => y(f[key](q))));
    });
    F.g.append("line").attr("x1", x(p)).attr("x2", x(p)).attr("y1", 0).attr("y2", F.ih).attr("stroke", SC.ink).attr("opacity", 0.35);
    ["series", "parallel", "kofn"].forEach(key => F.g.append("circle").attr("cx", x(p)).attr("cy", y(f[key](p)))
      .attr("r", 4.5).attr("fill", col[key]).attr("stroke", SC.bg));
    ST.legend(F.g, [
      { label: `parallel (1-of-${n})`, color: col.parallel }, { label: `${k}-of-${n} vote`, color: col.kofn },
      { label: `series (${n}-of-${n})`, color: col.series }, { label: "one component", color: col.single, dash: "4 4" }
    ], 12, 14);

    // block diagrams
    const gx = 430, gB = svg.append("g");
    const box = (gg, bx, by, s) => gg.append("rect").attr("x", bx).attr("y", by).attr("width", s).attr("height", s).attr("rx", 3)
      .attr("fill", SC.panel2).attr("stroke", SC.ink).attr("stroke-opacity", 0.6);
    const s = Math.min(16, 200 / n - 4);
    // series
    gB.append("text").attr("x", gx).attr("y", 26).attr("font-size", 11.5).attr("fill", col.series).text(`series: R = p^${n} = ${ST.fmt(f.series(p), 4)}`);
    gB.append("line").attr("x1", gx).attr("x2", gx + 230).attr("y1", 46).attr("y2", 46).attr("stroke", col.series);
    for (let i = 0; i < n; i++) box(gB, gx + 10 + i * (230 - 20) / n, 46 - s / 2, s);
    // parallel
    const py0 = 82, ph = Math.min(18, 120 / n);
    gB.append("text").attr("x", gx).attr("y", py0).attr("font-size", 11.5).attr("fill", col.parallel)
      .text(`parallel: R = 1 − (1 − p)^${n} = ${ST.fmt(f.parallel(p), 4)}`);
    for (let i = 0; i < n; i++) {
      const yy = py0 + 14 + i * ph + ph / 2;
      gB.append("line").attr("x1", gx + 40).attr("x2", gx + 190).attr("y1", yy).attr("y2", yy).attr("stroke", col.parallel);
      box(gB, gx + 107, yy - Math.min(ph - 3, 14) / 2, Math.min(ph - 3, 14));
    }
    gB.append("line").attr("x1", gx + 40).attr("x2", gx + 40).attr("y1", py0 + 14 + ph / 2).attr("y2", py0 + 14 + (n - 0.5) * ph).attr("stroke", col.parallel);
    gB.append("line").attr("x1", gx + 190).attr("x2", gx + 190).attr("y1", py0 + 14 + ph / 2).attr("y2", py0 + 14 + (n - 0.5) * ph).attr("stroke", col.parallel);
    // k-of-n
    const ky = py0 + 14 + n * ph + 26;
    gB.append("text").attr("x", gx).attr("y", ky).attr("font-size", 11.5).attr("fill", col.kofn)
      .text(`${k}-of-${n} vote: R = P(at least ${k} work) = ${ST.fmt(f.kofn(p), 4)}`);
    for (let i = 0; i < n; i++) {
      const bx = gx + 10 + i * (200 - 20) / n;
      box(gB, bx, ky + 14, s);
      gB.append("line").attr("x1", bx + s / 2).attr("x2", gx + 215).attr("y1", ky + 14 + s).attr("y2", ky + 50).attr("stroke", col.kofn).attr("opacity", 0.5);
    }
    gB.append("circle").attr("cx", gx + 222).attr("cy", ky + 52).attr("r", 13).attr("fill", SC.panel2).attr("stroke", col.kofn);
    gB.append("text").attr("x", gx + 222).attr("y", ky + 56).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", col.kofn).text(`≥${k}`);

    // where does k-of-n cross the diagonal? (bisection on R(p) − p, ignoring the trivial ends)
    let cross = NaN;
    if (k > 1 && k < n) {
      let lo = 1e-6, hi = 1 - 1e-6;
      const g = q => f.kofn(q) - q;
      if (g(lo) * g(hi) < 0) {
        for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (g(lo) * g(m) <= 0) hi = m; else lo = m; }
        cross = (lo + hi) / 2;
      }
    }
    out.innerHTML = `p = ${ST.fmt(p, 2)}, n = ${n}: series <b>${ST.fmt(f.series(p), 4)}</b> · parallel <b>${ST.fmt(f.parallel(p), 4)}</b> · ` +
      `${k}-of-${n} <b>${ST.fmt(f.kofn(p), 4)}</b> · one component ${ST.fmt(p, 4)}` +
      (isFinite(cross) ? `<br>the ${k}-of-${n} curve crosses the diagonal at p = <b>${ST.fmt(cross, 4)}</b>: above it voting helps, below it voting hurts` : "") +
      `<br>failure probability: series ${ST.sig(1 - f.series(p), 3)} · parallel ${ST.sig(1 - f.parallel(p), 3)} · one component ${ST.sig(1 - p, 3)}`;
  }
  [pIn, nIn, kIn].forEach(e => e.addEventListener("input", draw));
  draw();
})();

/* ─────────────────── 8 · Monty Hall ───────────────────
   PF.monty plays n games from one seeded stream. Each game: car uniform on 3 doors,
   contestant's pick uniform, then the host opens a goat door other than the pick —
   at random between two goats if the pick was the car (host who knows), or a uniformly
   random other door (host who does not know; games where he reveals the car are void).
   Both strategies are scored on the SAME games, so in "knows" mode stay + switch = 1.
   Caption audit (40 seeds scanned): seed 6 gives stay 331 / switch 669 of 1,000 with the
   knowing host, and 511 / 489 with the ignorant host (508 of 1,508 games void, ≈ 1/3 as
   theory predicts). The default draws all 1,000 games statically so the converged state
   is what the reader sees first.                                                        */
PF.monty = function (n, seed, knows) {
  const r = ST.rng(seed), games = [];
  let played = 0;
  while (games.length < n && played < 20 * n) {
    played++;
    const car = Math.floor(r() * 3), pick = Math.floor(r() * 3);
    const others = [0, 1, 2].filter(d => d !== pick);
    let open;
    if (knows) {
      const goats = others.filter(d => d !== car);
      open = goats[goats.length === 1 ? 0 : Math.floor(r() * 2)];
    } else {
      open = others[Math.floor(r() * 2)];
      if (open === car) continue;                     // car revealed: game void
    }
    const sw = [0, 1, 2].find(d => d !== pick && d !== open);
    games.push({ car, pick, open, sw, stayWin: pick === car, switchWin: sw === car });
  }
  return { games, played };
};
(function () {
  const svg = d3.select("#monty-svg");
  if (svg.empty() || typeof ST === "undefined") return;

  const W = 680, H = 360, N = 1000;
  const modeSel = document.getElementById("mh-mode"), out = document.getElementById("monty-readout");
  let seed = PF.seeds.monty, sim = null, shown = 0, timer = null;

  svg.selectAll("*").remove();
  const gDoors = svg.append("g").attr("transform", "translate(18,20)");
  const F = { g: svg.append("g").attr("transform", "translate(250,20)"), iw: 400, ih: 290 };
  const x = d3.scaleLog().domain([1, N]).range([0, F.iw]), y = d3.scaleLinear().domain([0, 1]).range([F.ih, 0]);
  ST.gridY(F.g, y, F.iw, 6);
  F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`)
    .call(d3.axisBottom(x).tickValues([1, 10, 100, 1000]).tickFormat(d3.format(",")));
  F.g.append("text").attr("x", F.iw).attr("y", F.ih + 32).attr("text-anchor", "end").attr("font-size", 11)
    .attr("fill", SC.muted).text("games played (log scale)");
  ST.axisL(F.g, y, 6, "running win rate");
  const gRef = F.g.append("g"), gLines = F.g.append("g");
  const pStay = gLines.append("path").attr("fill", "none").attr("stroke", SC.accent).attr("stroke-width", 2);
  const pSw = gLines.append("path").attr("fill", "none").attr("stroke", SC.a2).attr("stroke-width", 2);
  ST.legend(F.g, [{ label: "always stay", color: SC.accent }, { label: "always switch", color: SC.a2 }], F.iw - 120, F.ih - 34);

  function refLines() {
    gRef.selectAll("*").remove();
    const knows = modeSel.value === "knows";
    const refs = knows ? [[1 / 3, "1/3"], [2 / 3, "2/3"]] : [[0.5, "1/2"]];
    refs.forEach(([v, t]) => {
      gRef.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(v)).attr("y2", y(v))
        .attr("stroke", SC.good).attr("stroke-dasharray", "5 4");
      gRef.append("text").attr("x", 4).attr("y", y(v) - 5).attr("font-size", 11).attr("fill", SC.good).text(t);
    });
  }

  function drawDoors(gm, idx) {
    gDoors.selectAll("*").remove();
    gDoors.append("text").attr("x", 0).attr("y", 0).attr("font-size", 11.5).attr("fill", SC.muted)
      .text(gm ? `game ${d3.format(",")(idx)}` : "");
    if (!gm) return;
    for (let d = 0; d < 3; d++) {
      const gx = d * 72, g = gDoors.append("g").attr("transform", `translate(${gx},14)`);
      const opened = d === gm.open, picked = d === gm.pick;
      g.append("rect").attr("width", 62).attr("height", 96).attr("rx", 6)
        .attr("fill", opened ? SC.bg : SC.panel2).attr("stroke", picked ? SC.accent : d === gm.sw ? SC.a2 : SC.line)
        .attr("stroke-width", picked || d === gm.sw ? 3 : 1);
      g.append("text").attr("x", 31).attr("y", 22).attr("text-anchor", "middle").attr("font-size", 12)
        .attr("fill", SC.ink).text(`door ${d + 1}`);
      g.append("text").attr("x", 31).attr("y", 62).attr("text-anchor", "middle").attr("font-size", opened ? 13 : 12)
        .attr("fill", d === gm.car ? SC.good : SC.muted).text(opened ? "goat" : d === gm.car ? "car" : "goat")
        .attr("opacity", opened ? 1 : 0.55);
      g.append("text").attr("x", 31).attr("y", 112).attr("text-anchor", "middle").attr("font-size", 10.5)
        .attr("fill", picked ? SC.accent : d === gm.sw ? SC.a2 : SC.muted)
        .text(picked ? "first pick" : opened ? "host opens" : "switch to");
    }
    gDoors.append("text").attr("x", 0).attr("y", 150).attr("font-size", 11.5).attr("fill", SC.ink)
      .text(`stay ${gm.stayWin ? "wins" : "loses"} · switch ${gm.switchWin ? "wins" : "loses"}`);
    // tallies
    const g = sim.games.slice(0, shown);
    const st = d3.sum(g, d => d.stayWin), sw = d3.sum(g, d => d.switchWin);
    const bars = [["stay", st, SC.accent], ["switch", sw, SC.a2]];
    bars.forEach(([lab, v, c], i) => {
      const by = 172 + i * 46;
      gDoors.append("text").attr("x", 0).attr("y", by).attr("font-size", 11).attr("fill", c).text(`${lab}: ${v} wins of ${shown}`);
      gDoors.append("rect").attr("x", 0).attr("y", by + 6).attr("width", 210).attr("height", 14).attr("fill", SC.panel2);
      gDoors.append("rect").attr("x", 0).attr("y", by + 6).attr("width", 210 * (shown ? v / shown : 0)).attr("height", 14).attr("fill", c).attr("opacity", 0.85);
    });
  }

  function render() {
    const g = sim.games.slice(0, shown);
    let s = 0, w = 0;
    const st = [], sw = [];
    g.forEach((d, i) => { s += d.stayWin; w += d.switchWin; st.push(s / (i + 1)); sw.push(w / (i + 1)); });
    const line = d3.line().x((d, i) => x(i + 1)).y(d => y(d));
    pStay.attr("d", st.length ? line(st) : null);
    pSw.attr("d", sw.length ? line(sw) : null);
    drawDoors(g[g.length - 1], g.length);
    const knows = modeSel.value === "knows";
    out.innerHTML = `seed ${seed} · ${d3.format(",")(shown)} games` +
      (knows ? "" : ` (${d3.format(",")(sim.played)} played; ${d3.format(",")(sim.played - sim.games.length)} void because the host revealed the car)`) +
      `<br>stay wins <b>${s}</b> (${ST.pct(shown ? s / shown : NaN, 1)}) · switch wins <b>${w}</b> (${ST.pct(shown ? w / shown : NaN, 1)})` +
      ` · theory: ${knows ? "1/3 vs 2/3" : "1/2 vs 1/2"}` +
      `<br>±2 sd for a proportion over ${d3.format(",")(shown || 1)} games ≈ ±${ST.pct(2 * Math.sqrt((knows ? 2 / 9 : 0.25) / Math.max(shown, 1)), 1)}`;
  }

  function run(animate) {
    if (timer) { timer.stop(); timer = null; }
    sim = PF.monty(N, seed, modeSel.value === "knows");
    refLines();
    if (!animate) { shown = N; render(); return; }
    shown = 0;
    timer = d3.timer(el => {
      shown = Math.min(N, Math.max(1, Math.round(Math.pow(N, Math.min(1, el / 2600)))));
      render();
      if (shown >= N) { timer.stop(); timer = null; }
    });
  }

  document.getElementById("mh-run").addEventListener("click", () => { seed = (seed * 48271 + 7) % 2147483647; run(true); });
  document.getElementById("mh-default").addEventListener("click", () => { seed = PF.seeds.monty; run(false); });
  modeSel.addEventListener("change", () => run(false));
  run(false);
})();
