/* boltzmann-machines-rbms.viz.js — the visualizations on
   deep-learning/classical-nn/boltzmann-machines-rbms.html
   (Classical Neural Networks · part 3). Loaded after ../../data.js → ../../notes.js.
   Each block is an IIFE that exits quietly if its <svg> is not on the page.

     1  #bt-svg    temperature: energies → Boltzmann probabilities, with a Gibbs sampler
     2  #gc-svg    a tiny RBM (4 visible, 3 hidden): block Gibbs chain against exact p(v)
     3  #cd-svg    CD-k / PCD training on 4×4 bars-and-stripes, exact log-likelihood
     4  #gb-svg    a Gaussian–Bernoulli RBM on 2-D data: a mixture of 2ᴴ Gaussians
     5  #dbn-svg   greedy layer-wise stacking into a deep belief network, step by step
     6  #lv-svg    Langevin dynamics: sampling from e^(−E/T) with the score, no Z needed

   Every number the figures print is recomputed from the state they draw.   */

/* ══════════ page-local helpers ══════════ */
const RB = (function () {
  const mono = "SF Mono, Menlo, monospace", OFF = "#1b1f29";
  function rng(seed) {
    let a = (seed >>> 0) || 1;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  const sig = x => 1 / (1 + Math.exp(-x));
  const softplus = x => x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x));
  const fmt = (x, d) => (d === undefined ? String(x) : x.toFixed(d)).replace("-", "−");
  function lse(arr) { const m = Math.max.apply(null, arr); let s = 0; for (const v of arr) s += Math.exp(v - m); return m + Math.log(s); }
  function txt(g, x, y, s, o) {
    o = o || {};
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", o.size || 11).attr("fill", o.fill || C.ink)
      .attr("font-weight", o.bold ? 600 : null).attr("text-anchor", o.anchor || null)
      .attr("font-family", o.mono ? mono : null).text(s);
  }
  /* an image of values in [0,1] (or [−1,1] with diverging=true) */
  function img(g, vals, x, y, cell, opt) {
    opt = opt || {};
    const side = opt.side || Math.round(Math.sqrt(vals.length));
    const col = opt.div ? (d => d3.interpolateRdBu(0.5 - 0.5 * Math.max(-1, Math.min(1, d)))) : (d => d3.interpolateRgb(OFF, opt.on || C.A)(Math.max(0, Math.min(1, d))));
    const G = g.append("g").attr("transform", `translate(${x},${y})`);
    G.selectAll("rect").data(vals).join("rect")
      .attr("x", (d, i) => (i % side) * cell).attr("y", (d, i) => Math.floor(i / side) * cell)
      .attr("width", cell - 1).attr("height", cell - 1).attr("fill", col);
    if (opt.frame) G.append("rect").attr("x", -2).attr("y", -2).attr("width", side * cell + 3).attr("height", Math.ceil(vals.length / side) * cell + 3)
      .attr("fill", "none").attr("stroke", opt.frame).attr("rx", 2);
    return G;
  }
  /* 4×4 bars-and-stripes: 16 row patterns + 16 column patterns − 2 duplicates = 30 */
  function bas() {
    const out = [], seen = new Set();
    for (let m = 0; m < 16; m++) for (const o of [0, 1]) {
      const v = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) v.push((m >> (o ? c : r)) & 1);
      const k = v.join(""); if (!seen.has(k)) { seen.add(k); out.push(v); }
    }
    return out;
  }
  /* a binary RBM: W is nv × nh, row-major */
  function RBM(nv, nh, seed) {
    const r = rng(seed || 1);
    const m = { nv, nh, r, W: new Float64Array(nv * nh), a: new Float64Array(nv), b: new Float64Array(nh) };
    m.ph = v => { const p = new Float64Array(nh); for (let j = 0; j < nh; j++) { let z = m.b[j]; for (let i = 0; i < nv; i++) if (v[i]) z += v[i] * m.W[i * nh + j]; p[j] = sig(z); } return p; };
    m.pv = h => { const p = new Float64Array(nv); for (let i = 0; i < nv; i++) { let z = m.a[i]; for (let j = 0; j < nh; j++) if (h[j]) z += m.W[i * nh + j] * h[j]; p[i] = sig(z); } return p; };
    m.smp = p => Array.from(p, x => r() < x ? 1 : 0);
    m.F = v => { let f = 0; for (let i = 0; i < nv; i++) f -= m.a[i] * v[i]; for (let j = 0; j < nh; j++) { let z = m.b[j]; for (let i = 0; i < nv; i++) if (v[i]) z += m.W[i * nh + j]; f -= softplus(z); } return f; };
    m.logZ = () => { const n = 1 << nv, arr = new Array(n); const v = new Array(nv); for (let x = 0; x < n; x++) { for (let i = 0; i < nv; i++) v[i] = (x >> i) & 1; arr[x] = -m.F(v); } return lse(arr); };
    return m;
  }
  const bits = (x, n) => d3.range(n).map(i => (x >> (n - 1 - i)) & 1);
  return { mono, OFF, rng, randn, sig, softplus, fmt, lse, txt, img, bas, RBM, bits };
})();

/* ═════════════════ 1 · #bt-svg — temperature ═════════════════ */
(function () {
  const svg = d3.select("#bt-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id);
  const TE = $("bt-T"), preE = $("bt-pre");
  const PRE = {
    ferro: { w: [2, 2, 2], b: [-1.5, -1.5, -1.5], name: "all pairs agree (w = +2, b = −1.5)" },
    frust: { w: [-1.5, -1.5, -1.5], b: [1, 1, 1], name: "frustrated triangle (w = −1.5, b = +1)" },
    pair: { w: [3, 0, -1], b: [-1.5, -1.5, 0.5], name: "one strong pair, one veto" }
  };
  let emp = null;
  function E(s, p) { return -(p.w[0] * s[0] * s[1] + p.w[1] * s[0] * s[2] + p.w[2] * s[1] * s[2]) - (p.b[0] * s[0] + p.b[1] * s[1] + p.b[2] * s[2]); }
  function draw() {
    const T = Math.pow(10, +TE.value), p = PRE[preE.value];
    $("bt-Tv").textContent = RB.fmt(T, 2);
    const S = d3.range(8).map(x => RB.bits(x, 3)), En = S.map(s => E(s, p));
    const lw = En.map(e => -e / T), lZ = RB.lse(lw), pr = lw.map(v => Math.exp(v - lZ));
    svg.selectAll("*").remove();
    /* energies */
    const x0 = 40, w = 200, y0 = 36, h = 200;
    RB.txt(svg, x0 - 20, 20, "energy E(s) of each state", { bold: true });
    const xs = d3.scaleBand().domain(d3.range(8)).range([x0, x0 + w]).padding(0.2);
    const ey = d3.scaleLinear().domain([Math.min(-1, d3.min(En)) - 0.5, Math.max(1, d3.max(En)) + 0.5]).range([y0 + h, y0]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(ey).ticks(5));
    svg.append("line").attr("x1", x0).attr("x2", x0 + w).attr("y1", ey(0)).attr("y2", ey(0)).attr("stroke", "#3a4150");
    svg.append("g").selectAll("rect").data(En).join("rect").attr("x", (d, i) => xs(i)).attr("width", xs.bandwidth())
      .attr("y", d => Math.min(ey(d), ey(0))).attr("height", d => Math.abs(ey(d) - ey(0))).attr("fill", C.B).attr("fill-opacity", 0.8);
    S.forEach((s, i) => RB.txt(svg, xs(i) + xs.bandwidth() / 2, y0 + h + 14, s.join(""), { size: 9.5, mono: true, anchor: "middle", fill: C.muted }));
    /* probabilities */
    const x1 = 300;
    RB.txt(svg, x1 - 20, 20, "p(s) = e^(−E/T) / Z   (bars exact, dots Gibbs)", { bold: true });
    const xs2 = d3.scaleBand().domain(d3.range(8)).range([x1, x1 + w]).padding(0.2);
    const py = d3.scaleLinear().domain([0, 1]).range([y0 + h, y0]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(${x1},0)`).call(d3.axisLeft(py).ticks(5));
    svg.append("g").selectAll("rect").data(pr).join("rect").attr("x", (d, i) => xs2(i)).attr("width", xs2.bandwidth())
      .attr("y", d => py(d)).attr("height", d => py(0) - py(d)).attr("fill", C.A).attr("fill-opacity", 0.8);
    svg.append("line").attr("x1", x1).attr("x2", x1 + w).attr("y1", py(1 / 8)).attr("y2", py(1 / 8)).attr("stroke", C.muted).attr("stroke-dasharray", "3 3");
    RB.txt(svg, x1 + w + 3, py(1 / 8) + 3, "1/8", { size: 9, fill: C.muted });
    if (emp) svg.append("g").selectAll("circle").data(emp).join("circle").attr("cx", (d, i) => xs2(i) + xs2.bandwidth() / 2).attr("cy", d => py(d)).attr("r", 4).attr("fill", C.good).attr("stroke", "#0f1117");
    S.forEach((s, i) => RB.txt(svg, xs2(i) + xs2.bandwidth() / 2, y0 + h + 14, s.join(""), { size: 9.5, mono: true, anchor: "middle", fill: C.muted }));
    /* the stochastic unit at this T */
    const x2 = 560, w2 = 170;
    RB.txt(svg, x2 - 20, 20, "one unit: p(sᵢ = 1) = σ(Δᵢ / T)", { bold: true });
    const ds = d3.scaleLinear().domain([-6, 6]).range([x2, x2 + w2]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${y0 + h})`).call(d3.axisBottom(ds).ticks(5));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${x2},0)`).call(d3.axisLeft(py).ticks(3));
    svg.append("path").datum(d3.range(-6, 6.01, 0.05)).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2)
      .attr("d", d3.line().x(d => ds(d)).y(d => py(RB.sig(d / T))));
    svg.append("path").datum([[-6, 0], [0, 0], [0, 1], [6, 1]]).attr("fill", "none").attr("stroke", C.muted).attr("stroke-dasharray", "3 3")
      .attr("d", d3.line().x(d => ds(d[0])).y(d => py(d[1])));
    RB.txt(svg, x2 + w2, y0 + h + 30, "energy gap Δᵢ = E(sᵢ=0) − E(sᵢ=1)", { size: 9.5, anchor: "end", fill: C.muted });
    RB.txt(svg, x2 + 4, y0 + 20, "dashed: T → 0 (Hopfield)", { size: 9.5, fill: C.muted });
    const H = -d3.sum(pr, v => v > 0 ? v * Math.log2(v) : 0), emin = d3.min(En);
    const gs = d3.range(8).filter(i => En[i] < emin + 1e-9), pg = d3.sum(gs, i => pr[i]);
    $("bt-readout").innerHTML = `${p.name} · T = ${RB.fmt(T, 2)} · Z = ${RB.fmt(Math.exp(lZ), 3)} · lowest energy ${RB.fmt(emin, 1)} at ${gs.map(i => S[i].join("")).join(", ")}: together p = <b>${RB.fmt(pg, 3)}</b> · entropy <b>${RB.fmt(H, 2)}</b> bits of 3` +
      (emp ? ` · Gibbs total-variation gap ${RB.fmt(0.5 * d3.sum(pr, (v, i) => Math.abs(v - emp[i])), 3)}` : "");
  }
  function gibbs() {
    const T = Math.pow(10, +TE.value), p = PRE[preE.value], r = RB.rng(Math.floor(Math.random() * 1e9));
    const s = [0, 0, 0], cnt = new Array(8).fill(0), n = 4000;
    for (let t = 0; t < n + 200; t++) {
      for (let i = 0; i < 3; i++) {
        const s1 = s.slice(), s0 = s.slice(); s1[i] = 1; s0[i] = 0;
        s[i] = r() < RB.sig((E(s0, p) - E(s1, p)) / T) ? 1 : 0;
      }
      if (t >= 200) cnt[s[0] * 4 + s[1] * 2 + s[2]]++;
    }
    emp = cnt.map(c => c / n); draw();
  }
  TE.addEventListener("input", () => { emp = null; draw(); });
  preE.addEventListener("change", () => { emp = null; draw(); });
  $("bt-gibbs").addEventListener("click", gibbs);
  draw();
})();

/* ═════════════════ 2 · #gc-svg — a tiny RBM's Gibbs chain ═════════════════ */
(function () {
  const svg = d3.select("#gc-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id);
  const kE = $("gc-k");
  const nv = 4, nh = 3;
  /* base parameters: hidden 1 ⇒ "left pair on", hidden 2 ⇒ "right pair on", hidden 3 ⇒ weak "all on" */
  const W0 = [[2, -2, 0.5], [2, -2, 0.5], [-2, 2, 0.5], [-2, 2, 0.5]], a0 = [-1, -1, -1, -1], b0 = [-1, -1, -1];
  let m, v, h, hist, trace, steps, switches, timer = null, phase = 0;
  function build() {
    const k = +kE.value; $("gc-kv").textContent = RB.fmt(k, 2);
    m = RB.RBM(nv, nh, 5);
    for (let i = 0; i < nv; i++) { m.a[i] = k * a0[i]; for (let j = 0; j < nh; j++) m.W[i * nh + j] = k * W0[i][j]; }
    for (let j = 0; j < nh; j++) m.b[j] = k * b0[j];
    v = [1, 1, 0, 0]; h = [0, 0, 0]; hist = new Array(16).fill(0); trace = []; steps = 0; switches = 0; phase = 0;
  }
  const code = x => x[0] * 8 + x[1] * 4 + x[2] * 2 + x[3];
  const modeOf = x => (x[0] + x[1]) > (x[2] + x[3]) ? "L" : (x[0] + x[1]) < (x[2] + x[3]) ? "R" : "·";
  function half() {
    if (phase === 0) { h = m.smp(m.ph(v)); phase = 1; }
    else {
      const prev = modeOf(v); v = m.smp(m.pv(h)); phase = 0; steps++;
      hist[code(v)]++; trace.push(code(v)); if (trace.length > 200) trace.shift();
      const cur = modeOf(v); if (prev !== "·" && cur !== "·" && prev !== cur) switches++;
    }
  }
  function exact() {
    const lw = d3.range(16).map(x => -m.F(RB.bits(x, 4))), lZ = RB.lse(lw);
    return lw.map(t => Math.exp(t - lZ));
  }
  function draw() {
    svg.selectAll("*").remove();
    const vx = i => 50 + i * 70, hx = j => 85 + j * 70, vy = 250, hy = 90;
    RB.txt(svg, 20, 20, "the RBM: blue +, red −, width |w|", { bold: true, size: 11 });
    for (let i = 0; i < nv; i++) for (let j = 0; j < nh; j++) {
      const w = m.W[i * nh + j];
      svg.append("line").attr("x1", vx(i)).attr("y1", vy).attr("x2", hx(j)).attr("y2", hy)
        .attr("stroke", w > 0 ? C.A : C.bad).attr("stroke-opacity", 0.6).attr("stroke-width", Math.min(7, 0.6 + Math.abs(w) * 0.8));
    }
    const ph = m.ph(v), pv = m.pv(h);
    for (let j = 0; j < nh; j++) {
      svg.append("circle").attr("cx", hx(j)).attr("cy", hy).attr("r", 18).attr("fill", h[j] ? C.B : RB.OFF).attr("stroke", phase === 1 ? C.B : C.line).attr("stroke-width", 2);
      RB.txt(svg, hx(j), hy - 26, "p=" + RB.fmt(ph[j], 2), { size: 9.5, mono: true, anchor: "middle", fill: C.muted });
      RB.txt(svg, hx(j), hy + 4, "h" + "₁₂₃"[j], { size: 11, anchor: "middle", fill: h[j] ? "#0f1117" : C.ink });
    }
    for (let i = 0; i < nv; i++) {
      svg.append("circle").attr("cx", vx(i)).attr("cy", vy).attr("r", 18).attr("fill", v[i] ? C.A : RB.OFF).attr("stroke", phase === 0 ? C.A : C.line).attr("stroke-width", 2);
      RB.txt(svg, vx(i), vy + 36, "p=" + RB.fmt(pv[i], 2), { size: 9.5, mono: true, anchor: "middle", fill: C.muted });
      RB.txt(svg, vx(i), vy + 4, "v" + "₁₂₃₄"[i], { size: 11, anchor: "middle", fill: v[i] ? "#0f1117" : C.ink });
    }
    RB.txt(svg, 20, 330, phase === 0 ? "next half-step: sample h ~ p(h | v), all hidden at once" : "next half-step: sample v ~ p(v | h), all visible at once", { size: 10.5, fill: C.muted });

    /* histogram vs exact */
    const px = 330, pw = 410, py = 40, ph2 = 170, ex = exact(), tot = d3.sum(hist) || 1;
    RB.txt(svg, px, 20, "p(v): exact from the free energy (bars) · visited by the chain (dots)", { bold: true, size: 11 });
    const xs = d3.scaleBand().domain(d3.range(16)).range([px, px + pw]).padding(0.15);
    const ys = d3.scaleLinear().domain([0, Math.max(0.3, d3.max(ex) * 1.1)]).range([py + ph2, py]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(${px},0)`).call(d3.axisLeft(ys).ticks(4));
    svg.append("g").selectAll("rect").data(ex).join("rect").attr("x", (d, i) => xs(i)).attr("width", xs.bandwidth())
      .attr("y", d => ys(d)).attr("height", d => ys(0) - ys(d)).attr("fill", C.A).attr("fill-opacity", 0.55);
    if (steps) svg.append("g").selectAll("circle").data(hist).join("circle").attr("cx", (d, i) => xs(i) + xs.bandwidth() / 2)
      .attr("cy", d => ys(d / tot)).attr("r", 3.5).attr("fill", C.good).attr("stroke", "#0f1117");
    d3.range(16).forEach(x => RB.txt(svg, xs(x) + xs.bandwidth() / 2, py + ph2 + 12, RB.bits(x, 4).join(""), { size: 8, mono: true, anchor: "middle", fill: C.muted })
      .attr("transform", `rotate(-90,${xs(x) + xs.bandwidth() / 2},${py + ph2 + 12})`).attr("text-anchor", "end"));
    /* trace strip */
    const ty = 280, th = 60;
    RB.txt(svg, px, ty - 8, "last 200 visible states (row = state code)", { size: 10.5, bold: true });
    const tx = d3.scaleLinear().domain([0, 199]).range([px, px + pw]), tyS = d3.scaleLinear().domain([0, 15]).range([ty + th, ty]);
    svg.append("rect").attr("x", px).attr("y", ty).attr("width", pw).attr("height", th).attr("fill", "#11141b");
    svg.append("g").selectAll("rect").data(trace).join("rect").attr("x", (d, i) => tx(i)).attr("y", d => tyS(d) - 2).attr("width", Math.max(1.5, pw / 200)).attr("height", 4)
      .attr("fill", d => modeOf(RB.bits(d, 4)) === "L" ? C.A : modeOf(RB.bits(d, 4)) === "R" ? C.B : C.muted);
    const tv = steps ? 0.5 * d3.sum(ex, (p, i) => Math.abs(p - hist[i] / tot)) : NaN;
    $("gc-readout").innerHTML = `Gibbs steps ${steps} · total-variation distance to exact p(v): <b>${steps ? RB.fmt(tv, 3) : "—"}</b> · switches between the "left pair" and "right pair" modes: <b>${switches}</b>` +
      ` · p(1100) = ${RB.fmt(ex[12], 3)}, p(0011) = ${RB.fmt(ex[3], 3)}`;
  }
  function stop() { if (timer) { timer.stop(); timer = null; } $("gc-run").textContent = "run ▶"; }
  kE.addEventListener("input", () => { stop(); build(); draw(); });
  $("gc-half").addEventListener("click", () => { stop(); half(); draw(); });
  $("gc-1000").addEventListener("click", () => { stop(); for (let t = 0; t < 2000; t++) half(); draw(); });
  $("gc-run").addEventListener("click", () => {
    if (timer) { stop(); return; }
    $("gc-run").textContent = "pause ❚❚"; timer = d3.interval(() => { for (let t = 0; t < 6; t++) half(); draw(); }, 40);
  });
  $("gc-reset").addEventListener("click", () => { stop(); build(); draw(); });
  build(); draw();
})();

/* ═════════════════ 3 · #cd-svg — CD-k / PCD on bars-and-stripes ═════════════════ */
(function () {
  const svg = d3.select("#cd-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id);
  const nhE = $("cd-nh"), algE = $("cd-alg"), lrE = $("cd-lr"), decE = $("cd-decay");
  const data = RB.bas(), nv = 16, B = 10, MAXEP = 4000;
  let m, nh, ep, curve, chains, fant, mW, ma, mb, timer = null, seed = 1;

  function reset() {
    stop();
    nh = +nhE.value; $("cd-nhv").textContent = nh; $("cd-lrv").textContent = RB.fmt(+lrE.value, 3);
    m = RB.RBM(nv, nh, seed);
    for (let t = 0; t < nv * nh; t++) m.W[t] = 0.01 * RB.randn(m.r);
    for (let i = 0; i < nv; i++) { const p = d3.mean(data, v => v[i]); m.a[i] = Math.log(p / (1 - p)); }
    mW = new Float64Array(nv * nh); ma = new Float64Array(nv); mb = new Float64Array(nh);
    chains = d3.range(B).map(() => d3.range(nv).map(() => m.r() < 0.5 ? 1 : 0));
    fant = d3.range(8).map(() => d3.range(nv).map(() => m.r() < 0.5 ? 1 : 0));
    ep = 0; curve = []; measure(); draw();
  }
  function epoch() {
    const alg = algE.value, k = alg === "cd5" ? 5 : 1, pcd = alg === "pcd";
    const lr = +lrE.value / (decE.checked ? 1 + ep / 1000 : 1);
    const idx = d3.shuffle(d3.range(data.length), m.r);
    for (let bi = 0; bi < data.length; bi += B) {
      const batch = idx.slice(bi, bi + B).map(i => data[i]);
      const gW = new Float64Array(nv * nh), ga = new Float64Array(nv), gb = new Float64Array(nh);
      batch.forEach((v, q) => {
        const p0 = m.ph(v);
        let vk = pcd ? chains[q] : v, hk = pcd ? m.smp(m.ph(vk)) : m.smp(p0);
        for (let s = 0; s < k; s++) { vk = m.smp(m.pv(hk)); hk = m.smp(m.ph(vk)); }
        if (pcd) chains[q] = vk;
        const pk = m.ph(vk);
        for (let i = 0; i < nv; i++) { ga[i] += v[i] - vk[i]; for (let j = 0; j < nh; j++) gW[i * nh + j] += v[i] * p0[j] - vk[i] * pk[j]; }
        for (let j = 0; j < nh; j++) gb[j] += p0[j] - pk[j];
      });
      const n = batch.length;
      for (let t = 0; t < nv * nh; t++) { mW[t] = 0.5 * mW[t] + lr * (gW[t] / n - 1e-4 * m.W[t]); m.W[t] += mW[t]; }
      for (let i = 0; i < nv; i++) { ma[i] = 0.5 * ma[i] + lr * ga[i] / n; m.a[i] += ma[i]; }
      for (let j = 0; j < nh; j++) { mb[j] = 0.5 * mb[j] + lr * gb[j] / n; m.b[j] += mb[j]; }
    }
    ep++;
  }
  function measure() {
    let rec = 0; data.forEach(v => { const p = m.pv(m.smp(m.ph(v))); for (let i = 0; i < nv; i++) rec += (p[i] - v[i]) ** 2; });
    const lZ = m.logZ(); let ll = 0, mass = 0, pl = 0;
    data.forEach(v => {
      const f = m.F(v); ll += -f - lZ; mass += Math.exp(-f - lZ);
      const i = Math.floor(m.r() * nv), u = v.slice(); u[i] = 1 - u[i];
      pl += nv * Math.log(RB.sig(m.F(u) - f));
    });
    curve.push({ ep, rec: rec / data.length, ll: ll / data.length, pl: pl / data.length, mass });
  }
  function draw() {
    svg.selectAll("*").remove();
    const x0 = 50, w = 290, xs = d3.scaleLinear().domain([0, Math.max(200, ep)]).range([x0, x0 + w]);
    /* reconstruction error */
    const y1 = 34, h1 = 120;
    RB.txt(svg, x0 - 30, 20, "reconstruction error (squared, per image)", { bold: true, size: 11 });
    const yr = d3.scaleLinear().domain([0, Math.max(0.5, d3.max(curve, d => d.rec))]).range([y1 + h1, y1]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${y1 + h1})`).call(d3.axisBottom(xs).ticks(5));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(yr).ticks(4));
    svg.append("path").datum(curve).attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 1.8).attr("d", d3.line().x(d => xs(d.ep)).y(d => yr(d.rec)));
    /* log-likelihood */
    const y2 = 205, h2 = 150;
    RB.txt(svg, x0 - 30, y2 - 14, "exact mean log-likelihood (solid) · pseudo-likelihood (dotted)", { bold: true, size: 11 });
    const lo = Math.min(-12, d3.min(curve, d => Math.min(d.ll, d.pl)));
    const yl = d3.scaleLinear().domain([Math.max(lo, -25), -2]).range([y2 + h2, y2]).clamp(true);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${y2 + h2})`).call(d3.axisBottom(xs).ticks(5));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`).call(d3.axisLeft(yl).ticks(5));
    svg.append("line").attr("x1", x0).attr("x2", x0 + w).attr("y1", yl(-Math.log(30))).attr("y2", yl(-Math.log(30))).attr("stroke", C.good).attr("stroke-dasharray", "4 3");
    RB.txt(svg, x0 + w, yl(-Math.log(30)) - 5, "best possible: −ln 30 = −3.40", { size: 9.5, fill: C.good, anchor: "end" });
    svg.append("path").datum(curve).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2).attr("d", d3.line().x(d => xs(d.ep)).y(d => yl(d.ll)));
    svg.append("path").datum(curve).attr("fill", "none").attr("stroke", C.A).attr("stroke-opacity", 0.6).attr("stroke-width", 1.4).attr("stroke-dasharray", "2 3").attr("d", d3.line().x(d => xs(d.ep)).y(d => yl(d.pl)));
    RB.txt(svg, x0 + w, y2 + h2 + 28, "epochs (3 mini-batches of 10 each)", { size: 10, fill: C.muted, anchor: "end" });

    /* right: data, reconstructions, fantasies, filters */
    const rx = 390, cell = 8;
    RB.txt(svg, rx, 20, "data v", { size: 10.5, fill: C.muted });
    RB.txt(svg, rx, 78, "reconstruction p(v | h), h ~ p(h | v)", { size: 10.5, fill: C.muted });
    const show = [0, 3, 9, 14, 20, 27, 5, 22];
    show.forEach((d, q) => {
      RB.img(svg, data[d], rx + q * 44, 28, cell);
      RB.img(svg, Array.from(m.pv(m.smp(m.ph(data[d])))), rx + q * 44, 86, cell);
    });
    RB.txt(svg, rx, 140, "fantasies: 8 free-running Gibbs chains (p(v | h) shown)", { size: 10.5, fill: C.muted });
    fant.forEach((f, q) => { const hh = m.smp(m.ph(f)); RB.img(svg, Array.from(m.pv(hh)), rx + q * 44, 148, cell, { on: C.B }); });
    RB.txt(svg, rx, 204, `filters: each hidden unit's 16 weights (red −, blue +)`, { size: 10.5, fill: C.muted });
    const mx = d3.max(m.W, Math.abs) || 1;
    for (let j = 0; j < nh; j++) {
      const col = d3.range(nv).map(i => m.W[i * nh + j] / mx);
      RB.img(svg, col, rx + (j % 8) * 44, 212 + Math.floor(j / 8) * 44, cell, { div: true });
    }
    const c = curve[curve.length - 1];
    $("cd-readout").innerHTML = `epoch ${ep} · ${algE.options[algE.selectedIndex].text} · recon error <b>${RB.fmt(c.rec, 3)}</b> · exact log-likelihood <b>${RB.fmt(c.ll, 3)}</b> (optimum −3.401) · ` +
      `probability mass on the 30 training patterns <b>${RB.fmt(100 * c.mass, 1)}%</b>`;
  }
  function tick() {
    for (let t = 0; t < 20; t++) epoch();
    /* fantasy chains: 20 Gibbs steps per frame */
    fant = fant.map(f => { let v = f; for (let s = 0; s < 20; s++) v = m.smp(m.pv(m.smp(m.ph(v)))); return v; });
    measure(); draw();
    if (ep >= MAXEP) stop();
  }
  function stop() { if (timer) { timer.stop(); timer = null; } const b = $("cd-train"); if (b) b.textContent = "train ▶"; }
  $("cd-train").addEventListener("click", () => {
    if (timer) { stop(); return; }
    $("cd-train").textContent = "pause ❚❚"; timer = d3.interval(tick, 30);
  });
  $("cd-reset").addEventListener("click", () => { seed++; reset(); });
  nhE.addEventListener("input", () => { $("cd-nhv").textContent = nhE.value; });
  nhE.addEventListener("change", reset);
  lrE.addEventListener("input", () => { $("cd-lrv").textContent = RB.fmt(+lrE.value, 3); });
  algE.addEventListener("change", reset);
  lrE.addEventListener("change", reset);
  decE.addEventListener("change", reset);
  reset();
})();

/* ═════════════════ 4 · #gb-svg — Gaussian–Bernoulli RBM in 2-D ═════════════════ */
(function () {
  const svg = d3.select("#gb-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id);
  const nhE = $("gb-nh"), dsE = $("gb-data"), sE = $("gb-sigma");
  const L = 4, G = 64, X0 = 30, Y0 = 30, SZ = 330;
  const xs = d3.scaleLinear().domain([-L, L]).range([X0, X0 + SZ]), ys = d3.scaleLinear().domain([-L, L]).range([Y0 + SZ, Y0]);
  let data, a, b, W, nh, S, ep, timer = null, seed = 3, llc = [];
  const SETS = { four: [[-2, -2], [2, -2], [-2, 2], [2, 2]], three: [[-2, -2], [2, -2], [-2, 2]], two: [[-2, -1], [2, 1]] };
  function reset() {
    stop();
    nh = +nhE.value; S = +sE.value; $("gb-nhv").textContent = nh; $("gb-sigmav").textContent = RB.fmt(S, 2);
    const r = RB.rng(seed), cs = SETS[dsE.value];
    data = d3.range(240).map(i => { const c = cs[i % cs.length]; return [c[0] + 0.3 * RB.randn(r), c[1] + 0.3 * RB.randn(r)]; });
    a = [0, 0]; b = new Array(nh).fill(0); W = [d3.range(nh).map(() => 0.5 * RB.randn(r)), d3.range(nh).map(() => 0.5 * RB.randn(r))];
    ep = 0; llc = []; draw();
  }
  const logp = (x, y) => { let lp = -((x - a[0]) ** 2 + (y - a[1]) ** 2) / (2 * S * S); for (let j = 0; j < nh; j++) lp += RB.softplus(b[j] + (W[0][j] * x + W[1][j] * y) / S); return lp; };
  function epoch(r) {
    const lr = 0.02, n = 40;
    for (let bi = 0; bi < data.length; bi += n) {
      const gW = [new Array(nh).fill(0), new Array(nh).fill(0)], ga = [0, 0], gb = new Array(nh).fill(0);
      for (let q = bi; q < bi + n; q++) {
        const v = data[q];
        const p0 = b.map((bj, j) => RB.sig(bj + (W[0][j] * v[0] + W[1][j] * v[1]) / S));
        const h = p0.map(p => r() < p ? 1 : 0);
        const v1 = [0, 1].map(i => a[i] + S * d3.sum(h, (hj, j) => W[i][j] * hj) + S * RB.randn(r));
        const p1 = b.map((bj, j) => RB.sig(bj + (W[0][j] * v1[0] + W[1][j] * v1[1]) / S));
        for (let i = 0; i < 2; i++) { ga[i] += (v[i] - v1[i]) / (S * S); for (let j = 0; j < nh; j++) gW[i][j] += (v[i] * p0[j] - v1[i] * p1[j]) / S; }
        for (let j = 0; j < nh; j++) gb[j] += p0[j] - p1[j];
      }
      for (let i = 0; i < 2; i++) { a[i] += lr * ga[i] / n; for (let j = 0; j < nh; j++) W[i][j] += lr * gW[i][j] / n; }
      for (let j = 0; j < nh; j++) b[j] += lr * gb[j] / n;
    }
    ep++;
  }
  function draw() {
    svg.selectAll("*").remove();
    const vals = new Array(G * G);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) vals[j * G + i] = logp(-L + 2 * L * (i + 0.5) / G, L - 2 * L * (j + 0.5) / G);
    const lZ = RB.lse(vals) + Math.log((2 * L / G) ** 2);
    const dens = vals.map(v => Math.exp(v - lZ)), mx = d3.max(dens);
    const cs = d3.contours().size([G, G]).thresholds(d3.range(1, 13).map(k => mx * k / 13))(dens);
    const col = d3.scaleSequential(d3.interpolateViridis).domain([0, mx]);
    svg.append("rect").attr("x", X0).attr("y", Y0).attr("width", SZ).attr("height", SZ).attr("fill", "#11141b");
    svg.append("g").attr("transform", `translate(${X0},${Y0}) scale(${SZ / G})`).selectAll("path").data(cs).join("path")
      .attr("d", d3.geoPath()).attr("fill", d => col(d.value)).attr("fill-opacity", 0.85);
    svg.append("g").selectAll("circle").data(data).join("circle").attr("cx", d => xs(d[0])).attr("cy", d => ys(d[1])).attr("r", 1.8).attr("fill", "#fff").attr("fill-opacity", 0.7);
    /* the 2^nh component means and their mixing weights */
    const comps = d3.range(1 << nh).map(c => {
      const h = RB.bits(c, nh), mu = [0, 1].map(i => a[i] + S * d3.sum(h, (hj, j) => W[i][j] * hj));
      /* integrate v out of p(v, h):  weight(h) ∝ exp(bᵀh + aᵀWh/σ + ½|Wh|²) */
      const Wh = [0, 1].map(i => d3.sum(h, (hj, j) => W[i][j] * hj));
      const lw = d3.sum(h, (hj, j) => b[j] * hj) + (a[0] * Wh[0] + a[1] * Wh[1]) / S + 0.5 * (Wh[0] ** 2 + Wh[1] ** 2);
      return { h, mu, lw };
    });
    const lwZ = RB.lse(comps.map(c => c.lw));
    comps.forEach(c => { c.w = Math.exp(c.lw - lwZ); });
    comps.forEach(c => {
      const inside = Math.abs(c.mu[0]) < L && Math.abs(c.mu[1]) < L;
      if (!inside) return;
      svg.append("path").attr("d", d3.symbol(d3.symbolCross, 30 + 160 * c.w)()).attr("transform", `translate(${xs(c.mu[0])},${ys(c.mu[1])})`)
        .attr("fill", C.B).attr("stroke", "#0f1117").attr("stroke-width", 0.8).attr("fill-opacity", 0.35 + 0.65 * Math.min(1, c.w * 3));
      RB.txt(svg, xs(c.mu[0]) + 8, ys(c.mu[1]) - 6, "h=" + c.h.join(""), { size: 9, mono: true, fill: C.B });
    });
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${Y0 + SZ})`).call(d3.axisBottom(xs).ticks(5));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${X0},0)`).call(d3.axisLeft(ys).ticks(5));
    RB.txt(svg, X0, 18, "density p(v), data (white), means a + σWh (+)", { bold: true, size: 10.5 });
    const ll = d3.mean(data, d => logp(d[0], d[1]) - lZ);
    if (!llc.length || llc[llc.length - 1].ep !== ep) llc.push({ ep, ll });
    /* right: learning curve + table */
    const px = 430, pw = 300, py = 40, ph = 140;
    RB.txt(svg, px, 20, "data log-likelihood (normalised on a grid)", { bold: true, size: 10.5 });
    const cx = d3.scaleLinear().domain([0, Math.max(100, ep)]).range([px, px + pw]);
    const cy = d3.scaleLinear().domain([Math.min(-6, d3.min(llc, d => d.ll)), Math.max(-1, d3.max(llc, d => d.ll) + 0.2)]).range([py + ph, py]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${py + ph})`).call(d3.axisBottom(cx).ticks(5));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${px},0)`).call(d3.axisLeft(cy).ticks(4));
    svg.append("path").datum(llc).attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2).attr("d", d3.line().x(d => cx(d.ep)).y(d => cy(d.ll)));
    RB.txt(svg, px, 222, "components (one per hidden configuration h):", { size: 10.5, fill: C.muted });
    comps.slice(0, 16).sort((p, q) => q.w - p.w).slice(0, 8).forEach((c, k) => {
      RB.txt(svg, px + (k % 2) * 160, 242 + Math.floor(k / 2) * 18, `h=${c.h.join("")}  w=${RB.fmt(c.w, 3)}`, { size: 10.5, mono: true, fill: c.w > 0.05 ? C.ink : C.muted });
    });
    $("gb-readout").innerHTML = `epoch ${ep} · ${nh} hidden units → ${1 << nh} Gaussian components with tied means and weights · data log-likelihood <b>${RB.fmt(ll, 3)}</b> · components carrying &gt; 5% weight: <b>${comps.filter(c => c.w > 0.05).length}</b>`;
  }
  let r = RB.rng(99);
  function stop() { if (timer) { timer.stop(); timer = null; } $("gb-train").textContent = "train ▶"; }
  $("gb-train").addEventListener("click", () => {
    if (timer) { stop(); return; }
    $("gb-train").textContent = "pause ❚❚";
    timer = d3.interval(() => { for (let t = 0; t < 8; t++) epoch(r); draw(); if (ep >= 1500) stop(); }, 40);
  });
  $("gb-reset").addEventListener("click", () => { seed += 1; reset(); });
  [nhE, sE].forEach(e => e.addEventListener("input", () => { $("gb-nhv").textContent = nhE.value; $("gb-sigmav").textContent = RB.fmt(+sE.value, 2); }));
  [nhE, sE, dsE].forEach(e => e.addEventListener("change", reset));
  reset();
})();

/* ═════════════════ 5 · #dbn-svg — greedy stacking ═════════════════ */
(function () {
  const svg = d3.select("#dbn-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id);
  const layers = [{ n: 9, name: "v (data)" }, { n: 7, name: "h¹" }, { n: 6, name: "h²" }, { n: 5, name: "h³" }];
  /* each step: edge kinds for W¹..W³ ("none","train","frozen-undirected","directed-down","undirected-top","up-rec"), highlight, caption */
  const STEPS = [
    { e: ["train", "none", "none"], act: [0, 1], cap: "Step 1 · train RBM₁ on the data with CD or PCD. W¹ is an undirected bipartite model of v." },
    { e: ["frozen", "none", "none"], act: [1], cap: "Step 2 · freeze W¹. Push every training vector up: q(h¹ | v) = σ(b¹ + W¹ᵀv). These activations (or samples) become the \"data\" for the next layer." },
    { e: ["frozen", "train", "none"], act: [1, 2], cap: "Step 3 · train RBM₂ on q(h¹ | v). If W² starts at W¹ᵀ, adding it cannot lower the variational bound on log p(v), and training it raises the bound." },
    { e: ["frozen", "frozen", "train"], act: [2, 3], cap: "Step 4 · repeat: freeze W², push up, train RBM₃ on q(h² | h¹). Each layer models the aggregate posterior of the one below." },
    { e: ["down", "down", "top"], act: [], cap: "Step 5 · the result is a DBN. The top two layers form an undirected RBM (an associative memory), and every lower layer is a directed, top-down sigmoid belief net. The upward weights are kept only as a fast approximate inference network." },
    { e: ["down", "down", "top"], act: [3, 2], gen: true, cap: "Step 6 · to generate: run a long Gibbs chain in the top RBM (h² ⇄ h³), then do a single top-down pass h² → h¹ → v with p(h¹ | h²) and p(v | h¹)." },
    { e: ["fine", "fine", "fine"], act: [0, 1, 2, 3], cap: "Step 7 · fine-tune. Generatively with the up–down (wake–sleep) algorithm, or, as in most of the 2006–2012 practice, use the weights to initialise an MLP (or a deep autoencoder) and train the whole thing with backprop." }
  ];
  let k = 0;
  function draw() {
    svg.selectAll("*").remove();
    const st = STEPS[k], X = 380, gapY = 78, baseY = 320;
    const ly = i => baseY - i * gapY, ux = (i, u) => X + (u - (layers[i].n - 1) / 2) * 40;
    for (let l = 0; l < 3; l++) {
      const kind = st.e[l]; if (kind === "none") continue;
      const col = kind === "train" ? C.B : kind === "top" ? C.good : kind === "fine" ? C.A : kind === "down" ? C.A : C.muted;
      const g = svg.append("g").attr("stroke", col).attr("stroke-opacity", kind === "frozen" ? 0.3 : 0.45).attr("stroke-width", kind === "train" ? 1.4 : 1);
      for (let i = 0; i < layers[l].n; i++) for (let j = 0; j < layers[l + 1].n; j++)
        g.append("line").attr("x1", ux(l, i)).attr("y1", ly(l) - 11).attr("x2", ux(l + 1, j)).attr("y2", ly(l + 1) + 11);
      const lab = kind === "train" ? "training (undirected RBM)" : kind === "frozen" ? "frozen" : kind === "down" ? "directed, top-down  p(below | above)" : kind === "top" ? "undirected top-level RBM" : "fine-tuning";
      RB.txt(svg, X + 200, (ly(l) + ly(l + 1)) / 2 + 4, `W${"¹²³"[l]}: ${lab}`, { size: 11, fill: col });
      if (kind === "down") svg.append("path").attr("d", `M${X - 160},${ly(l + 1) + 14} L${X - 160},${ly(l) - 14}`).attr("stroke", C.A).attr("stroke-width", 2).attr("marker-end", "url(#dbn-ar)");
    }
    svg.append("defs").append("marker").attr("id", "dbn-ar").attr("viewBox", "0 0 10 10").attr("refX", 8).attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto")
      .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", C.A);
    layers.forEach((L, i) => {
      for (let u = 0; u < L.n; u++) svg.append("circle").attr("cx", ux(i, u)).attr("cy", ly(i)).attr("r", 10)
        .attr("fill", st.act.includes(i) ? (st.gen && i === 0 ? C.B : "#2a3550") : RB.OFF).attr("stroke", st.act.includes(i) ? C.B : C.line).attr("stroke-width", 1.5);
      RB.txt(svg, X - 220, ly(i) + 4, L.name, { size: 12, bold: true, anchor: "end" });
    });
    if (k >= 4) RB.txt(svg, X - 150, (ly(0) + ly(1)) / 2 + 4, "generate", { size: 10, fill: C.A });
    $("dbn-readout").innerHTML = `<b>${k + 1} / ${STEPS.length}</b> · ${st.cap}`;
    $("dbn-prev").disabled = k === 0; $("dbn-next").disabled = k === STEPS.length - 1;
  }
  $("dbn-prev").addEventListener("click", () => { if (k > 0) { k--; draw(); } });
  $("dbn-next").addEventListener("click", () => { if (k < STEPS.length - 1) { k++; draw(); } });
  draw();
})();

/* ═════════════════ 6 · #lv-svg — Langevin sampling ═════════════════ */
(function () {
  const svg = d3.select("#lv-svg");
  if (svg.empty()) return;
  const $ = id => document.getElementById(id);
  const epsE = $("lv-eps"), TE = $("lv-T"), noiseE = $("lv-noise");
  const MU = [[-2, -1.5], [2, -1], [0, 2]], PI = [0.5, 0.3, 0.2], s = 0.8, L = 4;
  const X0 = 30, Y0 = 30, SZ = 330;
  const xs = d3.scaleLinear().domain([-L, L]).range([X0, X0 + SZ]), ys = d3.scaleLinear().domain([-L, L]).range([Y0 + SZ, Y0]);
  let pts, timer = null, t = 0, r = RB.rng(8);
  function comps(x, y) { return MU.map((m, k) => Math.log(PI[k]) - ((x - m[0]) ** 2 + (y - m[1]) ** 2) / (2 * s * s)); }
  const E = (x, y) => -RB.lse(comps(x, y));
  function grad(x, y) {                     /* ∇E = −∇ log p = Σ_k resp_k (x − μ_k)/s² */
    const c = comps(x, y), l = RB.lse(c); let gx = 0, gy = 0;
    c.forEach((v, k) => { const w = Math.exp(v - l); gx += w * (x - MU[k][0]) / (s * s); gy += w * (y - MU[k][1]) / (s * s); });
    return [gx, gy];
  }
  function reset() { pts = d3.range(300).map(() => [-L + 2 * L * r(), -L + 2 * L * r()]); t = 0; draw(); }
  function step() {
    const eps = Math.pow(10, +epsE.value), T = +TE.value, noise = noiseE.checked;
    pts = pts.map(([x, y]) => {
      const [gx, gy] = grad(x, y);
      let nx = x - 0.5 * eps * gx / T, ny = y - 0.5 * eps * gy / T;
      if (noise) { nx += Math.sqrt(eps) * RB.randn(r); ny += Math.sqrt(eps) * RB.randn(r); }
      return [Math.max(-L, Math.min(L, nx)), Math.max(-L, Math.min(L, ny))];
    });
    t++;
  }
  function draw() {
    const eps = Math.pow(10, +epsE.value), T = +TE.value;
    $("lv-epsv").textContent = RB.fmt(eps, 3); $("lv-Tv").textContent = RB.fmt(T, 2);
    svg.selectAll("*").remove();
    const G = 60, vals = new Array(G * G);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) vals[j * G + i] = Math.min(12, E(-L + 2 * L * (i + 0.5) / G, L - 2 * L * (j + 0.5) / G));
    const lo = d3.min(vals);
    const cs = d3.contours().size([G, G]).thresholds(d3.range(lo, 12, 0.7))(vals);
    const col = d3.scaleSequential(d3.interpolateMagma).domain([lo, 12]);
    svg.append("g").attr("transform", `translate(${X0},${Y0}) scale(${SZ / G})`).selectAll("path").data(cs).join("path")
      .attr("d", d3.geoPath()).attr("fill", d => col(d.value)).attr("stroke", "#0f1117").attr("stroke-width", 0.02);
    svg.append("g").selectAll("circle").data(pts).join("circle").attr("cx", d => xs(d[0])).attr("cy", d => ys(d[1])).attr("r", 2.2).attr("fill", C.good).attr("fill-opacity", 0.85);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${Y0 + SZ})`).call(d3.axisBottom(xs).ticks(5));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${X0},0)`).call(d3.axisLeft(ys).ticks(5));
    RB.txt(svg, X0, 18, "energy E(x) = −log p(x) (dark = low) and 300 particles", { bold: true, size: 11 });
    MU.forEach((m, k) => RB.txt(svg, xs(m[0]) + 10, ys(m[1]) - 10, "mode " + (k + 1), { size: 10, fill: "#fff" }));
    /* mode occupancy vs target */
    const cnt = [0, 0, 0]; pts.forEach(([x, y]) => { const c = comps(x, y); cnt[c.indexOf(d3.max(c))]++; });
    const tgtRaw = PI.map(p => Math.pow(p, 1 / T)), tgt = tgtRaw.map(v => v / d3.sum(tgtRaw));
    const px = 430, pw = 290, py = 50, ph = 150;
    RB.txt(svg, px, 20, "share of particles in each mode's basin", { bold: true, size: 11 });
    const bx = d3.scaleBand().domain([0, 1, 2]).range([px, px + pw]).padding(0.3);
    const by = d3.scaleLinear().domain([0, 1]).range([py + ph, py]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(${px},0)`).call(d3.axisLeft(by).ticks(5));
    svg.append("g").selectAll("rect").data(cnt).join("rect").attr("x", (d, i) => bx(i)).attr("width", bx.bandwidth())
      .attr("y", d => by(d / pts.length)).attr("height", d => by(0) - by(d / pts.length)).attr("fill", C.good).attr("fill-opacity", 0.75);
    svg.append("g").selectAll("line").data(tgt).join("line").attr("x1", (d, i) => bx(i) - 6).attr("x2", (d, i) => bx(i) + bx.bandwidth() + 6)
      .attr("y1", d => by(d)).attr("y2", d => by(d)).attr("stroke", C.B).attr("stroke-width", 2.5);
    [0, 1, 2].forEach(i => RB.txt(svg, bx(i) + bx.bandwidth() / 2, py + ph + 14, `mode ${i + 1} (π=${PI[i]})`, { size: 10, anchor: "middle", fill: C.muted }));
    RB.txt(svg, px, py + ph + 36, "orange: target mass ∝ πₖ^(1/T) (approximate)", { size: 10, fill: C.B });
    const lines = [
      `update:  x ← x − (ε/2)·∇E(x)/T ${noiseE.checked ? "+ √ε·z" : "   (no noise)"}`,
      "∇E = −∇ log p: the score. Z never appears.",
      noiseE.checked ? "with noise: particles sample p(x)^(1/T)" : "no noise: gradient descent — every particle",
      noiseE.checked ? "(they cross between modes only rarely)" : "falls into the nearest minimum (Hopfield-like)"
    ];
    lines.forEach((l, i) => RB.txt(svg, px, 262 + i * 18, l, { size: 10.5, mono: i === 0, fill: i === 0 ? C.ink : C.muted }));
    const tv = 0.5 * d3.sum(tgt, (v, i) => Math.abs(v - cnt[i] / pts.length));
    $("lv-readout").innerHTML = `steps ${t} · shares ${cnt.map(c => RB.fmt(c / pts.length, 2)).join(" / ")} against target ${tgt.map(v => RB.fmt(v, 2)).join(" / ")} · gap <b>${RB.fmt(tv, 3)}</b>`;
  }
  function stop() { if (timer) { timer.stop(); timer = null; } $("lv-run").textContent = "run ▶"; }
  $("lv-run").addEventListener("click", () => {
    if (timer) { stop(); return; }
    $("lv-run").textContent = "pause ❚❚"; timer = d3.interval(() => { for (let q = 0; q < 6; q++) step(); draw(); }, 40);
  });
  $("lv-reset").addEventListener("click", () => { stop(); reset(); });
  [epsE, TE].forEach(e => e.addEventListener("input", draw));
  noiseE.addEventListener("change", draw);
  reset();
})();
