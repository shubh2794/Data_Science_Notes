/* topic-modeling.viz.js — figures for nlp/topic-modeling.html.
   Loaded after data.js / notes.js. One IIFE per figure; each returns early if its
   elements are missing. All randomness is seeded so every run draws the same thing. */

/* shared helpers (kept local to this file) */
const TM = (function(){
  function rng(seed){
    let a = seed >>> 0;
    return function(){
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r){ let u = 0; while(u === 0) u = r(); const v = r(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
  /* Gamma(a, 1): Marsaglia–Tsang for a ≥ 1, boosted for a < 1 */
  function gamma(a, r){
    if(a < 1){ let u = 0; while(u === 0) u = r(); return gamma(a + 1, r) * Math.pow(u, 1/a); }
    const d = a - 1/3, c = 1/Math.sqrt(9*d);
    for(;;){
      let z, v;
      do { z = gauss(r); v = 1 + c*z; } while(v <= 0);
      v = v*v*v;
      const u = r();
      if(Math.log(u) < 0.5*z*z + d - d*v + d*Math.log(v)) return d*v;
    }
  }
  /* Dirichlet by normalised gammas; guards the all-underflow case at tiny α */
  function dirichlet(alphas, r){
    const g = alphas.map(a => gamma(a, r));
    let s = g.reduce((x, y) => x + y, 0);
    if(!(s > 0)){ const i = Math.floor(r()*g.length); return g.map((_, j) => j === i ? 1 : 0); }
    return g.map(x => x / s);
  }
  function cat(p, r){ const u = r(); let c = 0; for(let i = 0; i < p.length; i++){ c += p[i]; if(u < c) return i; } return p.length - 1; }
  /* log Γ via Lanczos (g = 7, n = 9) */
  const LC = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  function lgamma(x){
    if(x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI*x))) - lgamma(1 - x);
    x -= 1; let a = LC[0]; const t = x + 7.5;
    for(let i = 1; i < 9; i++) a += LC[i] / (x + i);
    return 0.5*Math.log(2*Math.PI) + (x + 0.5)*Math.log(t) - t + Math.log(a);
  }
  const PAL = ["#2dd4bf", "#ffb454", "#c084fc", "#5b9cff", "#f87171", "#a3e635"];
  return { rng, gamma, dirichlet, cat, lgamma, PAL };
})();

/* ── Figure 1 · Dirichlet simplex explorer ───────────────────────────── */
(function(){
  const svgEl = document.getElementById("dir-svg");
  if(!svgEl || typeof d3 === "undefined") return;
  const svg = d3.select(svgEl);
  const aIn = document.getElementById("dir-alpha"), shapeIn = document.getElementById("dir-shape"),
        btn = document.getElementById("dir-resample"), out = document.getElementById("dir-read");
  const AV = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10, 20, 50];
  const N = 500;
  let seed = 11;
  /* triangle corners: topic 1 top, topic 2 bottom-left, topic 3 bottom-right */
  const P = [[200, 34], [40, 312], [360, 312]];
  const xy = t => [t[0]*P[0][0] + t[1]*P[1][0] + t[2]*P[2][0], t[0]*P[0][1] + t[1]*P[1][1] + t[2]*P[2][1]];
  const names = ["topic 1", "topic 2", "topic 3"];

  function draw(){
    svg.selectAll("*").remove();
    const a = AV[Math.max(0, Math.min(AV.length - 1, +(aIn ? aIn.value : 6)))];
    const shape = shapeIn ? shapeIn.value : "sym";
    const alphas = shape === "sym" ? [a, a, a] : [4*a, a, a];
    const r = TM.rng(seed);
    const pts = d3.range(N).map(() => TM.dirichlet(alphas, r));

    svg.append("path").attr("d", "M" + P.map(p => p.join(",")).join("L") + "Z")
      .attr("fill", "#11141b").attr("stroke", "#2a2f3a");
    /* interior guide: centre and mid-edges */
    svg.append("circle").attr("cx", 200).attr("cy", (34 + 312 + 312)/3).attr("r", 3).attr("fill", "none").attr("stroke", "#9aa3b2");
    P.forEach((p, i) => svg.append("text").attr("x", p[0]).attr("y", i === 0 ? p[1] - 12 : p[1] + 20)
      .attr("text-anchor", i === 0 ? "middle" : (i === 1 ? "start" : "end")).attr("font-size", 11).attr("fill", TM.PAL[i]).text(names[i] + " only"));
    svg.append("g").selectAll("circle").data(pts).enter().append("circle")
      .attr("cx", t => xy(t)[0]).attr("cy", t => xy(t)[1]).attr("r", 1.9)
      .attr("fill", t => TM.PAL[t.indexOf(Math.max(...t))]).attr("fill-opacity", .7);
    /* mean marker */
    const s = alphas[0] + alphas[1] + alphas[2], m = xy(alphas.map(x => x / s));
    svg.append("path").attr("d", `M${m[0]-6},${m[1]}L${m[0]+6},${m[1]}M${m[0]},${m[1]-6}L${m[0]},${m[1]+6}`).attr("stroke", "#e6e9ef").attr("stroke-width", 2);

    /* right panel: first five draws as "documents" (stacked bars) */
    const bx = 420, bw = 190, bh = 22;
    svg.append("text").attr("x", bx).attr("y", 46).attr("font-size", 11).attr("fill", "#9aa3b2").text("five sampled documents: θ_d");
    pts.slice(0, 5).forEach((t, i) => {
      let x0 = bx; const y = 60 + i*(bh + 12);
      t.forEach((v, k) => {
        svg.append("rect").attr("x", x0).attr("y", y).attr("width", Math.max(0, v*bw)).attr("height", bh).attr("fill", TM.PAL[k]).attr("fill-opacity", .85);
        x0 += v*bw;
      });
      svg.append("text").attr("x", bx - 6).attr("y", y + 15).attr("text-anchor", "end").attr("font-size", 10).attr("fill", "#9aa3b2").text("d" + (i + 1));
    });
    /* summary statistics */
    const maxc = d3.mean(pts, t => Math.max(...t));
    const dom = pts.filter(t => Math.max(...t) > 0.8).length / N;
    const H = d3.mean(pts, t => -t.reduce((h, v) => h + (v > 0 ? v*Math.log2(v) : 0), 0));
    const sy = 268;
    [["mean largest share", maxc.toFixed(2)], ["docs with one topic > 80%", Math.round(dom*100) + "%"], ["mean entropy (max 1.58 bits)", H.toFixed(2) + " bits"]]
      .forEach((row, i) => {
        svg.append("text").attr("x", bx).attr("y", sy + i*18).attr("font-size", 11).attr("fill", "#9aa3b2").text(row[0]);
        svg.append("text").attr("x", bx + bw + 10).attr("y", sy + i*18).attr("text-anchor", "end").attr("font-size", 11).attr("fill", "#ffb454").text(row[1]);
      });
    const regime = a < 1 ? "α &lt; 1: mass piles onto corners and edges — <b>sparse</b> mixtures, most documents about one or two topics"
      : a === 1 ? "α = 1: <b>uniform</b> over the simplex — every mixture equally likely"
      : "α &gt; 1: mass pulled to the centre — <b>smooth</b> mixtures, every document about everything";
    if(out) out.innerHTML = `α = (${alphas.map(x => +x.toFixed(2)).join(", ")}) · ${regime}${shape === "sym" ? "" : " · topic 1 favoured: the mean (cross) moves toward its corner"}`;
  }
  if(aIn) aIn.addEventListener("input", draw);
  if(shapeIn) shapeIn.addEventListener("change", draw);
  if(btn) btn.addEventListener("click", () => { seed = (seed * 48271) % 2147483647; draw(); });
  draw();
})();

/* ── Figure 2 · collapsed Gibbs sampler on a toy corpus ─────────────── */
(function(){
  const svgEl = document.getElementById("gibbs-svg");
  if(!svgEl || typeof d3 === "undefined") return;
  const svg = d3.select(svgEl);
  const kIn = document.getElementById("gibbs-k"), aIn = document.getElementById("gibbs-alpha"),
        stepB = document.getElementById("gibbs-step"), runB = document.getElementById("gibbs-run"),
        resetB = document.getElementById("gibbs-reset"), out = document.getElementById("gibbs-read");
  const DOCS = [
    "goal team coach goal match team score",
    "match score goal coach team match",
    "pasta sauce bake flour sauce pasta oven",
    "flour bake oven bake pasta sauce",
    "stock market rate bank stock profit",
    "bank rate market profit stock rate",
    "team coach match pasta sauce oven",
    "stock market profit goal team score",
    "bake flour oven bank rate profit"
  ].map(s => s.split(" "));
  const VOCAB = Array.from(new Set(DOCS.flat()));
  const W = DOCS.map(d => d.map(w => VOCAB.indexOf(w)));
  const V = VOCAB.length, BETA = 0.1;
  let K, ALPHA, z, ndk, nkw, nk, sweep, changed, r;

  function init(){
    K = +(kIn ? kIn.value : 3);
    ALPHA = [0.1, 0.5, 1, 2][+(aIn ? aIn.value : 1)] || 0.5;
    r = TM.rng(2024);
    ndk = W.map(() => new Array(K).fill(0));
    nkw = d3.range(K).map(() => new Array(V).fill(0));
    nk = new Array(K).fill(0);
    z = W.map((doc, d) => doc.map(w => { const k = Math.floor(r()*K); ndk[d][k]++; nkw[k][w]++; nk[k]++; return k; }));
    sweep = 0; changed = null;
  }
  /* one full sweep of the collapsed Gibbs update */
  function step(){
    const p = new Array(K);
    changed = 0;
    W.forEach((doc, d) => doc.forEach((w, i) => {
      const old = z[d][i];
      ndk[d][old]--; nkw[old][w]--; nk[old]--;
      let t = 0;
      for(let k = 0; k < K; k++){ t += (ndk[d][k] + ALPHA) * (nkw[k][w] + BETA) / (nk[k] + V*BETA); p[k] = t; }
      const u = r()*t; let k = 0; while(k < K - 1 && p[k] < u) k++;
      z[d][i] = k; ndk[d][k]++; nkw[k][w]++; nk[k]++;
      if(k !== old) changed++;
    }));
    sweep++;
  }
  /* collapsed joint log p(w, z | α, β) */
  function logJoint(){
    const L = TM.lgamma;
    let lw = K*(L(V*BETA) - V*L(BETA));
    for(let k = 0; k < K; k++){ for(let w = 0; w < V; w++) lw += L(nkw[k][w] + BETA); lw -= L(nk[k] + V*BETA); }
    let lz = W.length*(L(K*ALPHA) - K*L(ALPHA));
    W.forEach((doc, d) => { for(let k = 0; k < K; k++) lz += L(ndk[d][k] + ALPHA); lz -= L(doc.length + K*ALPHA); });
    return lw + lz;
  }
  function draw(){
    svg.selectAll("*").remove();
    const x0 = 34, cw = 50, rh = 30, y0 = 34;
    svg.append("text").attr("x", x0).attr("y", 18).attr("font-size", 11).attr("fill", "#9aa3b2").text("documents — each token coloured by its current topic z");
    W.forEach((doc, d) => {
      const y = y0 + d*rh;
      svg.append("text").attr("x", x0 - 6).attr("y", y + 15).attr("text-anchor", "end").attr("font-size", 10).attr("fill", "#9aa3b2").text("d" + (d + 1));
      doc.forEach((w, i) => {
        const col = TM.PAL[z[d][i]];
        svg.append("rect").attr("x", x0 + i*cw).attr("y", y).attr("width", cw - 4).attr("height", rh - 8).attr("rx", 4)
          .attr("fill", col).attr("fill-opacity", .22).attr("stroke", col);
        svg.append("text").attr("x", x0 + i*cw + (cw - 4)/2).attr("y", y + 15).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "#e6e9ef").text(VOCAB[w]);
      });
    });
    /* topic–word estimates φ̂_k,w = (n_k,w + β)/(n_k + Vβ): top five per topic */
    const px = 410, colW = (680 - px - 10) / K;
    svg.append("text").attr("x", px).attr("y", 18).attr("font-size", 11).attr("fill", "#9aa3b2").text("topics — top words by φ̂");
    for(let k = 0; k < K; k++){
      const phi = nkw[k].map((c, w) => [(c + BETA)/(nk[k] + V*BETA), w]).sort((a, b) => b[0] - a[0]).slice(0, 5);
      const cx = px + k*colW;
      svg.append("text").attr("x", cx).attr("y", 40).attr("font-size", 11).attr("font-weight", 700).attr("fill", TM.PAL[k]).text("topic " + (k + 1));
      phi.forEach(([v, w], j) => {
        const y = 52 + j*26;
        svg.append("rect").attr("x", cx).attr("y", y + 12).attr("width", Math.max(1, v*(colW - 14)*1.6)).attr("height", 7).attr("fill", TM.PAL[k]).attr("fill-opacity", .8);
        svg.append("text").attr("x", cx).attr("y", y + 9).attr("font-size", 10).attr("fill", "#e6e9ef").text(VOCAB[w] + " " + v.toFixed(2));
      });
    }
    if(out) out.innerHTML = `sweep <b>${sweep}</b> · log p(w, z) = <b>${logJoint().toFixed(1)}</b>` +
      (changed === null ? " · random initial assignment" : ` · ${changed} of ${W.flat().length} tokens changed topic this sweep`) +
      ` · K = ${K}, α = ${ALPHA}, β = ${BETA}`;
  }
  init(); draw();
  if(stepB) stepB.addEventListener("click", () => { step(); draw(); });
  if(runB) runB.addEventListener("click", () => { for(let s = 0; s < 25; s++) step(); draw(); });
  if(resetB) resetB.addEventListener("click", () => { init(); draw(); });
  if(kIn) kIn.addEventListener("input", () => { init(); draw(); });
  if(aIn) aIn.addEventListener("input", () => { init(); draw(); });
})();

/* ── Figure 3 · choosing K: held-out perplexity vs coherence vs stability ── */
(function(){
  const svgEl = document.getElementById("ksel-svg");
  if(!svgEl || typeof d3 === "undefined") return;
  const svg = d3.select(svgEl);
  const tIn = document.getElementById("ksel-true"), out = document.getElementById("ksel-read");
  const V = 48, D = 240, NW = 40, KMAX = 10, ALPHA = 0.5, BETA = 0.1, SWEEPS = 80;
  const cache = {};

  /* synthetic corpus from a true LDA with Kt block-structured topics */
  function corpus(Kt){
    const r = TM.rng(7 + Kt), per = Math.floor(V / Kt), phi = [];
    for(let k = 0; k < Kt; k++){
      const w = new Array(V).fill(0.02);
      for(let j = k*per; j < (k + 1)*per; j++) w[j] = 1 + 2*r();
      const s = w.reduce((a, b) => a + b, 0); phi.push(w.map(x => x / s));
    }
    return d3.range(D).map(() => { const th = TM.dirichlet(new Array(Kt).fill(0.3), r); return d3.range(NW).map(() => TM.cat(phi[TM.cat(th, r)], r)); });
  }
  function fit(docs, K, r){
    const ndk = docs.map(() => new Float64Array(K)), nkw = d3.range(K).map(() => new Float64Array(V)), nk = new Float64Array(K), p = new Float64Array(K);
    const z = docs.map((doc, d) => doc.map(w => { const k = Math.floor(r()*K); ndk[d][k]++; nkw[k][w]++; nk[k]++; return k; }));
    for(let s = 0; s < SWEEPS; s++) docs.forEach((doc, d) => doc.forEach((w, i) => {
      let k = z[d][i]; ndk[d][k]--; nkw[k][w]--; nk[k]--;
      let t = 0; for(let j = 0; j < K; j++){ t += (ndk[d][j] + ALPHA)*(nkw[j][w] + BETA)/(nk[j] + V*BETA); p[j] = t; }
      const u = r()*t; k = 0; while(k < K - 1 && p[k] < u) k++;
      z[d][i] = k; ndk[d][k]++; nkw[k][w]++; nk[k]++;
    }));
    return nkw.map((row, k) => Array.from(row, c => (c + BETA)/(nk[k] + V*BETA)));
  }
  /* document completion: fold θ in on even positions with φ fixed, score odd positions */
  function perplexity(test, phi, K){
    const r = TM.rng(5), p = new Float64Array(K); let ll = 0, n = 0;
    test.forEach(doc => {
      const a = doc.filter((_, i) => i % 2 === 0), b = doc.filter((_, i) => i % 2 === 1);
      const ndk = new Float64Array(K), z = a.map(() => Math.floor(r()*K)); z.forEach(k => ndk[k]++);
      for(let s = 0; s < 20; s++) a.forEach((w, i) => {
        ndk[z[i]]--; let t = 0; for(let j = 0; j < K; j++){ t += (ndk[j] + ALPHA)*phi[j][w]; p[j] = t; }
        const u = r()*t; let k = 0; while(k < K - 1 && p[k] < u) k++; z[i] = k; ndk[k]++;
      });
      b.forEach(w => { let q = 0; for(let j = 0; j < K; j++) q += (ndk[j] + ALPHA)/(a.length + K*ALPHA)*phi[j][w]; ll += Math.log(q); n++; });
    });
    return Math.exp(-ll / n);
  }
  const top = (phi, M) => phi.map(row => row.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, M).map(x => x[1]));
  function umass(docs, T){
    const sets = docs.map(d => new Set(d));
    const df = w => sets.reduce((c, s) => c + (s.has(w) ? 1 : 0), 0);
    const co = (a, b) => sets.reduce((c, s) => c + (s.has(a) && s.has(b) ? 1 : 0), 0);
    return d3.mean(T, t => { let c = 0; for(let m = 1; m < t.length; m++) for(let l = 0; l < m; l++) c += Math.log((co(t[m], t[l]) + 1)/df(t[l])); return c; });
  }
  function stability(A, B){
    return d3.mean(A, a => d3.max(B, b => { const s = new Set(b), i = a.filter(x => s.has(x)).length; return i / (2*a.length - i); }));
  }
  function compute(Kt){
    if(cache[Kt]) return cache[Kt];
    const docs = corpus(Kt), train = docs.slice(0, 180), test = docs.slice(180), rows = [];
    for(let K = 2; K <= KMAX; K++){
      let px = 0, um = 0; const Ts = [];
      [1, 2, 3].forEach(sd => { const phi = fit(train, K, TM.rng(100*sd + K)); px += perplexity(test, phi, K)/3; const T = top(phi, 5); um += umass(train, T)/3; Ts.push(T); });
      rows.push({K, px, um, st: (stability(Ts[0], Ts[1]) + stability(Ts[1], Ts[2]) + stability(Ts[0], Ts[2]))/3});
    }
    return (cache[Kt] = rows);
  }
  function draw(){
    svg.selectAll("*").remove();
    const Kt = +(tIn ? tIn.value : 4), rows = compute(Kt);
    const panels = [["held-out perplexity ↓", "px", "#5b9cff"], ["UMass coherence ↑", "um", "#2dd4bf"], ["stability across seeds ↑", "st", "#ffb454"]];
    const pw = 190, ph = 170, top0 = 36;
    const x = d3.scaleLinear().domain([2, KMAX]);
    panels.forEach(([name, key, col], i) => {
      const ox = 40 + i*(pw + 22);
      x.range([ox, ox + pw]);
      const ext = d3.extent(rows, r => r[key]), pad = (ext[1] - ext[0])*0.1 || 0.1;
      const y = d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad]).range([top0 + ph, top0]);
      svg.append("rect").attr("x", ox).attr("y", top0).attr("width", pw).attr("height", ph).attr("fill", "#11141b").attr("stroke", "#2a2f3a");
      svg.append("line").attr("x1", x(Kt)).attr("x2", x(Kt)).attr("y1", top0).attr("y2", top0 + ph).attr("stroke", "#9aa3b2").attr("stroke-dasharray", "3 3");
      svg.append("text").attr("x", x(Kt) + 3).attr("y", top0 + 12).attr("font-size", 9.5).attr("fill", "#9aa3b2").text("true K");
      svg.append("text").attr("x", ox).attr("y", top0 - 10).attr("font-size", 11).attr("fill", col).text(name);
      svg.append("path").datum(rows).attr("d", d3.line().x(r => x(r.K)).y(r => y(r[key]))).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2);
      const best = key === "px" ? d3.least(rows, r => r[key]) : d3.greatest(rows, r => r[key]);
      svg.append("g").selectAll("circle").data(rows).enter().append("circle").attr("cx", r => x(r.K)).attr("cy", r => y(r[key]))
        .attr("r", r => r === best ? 5 : 3).attr("fill", r => r === best ? "#e6e9ef" : col);
      x.ticks(KMAX - 2).forEach(t => svg.append("text").attr("x", x(t)).attr("y", top0 + ph + 14).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", "#9aa3b2").text(t));
      y.ticks(4).forEach(t => svg.append("text").attr("x", ox - 4).attr("y", y(t) + 3).attr("text-anchor", "end").attr("font-size", 9).attr("fill", "#9aa3b2").text(key === "px" ? t.toFixed(0) : t.toFixed(1)));
    });
    svg.append("text").attr("x", 340).attr("y", top0 + ph + 34).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", "#9aa3b2").text("number of topics K (each point: mean of 3 seeds)");
    const bp = d3.least(rows, r => r.px).K, bu = d3.greatest(rows, r => r.um).K, bs = d3.greatest(rows, r => r.st).K;
    const at = k => rows.find(r => r.K === k);
    if(out) out.innerHTML = `true K = <b>${Kt}</b> · best perplexity at K = <b>${bp}</b> (K = ${KMAX} is only ${((at(KMAX).px/at(Kt).px - 1)*100).toFixed(1)}% worse than K = ${Kt}) · ` +
      `best coherence at K = <b>${bu}</b> · most stable at K = <b>${bs}</b>`;
  }
  if(tIn) tIn.addEventListener("change", draw);
  draw();
})();
