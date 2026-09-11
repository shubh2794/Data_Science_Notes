/* transformers.viz.js — the visualizations on deep-learning/transformers.html
   (Deep Learning · part 6). Loaded after ../data.js → ../notes.js → dl-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the
   page, so a section can be reordered or removed without breaking the rest.

   Every numeric primitive — the attention block and its backward pass (part
   5), the positional encodings, the norms with their backward passes, the
   ungated and gated FFN, the pre-norm and post-norm block, the whole tiny
   decoder-only model, the parameter and FLOP counters, the schedule and the
   label-smoothing helpers — comes from dl-viz.js (DL.*). NOTHING here forks
   those. Read the TRANSFORMER header above DL.sinusoidalPE and THE ATTENTION
   LAYOUT LAW above DL.sdpa before touching any of it: one TOKEN per ROW,
   X is (n × d_model), heads split the LAST axis, masks are ADDITIVE and
   −Infinity, RoPE rotates ROWS of Q and K and never V.
   Page-specific numerics (task generators, the training loop, the decoding
   strategies, the fp16 rounding model) live in TR below.

     1  #bk-svg   one block as a tensor flow: shapes, parameters, MACs
     2  #tr-svg   a real matrix traced through one block, pre- and post-norm
     3  #pm-svg   the sinusoidal table and the wavelength of every column pair
     4  #sr-svg   PE(p+k) = PE(p)·M_k, measured; the offset kernel
     5  #le-svg   learned absolute positions: the Gram matrix and the wall
     6  #ro-svg   RoPE: the rotation, and q·k as a function of the offset only
     7  #rs-svg   RoPE's frequency spectrum against the context length
     8  #ab-svg   ALiBi slopes and T5-style relative buckets, as score biases
     9  #ex-svg   five positional schemes trained live and tested past the training length
    10  #np-svg   NoPE: the causal mask as a positional signal
    11  #rn-svg   the residual stream: norm growth and the identity Jacobian
    12  #pp-svg   pre-norm against post-norm: gradient scale per layer at init
    13  #ln-svg   LayerNorm against RMSNorm on real rows
    14  #ff-svg   the FFN's parameter share against d and the width ratio
    15  #gl-svg   gated units: the ⅔ correction and the three activations
    16  #km-svg   the FFN as key–value memory, on the trained model
    17  #fd-svg   the backward pass of every new primitive, audited
    18  #st-svg   encoder-only against decoder-only: masks and information flow
    19  #xa-svg   the encoder–decoder stack: cross-attention shapes and cost
    20  #em-svg   the √d embedding scale, measured against the positional table
    21  #un-svg   the unembedding: vocabulary share of parameters and MACs
    22  #pc-svg   parameters of five real configurations, by component
    23  #fl-svg   training FLOPs per token: the weight term and the n·d term
    24  #tf-svg   teacher forcing: the shift, the mask and the per-position loss
    25  #ls-svg   label smoothing: the target, the floor and the logit gap
    26  #sc-svg   the warmup-then-decay schedule, from the closed form
    27  #ad-svg   β₂ under a gradient spike; clipping
    28  #mp-svg   bf16 against fp16: the number line and the underflow
    29  #sp-svg   temperature, top-k and top-p on one distribution
    30  #dc-svg   decoding live from the trained model
    31  #dw-svg   depth against width at a fixed parameter budget
    32  #lv-svg   a two-layer decoder trained live: loss and attention
    33  #ll-svg   the logit lens on the trained model
    34  #f3-svg   the three families at one budget, counted exactly
    35  #bg-svg   the bugs that do not crash, measured

   Every number these print is recomputed from the data they draw.          */

/* ══════════ page-local helpers (deliberately NOT in dl-viz.js) ══════════ */
const TR = (function () {

  /* a (m × n) seeded Gaussian, scaled */
  function gauss(m, n, seed, s) {
    const r = DL.rng(seed || 2), A = DL.zeros2(m, n), sc = s === undefined ? 1 : s;
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) A[i][j] = DL.randn(r) * sc;
    return A;
  }
  function rowNorm(v) { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s); }
  function frobOf(M) { let s = 0; for (const r of M) for (const v of r) s += v * v; return Math.sqrt(s); }
  function maxAbs(M) { let s = 0; for (const r of M) for (const v of r) s = Math.max(s, Math.abs(v)); return s; }
  function scaleMat(M, s) { for (const r of M) for (let j = 0; j < r.length; j++) r[j] *= s; return M; }
  const pct = x => (100 * x).toFixed(x < 0.1 ? 2 : 1) + "%";

  /* small titles and notes, matching the other parts of the series */
  function title(g, x, y, t, col) {
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", 11.5)
      .attr("font-weight", 600).attr("fill", col || DC.ink).text(t);
  }
  function note(g, x, y, t, col, size, anchor) {
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", size || 10)
      .attr("fill", col || DC.muted).attr("text-anchor", anchor || "start").text(t);
  }
  /* a heat-map of a matrix, returning the group so the caller can annotate */
  function heat(g, M, x, y, cw, scale, opt) {
    const o = Object.assign({ stroke: null, ch: null }, opt || {});
    const n = M.length, m = M[0].length, ch = o.ch === null ? cw : o.ch;
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    const data = [];
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) data.push([j, i, M[i][j]]);
    gg.selectAll("rect").data(data).join("rect")
      .attr("x", d => d[0] * cw).attr("y", d => d[1] * ch)
      .attr("width", cw + 0.5).attr("height", ch + 0.5)
      .attr("shape-rendering", "crispEdges")
      .attr("fill", d => scale(d[2]))
      .attr("stroke", o.stroke).attr("stroke-width", o.stroke ? 0.4 : 0);
    return gg;
  }
  function strip(g, v, x, y, cw, scale, opt) {
    const o = Object.assign({ vertical: false, h: null }, opt || {});
    const h = o.h === null ? cw : o.h;
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    gg.selectAll("rect").data(v.map((val, i) => [i, val])).join("rect")
      .attr("x", d => o.vertical ? 0 : d[0] * cw).attr("y", d => o.vertical ? d[0] * cw : 0)
      .attr("width", o.vertical ? h : cw + 0.4).attr("height", o.vertical ? cw + 0.4 : h)
      .attr("shape-rendering", "crispEdges").attr("fill", d => scale(d[1]));
    return gg;
  }
  /* a labelled box in a flow diagram */
  function box(g, x, y, w, h, lines, opt) {
    const o = Object.assign({ stroke: DC.line, fill: DC.panel2, size: 10, color: DC.ink, rx: 4 }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    gg.append("rect").attr("width", w).attr("height", h).attr("rx", o.rx).attr("fill", o.fill).attr("stroke", o.stroke).attr("stroke-width", 1.1);
    const L = Array.isArray(lines) ? lines : [lines];
    const y0 = h / 2 - (L.length - 1) * (o.size + 2) / 2 + o.size / 2 - 1;
    L.forEach((t, i) => gg.append("text").attr("x", w / 2).attr("y", y0 + i * (o.size + 2)).attr("text-anchor", "middle")
      .attr("font-size", i === 0 ? o.size : o.size - 1).attr("fill", i === 0 ? o.color : DC.muted)
      .attr("font-family", i === 0 ? null : "SF Mono, Menlo, monospace").text(t));
    return gg;
  }
  const diverging = (lim) => d3.scaleDiverging(d3.interpolateRdBu).domain([lim, 0, -lim]);

  /* ── the synthetic tasks for the live models ─────────────────────────
     Vocabulary: 0 = BOS, V−1 = SEP, symbols 1 … V−2. Every task returns a
     sequence idx of length T and targets tgt aligned so that tgt[i] is
     the token to predict AT position i (i.e. idx[i+1]); −1 = no loss.  */
  function task(name, V, T, r) {
    const S = V - 2, sym = () => 1 + Math.floor(r() * S);
    const k = Math.floor((T - 2) / 2);
    let seq = [], mInd = 0;
    if (name === "copy" || name === "reverse" || name === "sort") {
      const s = []; for (let i = 0; i < k; i++) s.push(sym());
      const ans = s.slice();
      if (name === "reverse") ans.reverse();
      if (name === "sort") ans.sort((a, b) => a - b);
      seq = [0].concat(s, [V - 1], ans);
    } else if (name === "modadd") {
      seq = [0];
      while (seq.length < T + 1) { const a = sym(), b = sym(); seq.push(a, b, V - 1, 1 + ((a - 1 + b - 1) % S)); }
    } else if (name === "induction") {
      const m = 2 + Math.floor(r() * (k - 1)), pre = [];
      for (let i = 0; i < m; i++) pre.push(sym());
      seq = [0];
      while (seq.length < T + 1) seq.push(pre[(seq.length - 1) % m]);
      mInd = m;
    } else if (name.startsWith("shift")) {                  /* predict the token kS back: "shift1", "shift2" */
      for (let i = 0; i < T + 1; i++) seq.push(sym());
    }
    const kS = name.startsWith("shift") ? (+name.slice(5) || 1) : 0;
    while (seq.length < T + 1) seq.push(sym());
    seq = seq.slice(0, T + 1);
    const idx = seq.slice(0, T), tgt = [];
    for (let i = 0; i < T; i++) {
      const nxt = seq[i + 1];
      let on;
      if (name === "modadd") on = seq[i] === V - 1;
      else if (name === "induction") on = i >= mInd + 1;
      else if (kS) on = i >= kS;
      else on = i >= k + 1 && i <= 2 * k;
      tgt.push(on ? (kS ? idx[i - kS] : nxt) : -1);
    }
    return { idx: idx, tgt: tgt };
  }
  const TASKS = { copy: "copy the sequence after SEP", reverse: "reverse it after SEP", sort: "sort it after SEP",
                  induction: "repeat a prefix of random length", modadd: "a b = (a+b) mod S", shift1: "output the previous token", shift2: "output the token two back" };
  /* the extrapolation experiment of §09, shared with §10 so the cache hits */
  function exRun(pos, taskName, steps, L, seed) {
    return train({ task: taskName, steps: steps, B: 4, lr: 0.02, seed: seed, evalN: 24,
                   cfg: { V: 9, d: 16, L: L, h: 4, dff: 32, nCtx: 10, pos: pos, norm: "ln", pre: true } });
  }
  function evalAtLength(m, taskName, n, count, seed) {
    const r = DL.rng(seed || 77), set = [];
    for (let i = 0; i < count; i++) set.push(task(taskName, m.cfg.V, n, r));
    return evalLoss(m, set);
  }
  function accByPosition(m, taskName, n, count, seed) {
    const r = DL.rng(seed || 78), ok = new Array(n).fill(0), tot = new Array(n).fill(0);
    for (let t = 0; t < count; t++) {
      const ex = task(taskName, m.cfg.V, n, r); let f;
      try { f = DL.tfForward(m, ex.idx); } catch (e) { return null; }
      for (let i = 0; i < n; i++) { if (ex.tgt[i] < 0) continue; tot[i]++; if (argmax(f.logits[i]) === ex.tgt[i]) ok[i]++; }
    }
    return ok.map((v, i) => tot[i] ? v / tot[i] : NaN);
  }

  /* ── the training loop, shared by every live figure, cached by key ──── */
  const CACHE = {};
  function train(o) {
    const oo = Object.assign({ task: "copy", steps: 300, B: 8, lr: 0.02, warm: 20, seed: 1, wd: 0.01, beta2: 0.95,
                               cfg: {}, every: 10, evalN: 48, onStep: null }, o || {});
    const cfg = Object.assign({ V: 10, d: 16, L: 2, h: 2, dff: 32, nCtx: 10, pos: "learned", norm: "ln", pre: true,
                                gated: false, act: "gelu", tie: true, scaleEmb: true, seed: oo.seed }, oo.cfg);
    const key = JSON.stringify([oo.task, oo.steps, oo.B, oo.lr, oo.seed, oo.wd, oo.beta2, cfg]);
    if (CACHE[key]) return CACHE[key];
    const m = DL.tfInit(cfg), r = DL.rng(oo.seed * 7919 + 13), pl = DL.tfParamList(m), states = [];
    const curve = [], gnorms = [];
    const evalSet = []; const re = DL.rng(4242);
    for (let i = 0; i < oo.evalN; i++) evalSet.push(task(oo.task, cfg.V, cfg.nCtx, re));
    let last = 0;
    for (let s = 0; s < oo.steps; s++) {
      let acc = null, L = 0;
      for (let b = 0; b < oo.B; b++) {
        const ex = task(oo.task, cfg.V, cfg.nCtx, r), bw = DL.tfBackward(m, ex.idx, ex.tgt), gl = DL.tfGradList(bw.g, m);
        if (!acc) acc = gl.map(g => ({ k: g.k, M: Array.isArray(g.M[0]) ? g.M.map(rr => rr.map(v => v / oo.B)) : g.M.map(v => v / oo.B) }));
        else gl.forEach((g, i) => {
          if (Array.isArray(g.M[0])) { for (let p = 0; p < g.M.length; p++) for (let q = 0; q < g.M[0].length; q++) acc[i].M[p][q] += g.M[p][q] / oo.B; }
          else for (let p = 0; p < g.M.length; p++) acc[i].M[p] += g.M[p] / oo.B;
        });
        L += bw.loss / oo.B;
      }
      const lr = DL.lrAt("cosine", s, { peak: oo.lr, total: oo.steps, warm: oo.warm });
      const gn = DL.optMats(pl, acc, states, "adamw", { lr: lr, beta1: 0.9, beta2: oo.beta2, wd: oo.wd });
      last = L;
      if (s % oo.every === 0 || s === oo.steps - 1) { curve.push([s, L]); gnorms.push([s, gn]); }
    }
    const ev = evalLoss(m, evalSet);
    const res = { m: m, cfg: cfg, curve: curve, gnorms: gnorms, trainLoss: last, evalLoss: ev.loss, evalAcc: ev.acc, evalSet: evalSet, task: oo.task };
    CACHE[key] = res;
    return res;
  }
  function evalLoss(m, set) {
    let L = 0, c = 0, ok = 0;
    set.forEach(ex => {
      let f; try { f = DL.tfForward(m, ex.idx); } catch (e) { return; }
      for (let i = 0; i < ex.idx.length; i++) {
        if (ex.tgt[i] < 0) continue;
        L += DL.xent(f.logits[i], ex.tgt[i]); c++;
        if (argmax(f.logits[i]) === ex.tgt[i]) ok++;
      }
    });
    return { loss: c ? L / c : NaN, acc: c ? ok / c : NaN, n: c };
  }
  function argmax(v) { let k = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[k]) k = i; return k; }

  /* ── decoding strategies, on a logits row ───────────────────────────── */
  function sampleFrom(p, r) { let u = r(), c = 0; for (let i = 0; i < p.length; i++) { c += p[i]; if (u < c) return i; } return p.length - 1; }
  function truncate(logits, o) {                             /* → probabilities after temperature / top-k / top-p */
    const oo = Object.assign({ T: 1, k: 0, p: 1 }, o || {});
    let p = DL.softmax(logits, oo.T === 0 ? 1e-9 : oo.T);
    const order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]);
    const keep = new Array(p.length).fill(false);
    if (oo.k > 0) order.slice(0, oo.k).forEach(i => keep[i] = true); else keep.fill(true);
    if (oo.p < 1) {
      let c = 0; const keep2 = new Array(p.length).fill(false);
      for (const i of order) { keep2[i] = true; c += p[i]; if (c >= oo.p) break; }
      for (let i = 0; i < p.length; i++) keep[i] = keep[i] && keep2[i];
    }
    let s = 0; for (let i = 0; i < p.length; i++) { if (!keep[i]) p[i] = 0; s += p[i]; }
    return p.map(v => v / s);
  }
  /* beam search over the supervised positions: idx has the prompt, tgt marks where to decode */
  function beamSearch(m, idx, tgt, B) {
    const first = tgt.findIndex(t => t >= 0), last = tgt.length - 1 - tgt.slice().reverse().findIndex(t => t >= 0);
    let beams = [{ seq: idx.slice(0, first + 1), score: 0, toks: [] }];
    const history = [];
    for (let i = first; i <= last; i++) {
      const cand = [];
      beams.forEach(b => {
        const f = DL.tfForward(m, b.seq), lp = DL.logSoftmax(f.logits[i]);
        lp.forEach((v, t) => cand.push({ seq: b.seq.concat([t]), score: b.score + v, toks: b.toks.concat([t]) }));
      });
      cand.sort((a, b) => b.score - a.score);
      beams = cand.slice(0, B);
      history.push(cand.slice(0, Math.max(B, 6)).map(c => ({ toks: c.toks.slice(), score: c.score, kept: beams.indexOf(c) >= 0 })));
    }
    beams.forEach(b => b.seq = b.seq.slice(0, tgt.length));
    return { best: beams[0], beams: beams, history: history, first: first, last: last };
  }
  function seqLogProb(m, idx, toks, first) {
    const seq = idx.slice(0, first + 1); let lp = 0;
    toks.forEach((t, k) => { const f = DL.tfForward(m, seq), l = DL.logSoftmax(f.logits[first + k]); lp += l[t]; seq.push(t); });
    return lp;
  }
  function tokenLabel(v, V) { return v === 0 ? "BOS" : (v === V - 1 ? "SEP" : String(v)); }

  /* ── fp16 / bf16 rounding models (round-to-nearest-even on the mantissa) ── */
  function roundTo(x, mantBits, expBits) {
    if (x === 0 || !isFinite(x)) return x;
    const emax = Math.pow(2, expBits - 1) - 1, emin = 1 - emax;
    const a = Math.abs(x), e = Math.floor(Math.log2(a));
    const ee = Math.max(e, emin);                             /* subnormals share emin's spacing */
    const ulp = Math.pow(2, ee - mantBits);
    let q = Math.round(a / ulp);
    if (Math.abs(a / ulp - Math.floor(a / ulp) - 0.5) < 1e-12 && q % 2 === 1) q -= 1;   /* ties to even */
    const v = q * ulp;
    const maxv = (2 - Math.pow(2, -mantBits)) * Math.pow(2, emax);
    if (v > maxv) return Math.sign(x) * Infinity;
    return Math.sign(x) * v;
  }
  const fp16 = x => roundTo(x, 10, 5), bf16 = x => roundTo(x, 7, 8);

  /* the five configurations of §22–§23, exactly as DL.modelParams takes them */
  const CFGS = [
    { name: "125M-class", V: 50257, d: 768, L: 12, h: 12, dff: 3072, nCtx: 2048, pos: "learned", norm: "ln", gated: false, bias: true, tie: true, attnBias: true },
    { name: "350M-class", V: 50257, d: 1024, L: 24, h: 16, dff: 4096, nCtx: 2048, pos: "learned", norm: "ln", gated: false, bias: true, tie: true, attnBias: true },
    { name: "1.3B-class", V: 50257, d: 2048, L: 24, h: 16, dff: 8192, nCtx: 2048, pos: "learned", norm: "ln", gated: false, bias: true, tie: true, attnBias: true },
    { name: "7B-class", V: 32000, d: 4096, L: 32, h: 32, dff: 11008, nCtx: 4096, pos: "rope", norm: "rms", gated: true, bias: false, tie: false, attnBias: false },
    { name: "70B-class", V: 32000, d: 8192, L: 80, h: 64, g: 8, dff: 28672, nCtx: 4096, pos: "rope", norm: "rms", gated: true, bias: false, tie: false, attnBias: false }];
  const CO = { pre: DC.accent, post: DC.a2, good: DC.good, bad: DC.bad, alt: DC.violet, third: DC.teal, rose: DC.rose, lime: DC.lime };
  return { gauss: gauss, rowNorm: rowNorm, frobOf: frobOf, maxAbs: maxAbs, scaleMat: scaleMat, pct: pct,
           title: title, note: note, heat: heat, strip: strip, box: box, diverging: diverging,
           task: task, TASKS: TASKS, train: train, evalLoss: evalLoss, argmax: argmax, exRun: exRun, evalAtLength: evalAtLength, accByPosition: accByPosition,
           CFGS: CFGS, beamSearch: beamSearch, seqLogProb: seqLogProb, sampleFrom: sampleFrom, truncate: truncate, tokenLabel: tokenLabel, fp16: fp16, bf16: bf16, roundTo: roundTo, CO: CO };
})();

/* ═════════ 1 · #bk-svg — one block as a tensor flow. Every count comes from
   DL.mhaParams, DL.attnMacs, DL.ffnParams and DL.blockParams.  ═══════════ */
(function () {
  const svg = d3.select("#bk-svg"); if (svg.empty()) return;
  const W = 760, H = 470;
  const El = id => document.getElementById(id);
  function draw() {
    const d = +El("bk-d").value, h = +El("bk-h").value, rs = El("bk-r").value, n = +El("bk-n").value, wiring = El("bk-w").value;
    El("bk-nv").textContent = n;
    const gated = rs === "8/3", ratio = gated ? 8 / 3 : +rs, dff = Math.round(ratio * d), dk = d / h;
    const P = DL.mhaParams(d, dk, dk, h, h, false), Pf = DL.ffnParams(d, dff, { gated: gated, bias: !gated });
    const Mc = DL.attnMacs({ n: n, d: d, h: h, dh: dk, dff: dff, nmat: gated ? 3 : 2, causal: false });
    const normP = 2 * 2 * d, total = P.total + Pf + normP;
    const f = DL.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 }), g = f.g;
    const pre = wiring === "pre";
    const col = { x: 300, w: 160 };
    let y = 4;
    const step = 33;
    const flow = [];
    const add = (lines, o) => { flow.push({ y: y, lines: lines, o: o || {} }); y += step; };
    add([`X  (n × d) = (${n} × ${d})`, "input to the block"], { stroke: DC.accent });
    if (pre) add([`N₁(X)  (${n} × ${d})`, `${2 * d} params · per row`]);
    add([`Q, K, V = N(X)·W_Q, ·W_K, ·W_V`, `${DL.big(P.Wq + P.Wk + P.Wv)} params · ${DL.big(3 * n * d * d)} MACs`]);
    add([`split → ${h} heads × (${n} × ${dk})`, "a reshape, no arithmetic"]);
    add([`Sⱼ = QⱼKⱼᵀ/√${dk}  →  ${h} × (${n} × ${n})`, `${DL.big(Mc.quad / 2)} MACs · 0 params`], { stroke: DC.a2 });
    add([`Zⱼ = softmax(Sⱼ + mask)·Vⱼ`, `${DL.big(Mc.quad / 2)} MACs · concat → (${n} × ${h * dk})`], { stroke: DC.a2 });
    add([`Attn = concat·W_O  (${n} × ${d})`, `${DL.big(P.Wo)} params · ${DL.big(n * d * d)} MACs`]);
    add(pre ? [`X₁ = X + Attn`, "residual add"] : [`X₁ = N₁(X + Attn)`, `residual add, then ${2 * d}-param norm`], { stroke: DC.good });
    if (pre) add([`N₂(X₁)  (${n} × ${d})`, `${2 * d} params · per row`]);
    add(gated ? [`act(X·W_g) ⊙ (X·W_u)  (${n} × ${dff})`, `${DL.big(2 * d * dff)} params · ${DL.big(2 * n * d * dff)} MACs`]
              : [`act(X·W₁ + b₁)  (${n} × ${dff})`, `${DL.big(d * dff + dff)} params · ${DL.big(n * d * dff)} MACs`], { stroke: DC.violet });
    add(gated ? [`FFN = (·)·W_d  (${n} × ${d})`, `${DL.big(d * dff)} params · ${DL.big(n * d * dff)} MACs`]
              : [`FFN = (·)·W₂ + b₂  (${n} × ${d})`, `${DL.big(d * dff + d)} params · ${DL.big(n * d * dff)} MACs`], { stroke: DC.violet });
    add(pre ? [`Y = X₁ + FFN  (${n} × ${d})`, "residual add · same shape as X"] : [`Y = N₂(X₁ + FFN)  (${n} × ${d})`, `residual add, then ${2 * d}-param norm`], { stroke: DC.good });
    const bh = 28;
    flow.forEach((fl, i) => {
      TR.box(g, col.x, fl.y, col.w + 100, bh, fl.lines, { stroke: fl.o.stroke || DC.line, size: 10.5 });
      if (i + 1 < flow.length) DL.arrow(g, col.x + (col.w + 100) / 2, fl.y + bh, col.x + (col.w + 100) / 2, flow[i + 1].y, { color: DC.muted, w: 1, head: 4 });
    });
    /* residual bypass arrows */
    const iX = 0, iA = flow.findIndex(fl => fl.lines[0].startsWith("X₁")), iY = flow.length - 1;
    const bx = col.x - 40;
    [[iX, iA], [iA, iY]].forEach(([a, b], k) => {
      const ya = flow[a].y + bh / 2, yb = flow[b].y + bh / 2;
      g.append("path").attr("d", `M${col.x},${ya} L${bx - k * 14},${ya} L${bx - k * 14},${yb} L${col.x - 2},${yb}`)
        .attr("fill", "none").attr("stroke", DC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
      TR.note(g, bx - k * 14 - 6, (ya + yb) / 2, "residual", DC.good, 9, "end");
    });
    /* left column: the legend of what each colour is */
    TR.title(g, 6, 16, "one block, " + (pre ? "pre-norm" : "post-norm"));
    DL.legend(g, [{ label: "token mixing (attention)", color: DC.a2 }, { label: "per-token (FFN)", color: DC.violet },
                  { label: "residual sums", color: DC.good }, { label: "input", color: DC.accent }], 6, 34, { gap: 15, font: 10 });
    /* bottom bars: parameter share, MAC share */
    const by = 4 + flow.length * step + 6, bw = 740;
    const parts = [
      { label: "attention projections", v: P.total, m: Mc.proj, c: DC.a2 },
      { label: "attention n² term", v: 0, m: Mc.quad, c: DC.rose },
      { label: "FFN", v: Pf, m: Mc.ffn, c: DC.violet },
      { label: "norms", v: normP, m: 0, c: DC.muted }];
    const drawBar = (yy, key, tot, lab) => {
      let x = 0;
      TR.note(g, 6, yy - 4, lab, DC.ink, 10);
      parts.forEach(p => {
        const w = bw * p[key] / tot; if (w <= 0) return;
        g.append("rect").attr("x", 6 + x).attr("y", yy).attr("width", w).attr("height", 14).attr("fill", p.c).attr("fill-opacity", 0.85);
        if (w > 60) TR.note(g, 6 + x + 4, yy + 11, `${p.label} ${TR.pct(p[key] / tot)}`, DC.bg, 9);
        x += w;
      });
    };
    drawBar(by, "v", total, `parameters of the block: ${DL.commas(total)}`);
    drawBar(by + 36, "m", Mc.total, `MACs of the block at n = ${n}: ${DL.big(Mc.total)}`);
    El("bk-readout").innerHTML =
      `d = ${d}, h = ${h}, dₖ = ${dk}, d_ff = ${dff}${gated ? " (gated, three matrices)" : ""}, n = ${n}. ` +
      `Parameters: attention <b>${DL.commas(P.total)}</b> (${TR.pct(P.total / total)}), FFN <b>${DL.commas(Pf)}</b> (${TR.pct(Pf / total)}), norms ${normP}. ` +
      `MACs per forward pass: projections ${DL.big(Mc.proj)}, the n² term ${DL.big(Mc.quad)} (<b>${TR.pct(Mc.quad / Mc.total)}</b> of the block), FFN ${DL.big(Mc.ffn)} (${TR.pct(Mc.ffn / Mc.total)}). ` +
      `The quadratic term overtakes everything else only at n = ${DL.commas(DL.macCrossover({ d: d, h: h, dh: dk, dff: dff, nmat: gated ? 3 : 2 }, "rest"))} — ` +
      `<a href="attention.html#crossover">part 5 §19</a> derives that number; this figure only reads it.`;
  }
  ["bk-d", "bk-h", "bk-r", "bk-w"].forEach(id => d3.select("#" + id).on("change", draw));
  d3.select("#bk-n").on("input", draw);
  draw();
})();

/* ═════════ 2 · #tr-svg — a real matrix through one block.  ═══════════════ */
(function () {
  const svg = d3.select("#tr-svg"); if (svg.empty()) return;
  const W = 760, H = 430, n = 6, d = 16, h = 2, dff = 32;
  const El = id => document.getElementById(id);
  function draw() {
    const wiring = El("tr-w").value, norm = El("tr-n").value, ff = El("tr-f").value, sc = +El("tr-s").value, tok = +El("tr-t").value, seed = +El("tr-seed").value;
    El("tr-sv").textContent = sc.toFixed(2); El("tr-tv").textContent = tok; El("tr-seedv").textContent = seed;
    const gated = ff === "silu-g", act = gated ? "silu" : ff;
    const P = DL.blockInit(d, h, dff, { norm: norm, pre: wiring === "pre", act: act, gated: gated, seed: seed });
    [P.attn.Wq, P.attn.Wk, P.attn.Wv, P.attn.Wo].forEach(M => TR.scaleMat(M, sc));
    (gated ? [P.ffn.Wg, P.ffn.Wu, P.ffn.Wd] : [P.ffn.W1, P.ffn.W2]).forEach(M => TR.scaleMat(M, sc));
    const X = TR.gauss(n, d, seed * 3 + 1, 1);
    const out = DL.blockForward(X, P, { mask: DL.causalMask(n) }), st = out.st;
    const stages = wiring === "pre"
      ? [["X", X], ["N₁(X)", norm === "rms" ? DL.rmsNorm(X, P.n1.g) : DL.layerNorm(X, P.n1.g, P.n1.b)], ["Attn", st.attnOut], ["X₁", st.X1], ["N₂(X₁)", norm === "rms" ? DL.rmsNorm(st.X1, P.n2.g) : DL.layerNorm(st.X1, P.n2.g, P.n2.b)], ["FFN", st.ffnOut], ["Y", out.Y]]
      : [["X", X], ["Attn", st.attnOut], ["X+Attn", X.map((r, i) => r.map((v, j) => v + st.attnOut[i][j]))], ["X₁=N₁(·)", st.X1], ["FFN", st.ffnOut], ["X₁+FFN", st.X1.map((r, i) => r.map((v, j) => v + st.ffnOut[i][j]))], ["Y=N₂(·)", out.Y]];
    const f = DL.frame(svg, W, H, { l: 44, r: 12, t: 24, b: 30 }), g = f.g;
    /* upper: grouped bars of row norms */
    const uh = 190, norms = stages.map(s => s[1].map(TR.rowNorm));
    const ymax = Math.max(Math.sqrt(d) * 1.2, ...norms.flat()) * 1.08;
    const y = d3.scaleLinear().domain([0, ymax]).range([uh, 0]);
    const x0 = d3.scaleBand().domain(stages.map((_, i) => i)).range([0, f.iw]).paddingInner(0.25);
    const x1 = d3.scaleBand().domain(d3.range(n)).range([0, x0.bandwidth()]).paddingInner(0.1);
    DL.gridY(g, y, f.iw, 4); DL.axisL(g, y, 4, "‖row‖₂");
    g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y(Math.sqrt(d))).attr("y2", y(Math.sqrt(d))).attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    TR.note(g, f.iw - 2, y(Math.sqrt(d)) - 4, "√d = " + Math.sqrt(d).toFixed(2), DC.muted, 9, "end");
    const colOf = i => [DC.accent, DC.teal, DC.a2, DC.good, DC.teal, DC.violet, DC.good][i];
    stages.forEach((s, i) => {
      norms[i].forEach((v, r) => g.append("rect").attr("x", x0(i) + x1(r)).attr("y", y(v)).attr("width", x1.bandwidth()).attr("height", uh - y(v))
        .attr("fill", colOf(i)).attr("fill-opacity", r === tok ? 1 : 0.45));
      TR.note(g, x0(i) + x0.bandwidth() / 2, uh + 13, s[0], DC.ink, 10, "middle");
      TR.note(g, x0(i) + x0.bandwidth() / 2, uh + 25, "mean " + DL.fmt(DL.mean(norms[i]), 2), DC.muted, 9, "middle");
    });
    TR.title(g, 0, -8, `row norms through one ${wiring}-norm block (${norm === "rms" ? "RMSNorm" : "LayerNorm"}, ${gated ? "SwiGLU" : act.toUpperCase()} FFN), n = ${n}, d = ${d}, weights × ${sc.toFixed(2)}`);
    /* lower: strips of the selected token */
    const ly = uh + 52, lim = Math.max(...stages.map(s => Math.max(...s[1][tok].map(Math.abs))));
    const scl = TR.diverging(lim), cw = Math.min(13, (f.iw - 90) / d);
    TR.title(g, 0, ly - 6, `token ${tok}'s ${d}-vector at each stage (colour scale ±${DL.fmt(lim, 2)})`);
    stages.forEach((s, i) => {
      const yy = ly + i * 20;
      TR.note(g, 0, yy + 10, s[0], DC.muted, 9.5);
      TR.strip(g, s[1][tok], 70, yy, cw, scl, { h: 14 });
      TR.note(g, 70 + cw * d + 8, yy + 10, "‖·‖ = " + DL.fmt(norms[i][tok], 2), DC.ink, 9);
    });
    const ratio = DL.mean(norms[wiring === "pre" ? 5 : 4]) / DL.mean(norms[wiring === "pre" ? 2 : 1]);
    const growth = DL.mean(norms[6]) / DL.mean(norms[0]);
    El("tr-readout").innerHTML =
      stages.map((s, i) => `${s[0]} <b>${DL.fmt(DL.mean(norms[i]), 2)}</b>`).join(" · ") +
      ` (mean row norms). The FFN writes <b>${DL.fmt(ratio, 2)}×</b> as much as attention does at this initialisation. ` +
      (wiring === "pre"
        ? `Pre-norm: the output rows are <b>${DL.fmt(growth, 2)}×</b> the input rows on average — nothing rescaled the sum, so the stream grows (§11 follows it across a stack). The two normalised stages sit at ${norm === "rms" ? "≈" : "exactly"} √d = ${Math.sqrt(d).toFixed(2)}${norm === "rms" ? " when the rows are near zero-mean" : ""}.`
        : `Post-norm: every output row is pinned to ${norm === "rms" ? "≈" : ""}√d = ${Math.sqrt(d).toFixed(2)} whatever the input scale was, because the residual path itself passed through N₂. Nothing the block wrote survives at its own scale.`);
  }
  ["tr-w", "tr-n", "tr-f"].forEach(id => d3.select("#" + id).on("change", draw));
  ["tr-s", "tr-t", "tr-seed"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 3 · #pm-svg — the sinusoidal table and its wavelengths.  ══════ */
(function () {
  const svg = d3.select("#pm-svg"); if (svg.empty()) return;
  const W = 760, H = 440;
  const El = id => document.getElementById(id);
  function draw() {
    const n = +El("pm-n").value, d = +El("pm-d").value, base = +El("pm-b").value;
    const pairs = d / 2, iSel = Math.min(+El("pm-i").value, pairs - 1);
    El("pm-i").max = pairs - 1; El("pm-nv").textContent = n; El("pm-iv").textContent = iSel;
    const P = DL.sinusoidalPE(n, d, base), th = DL.ropeFreqs(d, base), lam = th.map(w => 2 * Math.PI / w);
    const f = DL.frame(svg, W, H, { l: 44, r: 12, t: 22, b: 30 }), g = f.g;
    /* heat map: n rows (subsampled if large) × d columns */
    const hw = 400, hh = 190, cw = hw / d, rows = Math.min(n, 190), ch = hh / rows;
    const sub = []; for (let i = 0; i < rows; i++) sub.push(P[Math.floor(i * n / rows)]);
    TR.title(g, 0, -8, `P (${n} × ${d}), base ${DL.commas(base)} — rows are positions, columns are dimensions`);
    TR.heat(g, sub, 0, 0, cw, TR.diverging(1), { ch: ch });
    g.append("rect").attr("x", 2 * iSel * cw).attr("y", 0).attr("width", 2 * cw).attr("height", hh).attr("fill", "none").attr("stroke", DC.lime).attr("stroke-width", 1.5);
    TR.note(g, 0, hh + 12, "col 0", DC.muted, 9); TR.note(g, hw, hh + 12, "col " + (d - 1), DC.muted, 9, "end");
    TR.note(g, -4, 8, "pos 0", DC.muted, 9, "end"); TR.note(g, -4, hh, "pos " + (n - 1), DC.muted, 9, "end");
    /* middle-right: the highlighted pair against position */
    const rx = hw + 30, rw = f.iw - rx, rh = 84;
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    TR.title(gg, 0, -8, `pair ${iSel}: columns ${2 * iSel} (sin) and ${2 * iSel + 1} (cos)`, DC.lime);
    const xs = d3.scaleLinear().domain([0, n - 1]).range([0, rw]), ys = d3.scaleLinear().domain([-1, 1]).range([rh, 0]);
    DL.axisB(gg, xs, rh, 4, "position"); DL.axisL(gg, ys, 3);
    DL.curve(gg, P.map((r, i) => [xs(i), ys(r[2 * iSel])]), { stroke: DC.accent, w: 1.5 });
    DL.curve(gg, P.map((r, i) => [xs(i), ys(r[2 * iSel + 1])]), { stroke: DC.a2, w: 1.5 });
    TR.note(gg, 0, rh + 28, `wavelength λ = 2π·${DL.commas(base)}^(${2 * iSel}/${d}) = ${DL.sig(lam[iSel], 4)} positions`, DC.ink, 9.5);
    TR.note(gg, 0, rh + 40, `so ${DL.sig(n / lam[iSel], 3)} cycles inside n = ${n}`, DC.muted, 9.5);
    /* lower: wavelength per pair, log axis */
    const ly = hh + 44, lh = f.ih - ly, lw = f.iw;
    const g2 = g.append("g").attr("transform", `translate(0,${ly})`);
    TR.title(g2, 0, -6, "wavelength of each pair (log axis) — pairs below the dashed line complete ≥ 1 cycle inside n");
    const xl = d3.scaleLinear().domain([0, pairs - 1]).range([0, lw]), yl = d3.scaleLog().domain([2 * Math.PI * 0.8, 2 * Math.PI * base * 1.5]).range([lh, 0]);
    DL.gridY(g2, yl, lw, 4); DL.axisB(g2, xl, lh, Math.min(8, pairs), "pair index i"); DL.axisL(g2, yl, 4, "λᵢ", v => DL.big(v));
    DL.curve(g2, lam.map((l, i) => [xl(i), yl(l)]), { stroke: DC.violet, w: 1.8 });
    g2.selectAll("circle").data(lam).join("circle").attr("cx", (l, i) => xl(i)).attr("cy", l => yl(l)).attr("r", 2.4).attr("fill", (l, i) => i === iSel ? DC.lime : DC.violet);
    g2.append("line").attr("x1", 0).attr("x2", lw).attr("y1", yl(n)).attr("y2", yl(n)).attr("stroke", DC.ink).attr("stroke-dasharray", "4 3");
    TR.note(g2, lw - 2, yl(n) - 4, "n = " + n, DC.ink, 9.5, "end");
    const below = lam.filter(l => l <= n).length;
    El("pm-readout").innerHTML =
      `d = ${d} gives ${pairs} pairs. Shortest wavelength <b>2π = ${(2 * Math.PI).toFixed(3)}</b> positions (pair 0), longest <b>${DL.sig(lam[pairs - 1], 4)}</b> (pair ${pairs - 1}, ≈ 2π·base^((d−2)/d)). ` +
      `At n = ${n}, <b>${below}</b> of ${pairs} pairs complete at least one cycle; the other <b>${pairs - below}</b> vary by less than one cycle across the whole sequence and act as slowly drifting, almost absolute coordinates. ` +
      `Pair ${iSel}: λ = ${DL.sig(lam[iSel], 4)}, ${DL.sig(n / lam[iSel], 3)} cycles. The slowest pair alone distinguishes every position up to ${DL.big(lam[pairs - 1])}; beyond that no single pair does, though the full row still varies.`;
  }
  ["pm-d", "pm-b"].forEach(id => d3.select("#" + id).on("change", draw));
  ["pm-n", "pm-i"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 4 · #sr-svg — PE(p+k) = PE(p)·M_k. The fit is ordinary least
   squares on a table long enough (3d positions) for PᵀP to be full rank;
   the closed form is DL.peShiftMatrix.  ════════════════════════════════════ */
(function () {
  const svg = d3.select("#sr-svg"); if (svg.empty()) return;
  const W = 760, H = 420;
  const El = id => document.getElementById(id);
  function draw() {
    const k = +El("sr-k").value, d = +El("sr-d").value, n = +El("sr-n").value, base = +El("sr-b").value;
    El("sr-kv").textContent = k; El("sr-nv").textContent = n;
    const M = DL.peShiftMatrix(k, d, base);
    /* least-squares fit on a long table: rows p with 0 ≤ p, p + k < nf */
    const nf = 3 * d + 8, Pf = DL.sinusoidalPE(nf, d, base), A = [], B = [];
    for (let p = 0; p < nf; p++) if (p + k >= 0 && p + k < nf) { A.push(Pf[p]); B.push(Pf[p + k]); }
    /* pair by pair: the two columns of pair i are regressed on the two shifted
       columns. The block structure is assumed; the angles are not. A full d×d
       fit is not identifiable from a short table: the slow pairs barely vary
       over 3d positions, so their columns are nearly collinear.           */
    const Mfit = DL.zeros2(d, d);
    for (let i = 0; 2 * i + 1 < d; i++) {
      const Ai = A.map(r => [r[2 * i], r[2 * i + 1]]), Bi = B.map(r => [r[2 * i], r[2 * i + 1]]);
      const At = DL.transpose(Ai), Gi = DL.matmul(At, Ai), Mi = DL.matmul(DL.pinvSym(Gi, 1e-14), DL.matmul(At, Bi));
      Mfit[2 * i][2 * i] = Mi[0][0]; Mfit[2 * i][2 * i + 1] = Mi[0][1]; Mfit[2 * i + 1][2 * i] = Mi[1][0]; Mfit[2 * i + 1][2 * i + 1] = Mi[1][1];
    }
    let res = 0; for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) res = Math.max(res, Math.abs(Mfit[i][j] - M[i][j]));
    /* the kernel and the score matrix at display size n */
    const P = DL.sinusoidalPE(n, d, base), th = DL.ropeFreqs(d, base);
    const kern = off => th.reduce((s, w) => s + Math.cos(off * w), 0);
    const S = DL.zeros2(n, n);
    for (let p = 0; p < n; p++) { const row = DL.vecmat(P[p], M); for (let q = 0; q < n; q++) S[p][q] = DL.dot(row, P[q]); }
    let hit = 0, cnt = 0;
    for (let p = 0; p < n; p++) { const t = p + k; if (t < 0 || t >= n) continue; cnt++; let am = 0; for (let q = 1; q < n; q++) if (S[p][q] > S[p][am]) am = q; if (am === t) hit++; }
    const f = DL.frame(svg, W, H, { l: 8, r: 8, t: 22, b: 8 }), g = f.g;
    /* left: M_k and the fit */
    const cw = Math.min(5, 150 / d), lim = 1;
    TR.title(g, 0, -8, `M_k, k = ${k}, closed form (${d} × ${d})`);
    TR.heat(g, M, 0, 0, cw, TR.diverging(lim));
    TR.title(g, 0, d * cw + 22, "the same matrix, fitted pair by pair by least squares");
    TR.heat(g, Mfit, 0, d * cw + 30, cw, TR.diverging(lim));
    TR.note(g, 0, 2 * d * cw + 46, `max |fit − closed form| = ${DL.fmtE(res, 1)}`, DC.good, 10);
    TR.note(g, 0, 2 * d * cw + 60, `fitted on ${A.length} rows of a ${nf}-row table`, DC.muted, 9);
    /* middle: kernel */
    const mx = 200, mw = 250, mh = 150;
    const gm = g.append("g").attr("transform", `translate(${mx},0)`);
    TR.title(gm, 0, -8, "offset kernel  ∑ᵢ cos(Δ·ωᵢ)  against Δ = p + k − p′");
    const offs = d3.range(-n + 1, n), kv = offs.map(kern), kv2 = offs.map(o => DL.ropeFreqs(d, base === 100 ? 10000 : 100).reduce((s, w) => s + Math.cos(o * w), 0));
    const xk = d3.scaleLinear().domain([-n + 1, n - 1]).range([0, mw]), yk = d3.scaleLinear().domain([Math.min(...kv, ...kv2) * 1.1, d / 2 * 1.08]).range([mh, 0]);
    DL.gridY(gm, yk, mw, 4); DL.axisB(gm, xk, mh, 6, "Δ"); DL.axisL(gm, yk, 4);
    DL.curve(gm, offs.map((o, i) => [xk(o), yk(kv2[i])]), { stroke: DC.muted, w: 1, dash: "3 3" });
    DL.curve(gm, offs.map((o, i) => [xk(o), yk(kv[i])]), { stroke: DC.violet, w: 1.8 });
    gm.append("line").attr("x1", xk(0)).attr("x2", xk(0)).attr("y1", 0).attr("y2", mh).attr("stroke", DC.line);
    TR.note(gm, xk(0) + 4, 10, `peak d/2 = ${d / 2}`, DC.violet, 9.5);
    DL.legend(gm, [{ label: `base ${DL.commas(base)}`, color: DC.violet }, { label: `base ${DL.commas(base === 100 ? 10000 : 100)} (reference)`, color: DC.muted }], 4, mh - 24, { gap: 12, font: 9 });
    TR.note(gm, 0, mh + 32, `kernel at Δ = 0: ${DL.fmt(kern(0), 2)}, at ±1: ${DL.fmt(kern(1), 2)}, at ±2: ${DL.fmt(kern(2), 2)}`, DC.ink, 9.5);
    TR.note(gm, 0, mh + 45, "short wavelengths sharpen the peak; long ones widen the base", DC.muted, 9);
    /* right: score matrix */
    const sx = 480, sc = Math.min(11, 260 / n);
    const gs = g.append("g").attr("transform", `translate(${sx},0)`);
    TR.title(gs, 0, -8, `scores P[p]·M_k·P[p′]ᵀ, (${n} × ${n})`);
    const smax = Math.max(...S.flat()), smin = Math.min(...S.flat());
    TR.heat(gs, S, 0, 0, sc, d3.scaleSequential(d3.interpolateViridis).domain([smin, smax]));
    TR.note(gs, 0, n * sc + 12, "rows: query position p · columns: key position p′", DC.muted, 9);
    TR.note(gs, 0, n * sc + 24, `bright diagonal at p′ = p ${k >= 0 ? "+ " + k : "− " + (-k)}`, DC.a2, 9.5);
    TR.note(gs, 0, n * sc + 36, `argmax exactly at the offset in ${hit}/${cnt} rows`, DC.good, 9.5);
    El("sr-readout").innerHTML =
      `k = ${k}, d = ${d}, base ${DL.commas(base)}. Fitting each 2×2 block of M_k from the table's own columns by least squares (${A.length} equations per block, block structure assumed, angles free) recovers the closed form to <b>${DL.fmtE(res, 1)}</b>, max absolute entry difference. ` +
      `The offset kernel peaks at <b>${DL.fmt(kern(0), 2)} = d/2</b> and falls to ${DL.fmt(kern(1), 2)} one position away and ${DL.fmt(kern(2), 2)} two away. ` +
      `A head with W_Q·W_Kᵀ = M_k puts its argmax exactly ${Math.abs(k)} position${Math.abs(k) === 1 ? "" : "s"} ${k >= 0 ? "ahead" : "back"} in <b>${hit} of ${cnt}</b> rows that have such a position — one fixed matrix, every position.`;
  }
  ["sr-d", "sr-b"].forEach(id => d3.select("#" + id).on("change", draw));
  ["sr-k", "sr-n"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 5 · #le-svg — learned against sinusoidal positions, both trained
   live on the copy task through TR.train (cached).  ══════════════════════ */
(function () {
  const svg = d3.select("#le-svg"); if (svg.empty()) return;
  const W = 760, H = 400;
  const El = id => document.getElementById(id);
  function gram(P, cos) {
    const n = P.length, G = DL.zeros2(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      G[i][j] = DL.dot(P[i], P[j]);
      if (cos) G[i][j] /= (TR.rowNorm(P[i]) * TR.rowNorm(P[j])) || 1;
    }
    return G;
  }
  function adj(G, gap) { let s = 0, c = 0; for (let i = 0; i + gap < G.length; i++) { s += G[i][i + gap]; c++; } return s / c; }
  function draw() {
    const steps = +El("le-s").value, seed = +El("le-seed").value, cos = El("le-w").value === "cos";
    El("le-seedv").textContent = seed;
    const A = TR.train({ task: "copy", steps: steps, seed: seed, cfg: { pos: "learned" } });
    const B = TR.train({ task: "copy", steps: steps, seed: seed, cfg: { pos: "sin" } });
    const n = A.cfg.nCtx, d = A.cfg.d;
    const GA = gram(A.m.Pos, cos), GB = gram(DL.sinusoidalPE(n, d), cos);
    const f = DL.frame(svg, W, H, { l: 8, r: 8, t: 22, b: 8 }), g = f.g;
    const cw = 20;
    const lim = cos ? 1 : Math.max(TR.maxAbs(GA), TR.maxAbs(GB));
    const scl = TR.diverging(lim);
    TR.title(g, 0, -8, `learned P·Pᵀ (${n} × ${n})${cos ? ", cosine" : ""} after ${steps} steps`);
    TR.heat(g, GA, 0, 0, cw, scl, { stroke: DC.bg });
    TR.note(g, 0, n * cw + 14, `adjacent (|i−j| = 1): ${DL.fmt(adj(GA, 1), 3)}`, DC.ink, 9.5);
    TR.note(g, 0, n * cw + 26, `distant (|i−j| = ${n - 2}): ${DL.fmt(adj(GA, n - 2), 3)}`, DC.ink, 9.5);
    TR.note(g, 0, n * cw + 38, `diagonal mean: ${DL.fmt(adj(GA, 0), 3)} · ${n * d} parameters`, DC.muted, 9.5);
    const bx = 250;
    TR.title(g, bx, -8, `sinusoidal P·Pᵀ (${n} × ${n})${cos ? ", cosine" : ""}, no training`);
    TR.heat(g, GB, bx, 0, cw, scl, { stroke: DC.bg });
    TR.note(g, bx, n * cw + 14, `adjacent: ${DL.fmt(adj(GB, 1), 3)}`, DC.ink, 9.5);
    TR.note(g, bx, n * cw + 26, `distant: ${DL.fmt(adj(GB, n - 2), 3)}`, DC.ink, 9.5);
    TR.note(g, bx, n * cw + 38, `diagonal: ${DL.fmt(adj(GB, 0), 3)} = d/2 · 0 parameters`, DC.muted, 9.5);
    /* right: loss curves */
    const rx = 500, rw = f.iw - rx, rh = 150;
    const gr = g.append("g").attr("transform", `translate(${rx},0)`);
    TR.title(gr, 0, -8, "training loss, copy task");
    const allv = A.curve.concat(B.curve).map(c => c[1]);
    const x = d3.scaleLinear().domain([0, steps]).range([0, rw]), y = d3.scaleLog().domain([Math.max(1e-4, Math.min(...allv) / 2), Math.max(...allv) * 1.5]).range([rh, 0]).clamp(true);
    DL.gridY(gr, y, rw, 4); DL.axisB(gr, x, rh, 4, "step"); DL.axisL(gr, y, 4, "loss", v => Math.abs(Math.log10(v) % 1) < 1e-9 ? DL.fmtE(v, 0) : "");
    DL.curve(gr, A.curve.map(c => [x(c[0]), y(Math.max(c[1], 1e-4))]), { stroke: DC.accent, w: 1.8 });
    DL.curve(gr, B.curve.map(c => [x(c[0]), y(Math.max(c[1], 1e-4))]), { stroke: DC.a2, w: 1.8 });
    gr.append("line").attr("x1", 0).attr("x2", rw).attr("y1", y(Math.log(A.cfg.V - 2))).attr("y2", y(Math.log(A.cfg.V - 2))).attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    TR.note(gr, rw, y(Math.log(A.cfg.V - 2)) - 4, "chance ln " + (A.cfg.V - 2), DC.muted, 9, "end");
    DL.legend(gr, [{ label: "learned table", color: DC.accent }, { label: "sinusoidal", color: DC.a2 }], 4, rh - 26, { gap: 12, font: 9.5 });
    /* the wall */
    const wy = rh + 50, cell = 20;
    TR.title(gr, 0, wy - 8, "rows of the learned table");
    for (let i = 0; i <= n; i++) {
      const exists = i < n;
      gr.append("rect").attr("x", i * cell).attr("y", wy).attr("width", cell - 2).attr("height", 18).attr("rx", 2)
        .attr("fill", exists ? DC.accent : "none").attr("fill-opacity", 0.8).attr("stroke", exists ? "none" : DC.bad).attr("stroke-dasharray", exists ? null : "3 2");
      TR.note(gr, i * cell + cell / 2 - 1, wy + 13, exists ? String(i) : "?", exists ? DC.bg : DC.bad, 9, "middle");
    }
    TR.note(gr, 0, wy + 34, `P[${n}] does not exist: length ${n + 1} raises an index error`, DC.bad, 9.5);
    TR.note(gr, 0, wy + 46, `the sinusoidal formula evaluates at ${n}, at 10⁶, at any p`, DC.a2, 9.5);
    let err = "";
    try { DL.tfForward(A.m, new Array(n + 1).fill(1)); } catch (e) { err = e.message; }
    El("le-readout").innerHTML =
      `Seed ${seed}, ${steps} steps, copy task, n = ${n}, d = ${d}. Final training loss: learned table <b>${DL.fmtE(A.trainLoss, 2)}</b>, sinusoidal <b>${DL.fmtE(B.trainLoss, 2)}</b> (held-out: ${DL.fmtE(A.evalLoss, 2)} and ${DL.fmtE(B.evalLoss, 2)}). ` +
      `${cos ? "Cosine" : "Gram"} of adjacent positions: learned <b>${DL.fmt(adj(GA, 1), 3)}</b>, sinusoidal <b>${DL.fmt(adj(GB, 1), 3)}</b>; of the most distant pair: ${DL.fmt(adj(GA, n - 2), 3)} and ${DL.fmt(adj(GB, n - 2), 3)}. ` +
      `The learned table costs ${n * d} parameters and is exactly as banded as this task made it. Evaluating the learned model at length ${n + 1} throws: <i>${err.replace(/^tfForward: /, "")}</i>.`;
  }
  ["le-s", "le-w"].forEach(id => d3.select("#" + id).on("change", draw));
  d3.select("#le-seed").on("input", draw);
  draw();
})();

/* ═════════ 6 · #ro-svg — RoPE on rows.  ══════════════════════════════════ */
(function () {
  const svg = d3.select("#ro-svg"); if (svg.empty()) return;
  const W = 760, H = 420;
  const El = id => document.getElementById(id);
  function draw() {
    const d = +El("ro-d").value, n = +El("ro-n").value, base = +El("ro-b").value, shift = +El("ro-s").value;
    const pairs = d / 2, iSel = Math.min(+El("ro-i").value, pairs - 1);
    El("ro-i").max = pairs - 1; El("ro-nv").textContent = n; El("ro-iv").textContent = iSel; El("ro-sv").textContent = shift;
    const q = TR.gauss(1, d, 5, 1)[0], kk = TR.gauss(1, d, 6, 1)[0], th = DL.ropeFreqs(d, base);
    const scoreMat = pairing => {
      const S = DL.zeros2(n, n);
      for (let m = 0; m < n; m++) { const qm = DL.rope([q], { base: base, offset: m, pairing: pairing })[0]; for (let nn = 0; nn < n; nn++) S[m][nn] = DL.dot(qm, DL.rope([kk], { base: base, offset: nn, pairing: pairing })[0]); }
      return S;
    };
    const resid = pairing => {
      let w = 0;
      for (let m = 0; m < n; m++) for (let nn = 0; nn < n; nn++) {
        const a = DL.dot(DL.rope([q], { base: base, offset: m, pairing: pairing })[0], DL.rope([kk], { base: base, offset: nn, pairing: pairing })[0]);
        const b = DL.dot(DL.rope([q], { base: base, offset: m + shift, pairing: pairing })[0], DL.rope([kk], { base: base, offset: nn + shift, pairing: pairing })[0]);
        w = Math.max(w, Math.abs(a - b));
      }
      return w;
    };
    const SA = scoreMat("adjacent"), SH = scoreMat("half"), rA = resid("adjacent"), rH = resid("half");
    const f = DL.frame(svg, W, H, { l: 8, r: 8, t: 22, b: 8 }), g = f.g;
    /* left: the circle */
    const cx = 80, cy = 90, R = 70;
    TR.title(g, 0, -8, `pair ${iSel} of q, rotated by position`);
    g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", R).attr("fill", "none").attr("stroke", DC.line);
    const a0 = q[2 * iSel], b0 = q[2 * iSel + 1], rr = Math.sqrt(a0 * a0 + b0 * b0) || 1;
    for (let m = 0; m < n; m++) {
      const ang = m * th[iSel], a = a0 * Math.cos(ang) - b0 * Math.sin(ang), b = a0 * Math.sin(ang) + b0 * Math.cos(ang);
      g.append("circle").attr("cx", cx + R * a / rr).attr("cy", cy - R * b / rr).attr("r", 3.2).attr("fill", d3.interpolateViridis(m / (n - 1)));
      if (m === 0 || m === n - 1) TR.note(g, cx + (R + 12) * a / rr, cy - (R + 12) * b / rr + 3, "m=" + m, DC.ink, 9, "middle");
    }
    TR.note(g, 0, cy + R + 22, `θ${iSel} = ${DL.commas(base)}^(−${2 * iSel}/${d}) = ${DL.fmtE(th[iSel], 2)} rad/position`, DC.ink, 9.5);
    TR.note(g, 0, cy + R + 34, `one full turn every ${DL.sig(2 * Math.PI / th[iSel], 4)} positions`, DC.muted, 9.5);
    TR.note(g, 0, cy + R + 46, `pair 0 turns every 2π = 6.28; pair ${pairs - 1} every ${DL.big(2 * Math.PI / th[pairs - 1])}`, DC.muted, 9.5);
    /* middle: two score matrices */
    const sx = 205, cw = Math.min(8, 120 / n);
    const lim = Math.max(TR.maxAbs(SA), TR.maxAbs(SH)), scl = TR.diverging(lim);
    TR.title(g, sx, -8, `score(m, n) — "adjacent" pairing`);
    TR.heat(g, SA, sx, 0, cw, scl);
    TR.note(g, sx, n * cw + 12, "constant along every diagonal", DC.good, 9.5);
    TR.title(g, sx, n * cw + 40, `score(m, n) — "half" pairing`);
    TR.heat(g, SH, sx, n * cw + 48, cw, scl);
    TR.note(g, sx, 2 * n * cw + 60, "also Toeplitz — and a different matrix", DC.a2, 9.5);
    TR.note(g, sx, 2 * n * cw + 72, `max |S_adj − S_half| = ${DL.fmt(Math.max(...SA.flat().map((v, i) => Math.abs(v - SH.flat()[i]))), 2)}`, DC.muted, 9);
    /* right: score against offset, both conventions; residual */
    const rx = 400, rw = f.iw - rx, rh = 170;
    const gr = g.append("g").attr("transform", `translate(${rx},0)`);
    TR.title(gr, 0, -8, "the score as a function of the offset m − n only");
    const offs = d3.range(-(n - 1), n);
    const kA = offs.map(o => DL.dot(DL.rope([q], { base: base, offset: Math.max(o, 0), pairing: "adjacent" })[0], DL.rope([kk], { base: base, offset: Math.max(-o, 0), pairing: "adjacent" })[0]));
    const kH = offs.map(o => DL.dot(DL.rope([q], { base: base, offset: Math.max(o, 0), pairing: "half" })[0], DL.rope([kk], { base: base, offset: Math.max(-o, 0), pairing: "half" })[0]));
    const x = d3.scaleLinear().domain([-(n - 1), n - 1]).range([0, rw]), y = d3.scaleLinear().domain([Math.min(...kA, ...kH) * 1.1, Math.max(...kA, ...kH) * 1.1]).range([rh, 0]);
    DL.gridY(gr, y, rw, 4); DL.axisB(gr, x, rh, 6, "m − n"); DL.axisL(gr, y, 4, "q·R(m−n)·kᵀ");
    DL.curve(gr, offs.map((o, i) => [x(o), y(kA[i])]), { stroke: DC.good, w: 1.8 });
    DL.curve(gr, offs.map((o, i) => [x(o), y(kH[i])]), { stroke: DC.a2, w: 1.6, dash: "4 2" });
    DL.legend(gr, [{ label: "adjacent", color: DC.good }, { label: "half", color: DC.a2 }], 4, 8, { gap: 12, font: 9.5 });
    const ry = rh + 40;
    TR.title(gr, 0, ry, `shift BOTH positions by ${shift}: max change in any score`);
    TR.note(gr, 0, ry + 18, `adjacent: ${DL.fmtE(rA, 1)}`, DC.good, 11);
    TR.note(gr, 0, ry + 34, `half: ${DL.fmtE(rH, 1)}`, DC.a2, 11);
    TR.note(gr, 0, ry + 50, "round-off, not zero: sin and cos of large angles", DC.muted, 9);
    TR.note(gr, 0, ry + 62, "are evaluated in floating point", DC.muted, 9);
    El("ro-readout").innerHTML =
      `dₖ = ${d} (${pairs} pairs), base ${DL.commas(base)}, ${n} positions. Pair ${iSel} advances <b>${DL.fmtE(th[iSel], 2)} rad</b> per position, a full turn every ${DL.sig(2 * Math.PI / th[iSel], 4)} positions. ` +
      `Shifting both the query's and the key's position by ${shift} changes the score by at most <b>${DL.fmtE(rA, 1)}</b> (adjacent pairing) and <b>${DL.fmtE(rH, 1)}</b> (half pairing) — the offset property, at round-off. ` +
      `The two pairings give score matrices that differ by up to ${DL.fmt(Math.max(...SA.flat().map((v, i) => Math.abs(v - SH.flat()[i]))), 2)} on the same q and k: same property, different model.`;
  }
  ["ro-d", "ro-b"].forEach(id => d3.select("#" + id).on("change", draw));
  ["ro-n", "ro-i", "ro-s"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 7 · #rs-svg — RoPE's spectrum against the context.  ═══════════ */
(function () {
  const svg = d3.select("#rs-svg"); if (svg.empty()) return;
  const W = 760, H = 400;
  const El = id => document.getElementById(id);
  function draw() {
    const d = +El("rs-d").value, base = +El("rs-b").value, n = Math.pow(2, +El("rs-n").value);
    const pairs = d / 2, iSel = Math.min(+El("rs-i").value, pairs - 1);
    El("rs-i").max = pairs - 1; El("rs-nv").textContent = DL.commas(n); El("rs-iv").textContent = iSel;
    const th = DL.ropeFreqs(d, base), lam = th.map(w => 2 * Math.PI / w);
    const below = lam.filter(l => l <= n).length;
    const f = DL.frame(svg, W, H, { l: 46, r: 14, t: 22, b: 30 }), g = f.g;
    const uh = 190;
    TR.title(g, 0, -8, `wavelength λᵢ of each pair, dₖ = ${d}, base ${DL.commas(base)} — dashed line: context n = ${DL.commas(n)}`);
    const x = d3.scaleLinear().domain([0, pairs - 1]).range([0, f.iw]), y = d3.scaleLog().domain([4, 2 * Math.PI * base * 2]).range([uh, 0]);
    DL.gridY(g, y, f.iw, 5); DL.axisB(g, x, uh, 8, "pair index i"); DL.axisL(g, y, 5, "λᵢ (positions)", v => Math.abs(Math.log10(v) % 1) < 1e-9 ? DL.big(v) : "");
    g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y(n)).attr("y2", y(n)).attr("stroke", DC.ink).attr("stroke-dasharray", "4 3");
    g.selectAll("circle").data(lam).join("circle").attr("cx", (l, i) => x(i)).attr("cy", l => y(l)).attr("r", 3)
      .attr("fill", (l, i) => i === iSel ? DC.lime : (l <= n ? DC.accent : DC.a2));
    DL.legend(g, [{ label: `${below} pairs periodic inside n (λ ≤ n)`, color: DC.accent }, { label: `${pairs - below} pairs monotone (λ > n): absolute coordinates`, color: DC.a2 }], 6, 12, { gap: 13, font: 9.5 });
    /* lower: cos((offset)·θ) for the selected pair over the context */
    const ly = uh + 44, lh = f.ih - ly;
    const gl = g.append("g").attr("transform", `translate(0,${ly})`);
    const turns = n / lam[iSel];
    TR.title(gl, 0, -8, `pair ${iSel}: cos(Δ·θ${iSel}) over Δ = 0 … ${DL.commas(n)} — ${DL.sig(turns, 3)} turn${turns === 1 ? "" : "s"} completed (λ = ${DL.big(lam[iSel])})`, DC.lime);
    const xs = d3.scaleLinear().domain([0, n]).range([0, f.iw]), ys = d3.scaleLinear().domain([-1, 1]).range([lh, 0]);
    DL.axisB(gl, xs, lh, 6, "offset Δ", v => DL.big(v)); DL.axisL(gl, ys, 3);
    const K = 600, pts = []; for (let t = 0; t <= K; t++) { const D = n * t / K; pts.push([xs(D), ys(Math.cos(D * th[iSel]))]); }
    DL.curve(gl, pts, { stroke: DC.lime, w: 1.5 });
    const slowestTurns = n / lam[pairs - 1], wrapAll = Math.ceil(lam[pairs - 1]);
    El("rs-readout").innerHTML =
      `dₖ = ${d} → ${pairs} pairs, base ${DL.commas(base)}, context ${DL.commas(n)}. <b>${below}</b> pairs complete at least one turn inside the context and <b>${pairs - below}</b> do not. ` +
      `Pair ${iSel} has λ = ${DL.big(lam[iSel])} and completes ${DL.sig(turns, 3)} turns; the slowest pair (λ = ${DL.big(lam[pairs - 1])}) completes <b>${DL.sig(slowestTurns, 3)}</b> of a turn. ` +
      (slowestTurns >= 1 ? `Every pair has wrapped at this context: no monotone coordinate is left, which is where extension methods have to intervene.` : `The smallest context at which every pair has wrapped is <b>${DL.commas(wrapAll)}</b> tokens.`);
  }
  ["rs-d", "rs-b"].forEach(id => d3.select("#" + id).on("change", draw));
  ["rs-n", "rs-i"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 8 · #ab-svg — ALiBi and T5 buckets as score biases.  ══════════ */
(function () {
  const svg = d3.select("#ab-svg"); if (svg.empty()) return;
  const W = 760, H = 420;
  const El = id => document.getElementById(id);
  function draw() {
    const h = +El("ab-h").value, n = +El("ab-n").value, nb = +El("ab-b").value, md = +El("ab-m").value, bidir = El("ab-dir").value === "bi";
    El("ab-nv").textContent = n;
    const slopes = DL.alibiSlopes(h), cm = DL.causalMask(n);
    const f = DL.frame(svg, W, H, { l: 6, r: 6, t: 22, b: 6 }), g = f.g;
    const show = Math.min(h, 8), pw = Math.floor((f.iw - 10) / show), cw = Math.min((pw - 10) / n, 4);
    TR.title(g, 0, -8, `ALiBi bias −m·(i − j) on the causal lower triangle, ${h} heads${h > 8 ? " (first 8 drawn)" : ""}, n = ${n}`);
    const lim = slopes[0] * (n - 1);
    const scl = d3.scaleSequential(d3.interpolateInferno).domain([-lim, 0]);
    for (let j = 0; j < show; j++) {
      const B = DL.addMasks(cm, DL.alibiBias(n, slopes[j]));
      const Bd = B.map(r => r.map(v => isFinite(v) ? v : -lim));
      TR.heat(g, Bd, j * pw, 0, cw, scl);
      TR.note(g, j * pw, n * cw + 12, `m = ${DL.sig(slopes[j], 3)}`, DC.ink, 9);
      TR.note(g, j * pw, n * cw + 23, `−ln 100 at Δ = ${DL.sig(Math.log(100) / slopes[j], 3)}`, DC.muted, 8.5);
    }
    /* lower left: bias against distance */
    const ly = n * cw + 50, lh = f.ih - ly - 10, lw = 330;
    const gl = g.append("g").attr("transform", `translate(40,${ly})`);
    TR.title(gl, 0, -8, "bias against distance Δ, every head");
    const dmax = 64, x = d3.scaleLinear().domain([0, dmax]).range([0, lw]), y = d3.scaleLinear().domain([-slopes[0] * dmax, 0]).range([lh, 0]);
    DL.gridY(gl, y, lw, 4); DL.axisB(gl, x, lh, 6, "Δ = i − j"); DL.axisL(gl, y, 4, "bias");
    slopes.forEach((m, j) => DL.curve(gl, [[x(0), y(0)], [x(dmax), y(-m * dmax)]], { stroke: d3.interpolateViridis(j / Math.max(1, h - 1)), w: 1.4 }));
    gl.append("line").attr("x1", 0).attr("x2", lw).attr("y1", y(-Math.log(100))).attr("y2", y(-Math.log(100))).attr("stroke", DC.bad).attr("stroke-dasharray", "3 3");
    TR.note(gl, lw, y(-Math.log(100)) - 4, "−ln 100: weight cut 100×", DC.bad, 9, "end");
    /* lower right: bucket matrix */
    const bx = 420, bw = Math.min(6, (f.iw - bx) / n);
    const gb = g.append("g").attr("transform", `translate(${bx},${ly})`);
    const BM = DL.relBucketMatrix(n, { bidir: bidir, buckets: nb, maxDist: md });
    const present = new Set(BM.flat()).size;
    TR.title(gb, 0, -8, `bucket(j − i), ${nb} buckets, max ${md}, ${bidir ? "bidirectional" : "causal"}`);
    TR.heat(gb, BM, 0, 0, bw, d3.scaleSequential(d3.interpolateTurbo).domain([0, nb - 1]));
    TR.note(gb, 0, n * bw + 12, `${present} distinct buckets present at n = ${n}`, DC.ink, 9.5);
    TR.note(gb, 0, n * bw + 24, `exact for |Δ| < ${Math.floor((bidir ? nb / 2 : nb) / 2)}, log-spaced to ${md}, then shared`, DC.muted, 9);
    El("ab-readout").innerHTML =
      `ALiBi, h = ${h}: slopes ${slopes.map(m => DL.sig(m, 3)).join(", ")}. Each head's weight on a key at distance Δ is multiplied by e^(−mΔ); the distance at which that factor is 1/100 is ` +
      `<b>${slopes.map(m => DL.sig(Math.log(100) / m, 3)).join(", ")}</b> tokens — ${h} recency windows from ${DL.sig(Math.log(100) / slopes[0], 3)} to ${DL.sig(Math.log(100) / slopes[h - 1], 3)}. ` +
      `Buckets: ${nb} learned scalars per head, ${present} of them in use at n = ${n}; every |Δ| ≥ ${md} shares one value.`;
  }
  ["ab-h", "ab-b", "ab-m", "ab-dir"].forEach(id => d3.select("#" + id).on("change", draw));
  d3.select("#ab-n").on("input", draw);
  draw();
})();

/* ═════════ 9 · #ex-svg — five positional schemes, trained live, tested past
   the training length. Everything goes through TR.exRun → TR.train.  ═════ */
(function () {
  const svg = d3.select("#ex-svg"); if (svg.empty()) return;
  const W = 760, H = 420;
  const El = id => document.getElementById(id);
  const POS = [["sin", "sinusoidal", DC.a2], ["learned", "learned", DC.accent], ["rope", "RoPE", DC.good], ["alibi", "ALiBi", DC.violet], ["none", "none", DC.muted]];
  const LENS = [10, 12, 16, 24, 32, 48, 64];
  function draw() {
    const taskName = El("ex-t").value, steps = +El("ex-s").value, L = +El("ex-L").value, seed = +El("ex-seed").value;
    El("ex-seedv").textContent = seed;
    const runs = POS.map(p => { const R = TR.exRun(p[0], taskName, steps, L, seed); return { pos: p[0], label: p[1], col: p[2], R: R,
      acc: LENS.map(n => TR.evalAtLength(R.m, taskName, n, 24, 90 + n).acc) }; });
    const f = DL.frame(svg, W, H, { l: 44, r: 12, t: 22, b: 30 }), g = f.g;
    const lw = 400, lh = 190;
    TR.title(g, 0, -8, `held-out accuracy against length — trained at n = 10 on "${TR.TASKS[taskName]}", ${L} layer${L > 1 ? "s" : ""}, ${steps} steps`);
    const x = d3.scaleLog().domain([10, 64]).range([0, lw]), y = d3.scaleLinear().domain([0, 1]).range([lh, 0]);
    DL.gridY(g, y, lw, 4); DL.axisB(g, x, lh, 7, "sequence length", v => String(v)); DL.axisL(g, y, 4, "accuracy", v => Math.round(v * 100) + "%");
    g.append("line").attr("x1", x(10)).attr("x2", x(10)).attr("y1", 0).attr("y2", lh).attr("stroke", DC.ink).attr("stroke-dasharray", "4 3");
    TR.note(g, x(10) + 4, 10, "training length", DC.ink, 9);
    const chance = 1 / 8;
    g.append("line").attr("x1", 0).attr("x2", lw).attr("y1", y(chance)).attr("y2", y(chance)).attr("stroke", DC.bad).attr("stroke-dasharray", "3 3");
    TR.note(g, lw, y(chance) - 4, "chance 1/8", DC.bad, 9, "end");
    runs.forEach(r => {
      const pts = LENS.map((n, i) => [n, r.acc[i]]).filter(p => isFinite(p[1]));
      DL.curve(g, pts.map(p => [x(p[0]), y(p[1])]), { stroke: r.col, w: r.pos === "learned" ? 2.4 : 1.8 });
      g.selectAll(null).data(pts).join("circle").attr("cx", p => x(p[0])).attr("cy", p => y(p[1])).attr("r", 3).attr("fill", r.col);
      if (r.pos === "learned") TR.note(g, x(10) + 6, y(pts[0][1]) + 12, "← learned: no row past 10", DC.accent, 9);
    });
    DL.legend(g, runs.map(r => ({ label: r.label, color: r.col })), lw - 80, lh - 70, { gap: 12, font: 9.5 });
    /* right: loss curves */
    const rx = lw + 40, rw = f.iw - rx;
    const gr = g.append("g").attr("transform", `translate(${rx},0)`);
    TR.title(gr, 0, -8, "training loss");
    const allv = runs.flatMap(r => r.R.curve.map(c => c[1]));
    const xl = d3.scaleLinear().domain([0, steps]).range([0, rw]), yl = d3.scaleLog().domain([Math.max(1e-3, Math.min(...allv) / 2), Math.max(...allv) * 1.5]).range([lh, 0]).clamp(true);
    DL.axisB(gr, xl, lh, 3, "step"); DL.axisL(gr, yl, 3, "", v => Math.abs(Math.log10(v) % 1) < 1e-9 ? DL.fmtE(v, 0) : "");
    runs.forEach(r => DL.curve(gr, r.R.curve.map(c => [xl(c[0]), yl(Math.max(c[1], 1e-3))]), { stroke: r.col, w: 1.4 }));
    /* table */
    const ty = lh + 46, cwid = 60;
    TR.note(g, 0, ty, "scheme", DC.muted, 9.5);
    LENS.forEach((n, i) => TR.note(g, 120 + i * cwid, ty, "n = " + n, DC.muted, 9.5));
    runs.forEach((r, k) => {
      const yy = ty + 15 + k * 14;
      TR.note(g, 0, yy, r.label, r.col, 10);
      LENS.forEach((n, i) => TR.note(g, 120 + i * cwid, yy, isFinite(r.acc[i]) ? Math.round(100 * r.acc[i]) + "%" : "— (no row)", isFinite(r.acc[i]) ? DC.ink : DC.bad, 10));
    });
    const at10 = runs.map(r => r.acc[0]), at64 = runs.map(r => r.acc[LENS.length - 1]);
    const best10 = runs[at10.indexOf(Math.max(...at10))], fin = runs.filter(r => isFinite(r.acc[LENS.length - 1]));
    const best64 = fin.reduce((a, b) => b.acc[LENS.length - 1] > a.acc[LENS.length - 1] ? b : a, fin[0]);
    El("ex-readout").innerHTML =
      `Task "${TR.TASKS[taskName]}", ${steps} steps, ${L} layer${L > 1 ? "s" : ""}, seed ${seed}. At the training length the best scheme is <b>${best10.label}</b> (${Math.round(100 * best10.acc[0])}%); at n = 64 it is <b>${best64.label}</b> (${Math.round(100 * best64.acc[LENS.length - 1])}%). ` +
      `Drop from n = 10 to n = 64: ` + fin.map(r => `${r.label} ${Math.round(100 * r.acc[0])}% → ${Math.round(100 * r.acc[LENS.length - 1])}%`).join(", ") +
      `; the learned table cannot be evaluated past 10 at all. ` +
      (taskName === "shift2" ? `On "two back", ALiBi sits at ${Math.round(100 * runs[3].acc[0])}% at the training length against ${Math.round(100 * runs[4].acc[0])}% with no positional encoding and ${Math.round(100 * runs[2].acc[0])}% for RoPE — a monotone recency bias cannot land on an exact offset. ` : `Switch the task to "two back" to see ALiBi fall to the no-position level in-distribution. `) +
      `Chance is 12.5%.`;
  }
  ["ex-t", "ex-s", "ex-L"].forEach(id => d3.select("#" + id).on("change", draw));
  d3.select("#ex-seed").on("input", draw);
  draw();
})();

/* ═════════ 10 · #np-svg — the causal mask as a positional signal.  ═══════ */
(function () {
  const svg = d3.select("#np-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const El = id => document.getElementById(id);
  function draw() {
    const n = +El("np-n").value, mode = El("np-w").value, seed = +El("np-seed").value, bi = El("np-bi").checked;
    El("np-nv").textContent = n; El("np-seedv").textContent = seed;
    const dk = 8, dv = 8;
    const X = TR.gauss(n, 16, seed, 1);
    const Wq = mode === "uniform" ? DL.zeros2(16, dk) : TR.gauss(16, dk, seed + 10, 1 / 4), Wk = TR.gauss(16, dk, seed + 20, 1 / 4);
    /* values: the BOS row owns coordinate 0, every other row lives in coordinates 1 … dv−1 */
    const V = DL.zeros2(n, dv);
    V[0][0] = 1;
    const rv = DL.rng(seed + 30);
    for (let i = 1; i < n; i++) for (let c = 1; c < dv; c++) V[i][c] = DL.randn(rv);
    const Q = DL.matmul(X, Wq), K = DL.matmul(X, Wk);
    const out = DL.sdpa(Q, K, V, { mask: bi ? null : DL.causalMask(n) });
    const comp = out.Z.map(r => r[0]), closed = comp.map((_, i) => bi ? 1 / n : 1 / (i + 1));
    let dev = 0; comp.forEach((c, i) => dev = Math.max(dev, Math.abs(c - closed[i])));
    const f = DL.frame(svg, W, H, { l: 44, r: 12, t: 22, b: 30 }), g = f.g;
    const lw = 380, lh = f.ih;
    TR.title(g, 0, -8, `BOS component of zᵢ against position — ${mode === "uniform" ? "uniform attention" : "random untrained head"}, ${bi ? "bidirectional" : "causal"}`);
    const x = d3.scaleLinear().domain([0, n - 1]).range([0, lw]), y = d3.scaleLinear().domain([0, 1.05]).range([lh, 0]);
    DL.gridY(g, y, lw, 4); DL.axisB(g, x, lh, 6, "position i"); DL.axisL(g, y, 4, "weight on BOS's value");
    DL.curve(g, closed.map((c, i) => [x(i), y(c)]), { stroke: DC.ink, w: 1.2, dash: "4 3" });
    g.selectAll(null).data(comp).join("circle").attr("cx", (c, i) => x(i)).attr("cy", c => y(c)).attr("r", 3).attr("fill", DC.accent);
    TR.note(g, lw, 12, bi ? "closed form 1/n = " + DL.fmt(1 / n, 4) : "closed form 1/(i+1)", DC.ink, 9.5, "end");
    TR.note(g, lw, 24, `max deviation ${DL.fmtE(dev, 1)}`, mode === "uniform" ? DC.good : DC.a2, 9.5, "end");
    /* right: the trained NoPE model's accuracy by position */
    const rx = lw + 44, rw = f.iw - rx;
    const gr = g.append("g").attr("transform", `translate(${rx},0)`);
    const R = TR.exRun("none", "shift1", 300, 2, 1);
    const a10 = TR.accByPosition(R.m, "shift1", 10, 40, 5), a32 = TR.accByPosition(R.m, "shift1", 32, 40, 6);
    TR.title(gr, 0, -8, "§09's no-position model, accuracy by position");
    const xb = d3.scaleBand().domain(d3.range(32)).range([0, rw]).paddingInner(0.15), yb = d3.scaleLinear().domain([0, 1]).range([lh, 0]);
    DL.axisL(gr, yb, 4, "", v => Math.round(v * 100) + "%");
    a32.forEach((a, i) => { if (!isFinite(a)) return; gr.append("rect").attr("x", xb(i)).attr("y", yb(a)).attr("width", xb.bandwidth()).attr("height", lh - yb(a)).attr("fill", DC.muted).attr("fill-opacity", 0.55); });
    a10.forEach((a, i) => { if (!isFinite(a)) return; gr.append("rect").attr("x", xb(i)).attr("y", yb(a)).attr("width", xb.bandwidth()).attr("height", lh - yb(a)).attr("fill", DC.accent).attr("fill-opacity", 0.85); });
    gr.append("line").attr("x1", 0).attr("x2", rw).attr("y1", yb(1 / 8)).attr("y2", yb(1 / 8)).attr("stroke", DC.bad).attr("stroke-dasharray", "3 3");
    TR.note(gr, 0, lh + 14, "position 0 … 31 (n = 32, grey) · 0 … 9 (n = 10, blue) · red: chance", DC.muted, 9);
    const m10 = DL.mean(a10.filter(isFinite)), m32 = DL.mean(a32.filter(isFinite));
    DL.legend(gr, [{ label: `n = 10: mean ${Math.round(100 * m10)}%`, color: DC.accent }, { label: `n = 32: mean ${Math.round(100 * m32)}%`, color: DC.muted }], rw - 110, 10, { gap: 12, font: 9.5 });
    El("np-readout").innerHTML =
      `${mode === "uniform" ? "Uniform" : "Random-head"} ${bi ? "bidirectional" : "causal"} attention: the BOS value's weight in row i deviates from the closed form ${bi ? "1/n" : "1/(i+1)"} by at most <b>${DL.fmtE(dev, 1)}</b>. ` +
      (bi ? `Every position sees all ${n} keys, so the running mean is the same everywhere and nothing about position leaks.` : `Position 0 reads 1.000, position ${n - 1} reads ${DL.fmt(1 / n, 4)}: a monotone coordinate, derived from the mask alone.`) +
      ` The trained two-layer model with no positional encoding (task "previous token", 300 steps) reaches a mean accuracy of <b>${Math.round(100 * m10)}%</b> at its training length and <b>${Math.round(100 * m32)}%</b> at n = 32, against 12.5% chance.`;
  }
  ["np-w", "np-bi"].forEach(id => d3.select("#" + id).on("change", draw));
  ["np-n", "np-seed"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 11 · #rn-svg — the residual stream through L blocks.  ═════════ */
(function () {
  const svg = d3.select("#rn-svg"); if (svg.empty()) return;
  const W = 760, H = 400, n = 6, h = 4;
  const El = id => document.getElementById(id);
  function stackRun(L, d, pre, sc, seed) {
    const dff = 4 * d, blocks = [], r = DL.rng(seed * 11 + 3);
    for (let l = 0; l < L; l++) {
      const B = DL.blockInit(d, h, dff, { pre: pre, seed: seed * 100 + l });
      [B.attn.Wq, B.attn.Wk, B.attn.Wv, B.attn.Wo, B.ffn.W1, B.ffn.W2].forEach(M => TR.scaleMat(M, sc));
      blocks.push(B);
    }
    let X = DL.zeros2(n, d); for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) X[i][j] = DL.randn(r);
    const X0 = X.map(r => r.slice()), rms = [], cos = [], dev = [];
    const rmsOf = M => Math.sqrt(M.flat().reduce((s, v) => s + v * v, 0) / M.length);
    const cosOf = (A, B) => { let ab = 0, aa = 0, bb = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) { ab += A[i][j] * B[i][j]; aa += A[i][j] * A[i][j]; bb += B[i][j] * B[i][j]; } return ab / Math.sqrt(aa * bb); };
    rms.push(rmsOf(X)); cos.push(1);
    const mask = DL.causalMask(n);
    for (let l = 0; l < L; l++) {
      const o = DL.blockForward(X, blocks[l], { mask: mask });
      const G = DL.zeros2(n, d); for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) G[i][j] = DL.randn(r);
      const g = DL.blockBackward(G, o.st, blocks[l]);
      let num = 0, den = 0; for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) { num += (g.X[i][j] - G[i][j]) ** 2; den += G[i][j] ** 2; }
      dev.push(Math.sqrt(num / den));
      X = o.Y; rms.push(rmsOf(X)); cos.push(cosOf(X, X0));
    }
    return { rms: rms, cos: cos, dev: dev };
  }
  function draw() {
    const L = +El("rn-L").value, d = +El("rn-d").value, sc = +El("rn-s").value, seed = +El("rn-seed").value;
    El("rn-Lv").textContent = L; El("rn-sv").textContent = sc.toFixed(1); El("rn-seedv").textContent = seed;
    const P = stackRun(L, d, true, sc, seed), Q = stackRun(L, d, false, sc, seed);
    const f = DL.frame(svg, W, H, { l: 40, r: 10, t: 24, b: 34 }), g = f.g;
    const pw = (f.iw - 60) / 3, ph = f.ih - 10;
    const panel = (k, ttl) => { const gg = g.append("g").attr("transform", `translate(${k * (pw + 30)},0)`); TR.title(gg, 0, -8, ttl); return gg; };
    /* 1: rms */
    const g1 = panel(0, "rms of one token's stream");
    const x = d3.scaleLinear().domain([0, L]).range([0, pw]);
    const y1 = d3.scaleLinear().domain([0, Math.max(...P.rms, ...Q.rms) * 1.1]).range([ph, 0]);
    DL.gridY(g1, y1, pw, 4); DL.axisB(g1, x, ph, 5, "layer"); DL.axisL(g1, y1, 4);
    DL.curve(g1, P.rms.map((v, i) => [x(i), y1(v)]), { stroke: TR.CO.pre, w: 2 });
    DL.curve(g1, Q.rms.map((v, i) => [x(i), y1(v)]), { stroke: TR.CO.post, w: 2 });
    /* fit rms ~ a·L^b on the pre curve, l ≥ 1 */
    const lx = [], ly = []; for (let l = 1; l <= L; l++) { lx.push(Math.log(l)); ly.push(Math.log(P.rms[l])); }
    const mx = DL.mean(lx), my = DL.mean(ly); let sxy = 0, sxx = 0; lx.forEach((v, i) => { sxy += (v - mx) * (ly[i] - my); sxx += (v - mx) ** 2; });
    const bExp = sxy / sxx;
    g1.append("line").attr("x1", 0).attr("x2", pw).attr("y1", y1(Math.sqrt(d))).attr("y2", y1(Math.sqrt(d))).attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    TR.note(g1, pw, y1(Math.sqrt(d)) - 4, "√d", DC.muted, 9, "end");
    DL.legend(g1, [{ label: "pre-norm", color: TR.CO.pre }, { label: "post-norm", color: TR.CO.post }], 4, 10, { gap: 12, font: 9.5 });
    TR.note(g1, 4, 40, `pre-norm growth ∝ L^${DL.fmt(bExp, 2)}`, TR.CO.pre, 9.5);
    /* 2: cosine with input */
    const g2 = panel(1, "cosine(stream, input x⁰)");
    const y2 = d3.scaleLinear().domain([0, 1]).range([ph, 0]);
    DL.gridY(g2, y2, pw, 4); DL.axisB(g2, x, ph, 5, "layer"); DL.axisL(g2, y2, 4);
    DL.curve(g2, P.cos.map((v, i) => [x(i), y2(v)]), { stroke: TR.CO.pre, w: 2 });
    DL.curve(g2, Q.cos.map((v, i) => [x(i), y2(v)]), { stroke: TR.CO.post, w: 2 });
    /* 3: deviation from identity */
    const g3 = panel(2, "‖(∂Y/∂X − I)ᵀg‖ / ‖g‖ per block");
    const y3 = d3.scaleLog().domain([Math.min(0.05, ...P.dev, ...Q.dev) / 1.5, Math.max(1.5, ...P.dev, ...Q.dev) * 1.5]).range([ph, 0]);
    DL.gridY(g3, y3, pw, 4); DL.axisB(g3, x, ph, 5, "block"); DL.axisL(g3, y3, 4, "", v => Math.abs(Math.log10(v) % 1) < 1e-9 || Math.abs(Math.log10(v / 3) % 1) < 1e-9 ? DL.sig(v, 2) : "");
    g3.append("line").attr("x1", 0).attr("x2", pw).attr("y1", y3(1)).attr("y2", y3(1)).attr("stroke", DC.bad).attr("stroke-dasharray", "3 3");
    TR.note(g3, pw, y3(1) - 4, "1 = as large as the identity", DC.bad, 9, "end");
    g3.selectAll(null).data(P.dev).join("circle").attr("cx", (v, i) => x(i + 1)).attr("cy", v => y3(v)).attr("r", 3).attr("fill", TR.CO.pre);
    g3.selectAll(null).data(Q.dev).join("circle").attr("cx", (v, i) => x(i + 1)).attr("cy", v => y3(v)).attr("r", 3).attr("fill", TR.CO.post);
    El("rn-readout").innerHTML =
      `L = ${L}, d = ${d}, weights × ${sc.toFixed(1)}, seed ${seed}. Stream rms after L blocks: pre-norm <b>${DL.fmt(P.rms[L], 2)}</b> (from ${DL.fmt(P.rms[0], 2)}; growth fits L^${DL.fmt(bExp, 2)} — √L is 0.50), post-norm <b>${DL.fmt(Q.rms[L], 2)}</b> (√d = ${Math.sqrt(d).toFixed(2)}). ` +
      `Cosine with the input after L blocks: pre-norm ${DL.fmt(P.cos[L], 3)}, post-norm ${DL.fmt(Q.cos[L], 3)}. ` +
      `Mean relative deviation of a block's Jacobian from the identity: pre-norm <b>${DL.fmt(DL.mean(P.dev), 2)}</b>, post-norm <b>${DL.fmt(DL.mean(Q.dev), 2)}</b>` +
      (DL.mean(P.dev) < 1 ? " — the pre-norm block is near-identity, the post-norm block is not." : " — at this weight scale even the pre-norm block has left the near-identity regime.");
  }
  d3.select("#rn-d").on("change", draw);
  ["rn-L", "rn-s", "rn-seed"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 12 · #pp-svg — pre- against post-norm: the gradient profile at
   init, and four short live trainings at two peak learning rates.  ═══════ */
(function () {
  const svg = d3.select("#pp-svg"); if (svg.empty()) return;
  const W = 760, H = 400, n = 8, h = 4;
  const El = id => document.getElementById(id);
  function profile(L, d, pre, seed) {
    const dff = 4 * d, blocks = [], r = DL.rng(seed * 17 + 1);
    for (let l = 0; l < L; l++) blocks.push(DL.blockInit(d, h, dff, { pre: pre, seed: seed * 100 + l }));
    let X = DL.zeros2(n, d); for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) X[i][j] = DL.randn(r);
    const sts = [], mask = DL.causalMask(n);
    for (let l = 0; l < L; l++) { const o = DL.blockForward(X, blocks[l], { mask: mask }); sts.push(o.st); X = o.Y; }
    let dY = DL.zeros2(n, d); for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) dY[i][j] = DL.randn(r) / Math.sqrt(d);
    if (pre) { const ones = DL.zeros(d).map(() => 1), nf = DL.lnForward(X, ones, DL.zeros(d)); dY = DL.lnBackward(dY, nf.st).X; }
    const gW = new Array(L);
    for (let l = L - 1; l >= 0; l--) { const g = DL.blockBackward(dY, sts[l], blocks[l]); gW[l] = TR.frobOf(g.ffn.W2); dY = g.X; }
    return gW;
  }
  function draw() {
    const L = +El("pp-L").value, d = +El("pp-d").value, seed = +El("pp-seed").value, lrs = El("pp-lr").value.split(",").map(Number);
    El("pp-Lv").textContent = L; El("pp-seedv").textContent = seed;
    const P = profile(L, d, true, seed), Q = profile(L, d, false, seed);
    const f = DL.frame(svg, W, H, { l: 44, r: 10, t: 24, b: 34 }), g = f.g;
    const lw = 340, ph = f.ih;
    TR.title(g, 0, -8, `‖∂L/∂W₂‖ per block at init, L = ${L}, d = ${d}`);
    const x = d3.scaleLinear().domain([0, L - 1]).range([0, lw]), y = d3.scaleLog().domain([Math.min(...P, ...Q) / 1.5, Math.max(...P, ...Q) * 1.5]).range([ph, 0]);
    DL.gridY(g, y, lw, 4); DL.axisB(g, x, ph, 6, "block (0 = bottom)"); DL.axisL(g, y, 4, "", v => DL.sig(v, 2));
    DL.curve(g, P.map((v, i) => [x(i), y(v)]), { stroke: TR.CO.pre, w: 2 });
    DL.curve(g, Q.map((v, i) => [x(i), y(v)]), { stroke: TR.CO.post, w: 2 });
    g.selectAll(null).data(P).join("circle").attr("cx", (v, i) => x(i)).attr("cy", v => y(v)).attr("r", 2.6).attr("fill", TR.CO.pre);
    g.selectAll(null).data(Q).join("circle").attr("cx", (v, i) => x(i)).attr("cy", v => y(v)).attr("r", 2.6).attr("fill", TR.CO.post);
    DL.legend(g, [{ label: "pre-norm (+ final norm)", color: TR.CO.pre }, { label: "post-norm", color: TR.CO.post }], 4, 10, { gap: 12, font: 9.5 });
    /* top/bottom ratio across depths for pre-norm, and its power law */
    const depths = [4, 8, 16, 32], ratios = depths.map(LL => { const p = profile(LL, d, true, seed); return p[LL - 1] / p[0]; });
    const ratiosQ = depths.map(LL => { const p = profile(LL, d, false, seed); return p[LL - 1] / p[0]; });
    const lx = depths.map(Math.log), ly = ratios.map(Math.log), mx = DL.mean(lx), my = DL.mean(ly);
    let sxy = 0, sxx = 0; lx.forEach((v, i) => { sxy += (v - mx) * (ly[i] - my); sxx += (v - mx) ** 2; });
    const powr = sxy / sxx;
    TR.note(g, 4, 40, `pre-norm top/bottom ratio: ${depths.map((LL, i) => "L=" + LL + " → " + DL.fmt(ratios[i], 2)).join(", ")}`, TR.CO.pre, 9);
    TR.note(g, 4, 52, `fits L^${DL.fmt(powr, 2)} (1/√L would be −0.50)`, TR.CO.pre, 9);
    TR.note(g, 4, 64, `post-norm top/bottom ratio: ${depths.map((LL, i) => "L=" + LL + " → " + DL.fmt(ratiosQ[i], 2)).join(", ")}`, TR.CO.post, 9);
    /* right: live trainings */
    const rx = lw + 44, rw = f.iw - rx;
    const gr = g.append("g").attr("transform", `translate(${rx},0)`);
    TR.title(gr, 0, -8, "copy task, live: pre/post × two peak rates");
    const runs = [];
    lrs.forEach((lr, k) => [true, false].forEach(pre => runs.push({ lr: lr, pre: pre, R: TR.train({ task: "copy", steps: 150, B: 6, lr: lr, warm: 30, seed: 3, cfg: { pre: pre } }) })));
    const allv = runs.flatMap(r => r.R.curve.map(c => c[1]));
    const xs = d3.scaleLinear().domain([0, 150]).range([0, rw]), ys = d3.scaleLog().domain([Math.max(1e-3, Math.min(...allv) / 2), Math.max(...allv) * 1.5]).range([ph, 0]).clamp(true);
    DL.gridY(gr, ys, rw, 4); DL.axisB(gr, xs, ph, 4, "step"); DL.axisL(gr, ys, 4, "", v => Math.abs(Math.log10(v) % 1) < 1e-9 ? DL.fmtE(v, 0) : "");
    runs.forEach(r => DL.curve(gr, r.R.curve.map(c => [xs(c[0]), ys(Math.max(c[1], 1e-3))]), { stroke: r.pre ? TR.CO.pre : TR.CO.post, w: 1.8, dash: r.lr === lrs[0] ? "4 3" : null }));
    DL.legend(gr, [{ label: `pre-norm, lr ${lrs[0]}`, color: TR.CO.pre, dash: "4 3" }, { label: `post-norm, lr ${lrs[0]}`, color: TR.CO.post, dash: "4 3" },
                   { label: `pre-norm, lr ${lrs[1]}`, color: TR.CO.pre }, { label: `post-norm, lr ${lrs[1]}`, color: TR.CO.post }], 4, 10, { gap: 12, font: 9.5 });
    El("pp-readout").innerHTML =
      `At init, L = ${L}, d = ${d}: top-block / bottom-block gradient ratio is <b>${DL.fmt(P[L - 1] / P[0], 2)}</b> for pre-norm and <b>${DL.fmt(Q[L - 1] / Q[0], 2)}</b> for post-norm; across depths 4–32 the pre-norm ratio falls as L^${DL.fmt(powr, 2)} while the post-norm ratio stays at ${ratiosQ.map(v => DL.fmt(v, 2)).join(" / ")}. ` +
      `Live copy-task runs (150 steps, 30-step warmup): final loss ` + runs.map(r => `${r.pre ? "pre" : "post"}@${r.lr} <b>${DL.fmtE(r.R.trainLoss, 1)}</b>`).join(", ") +
      `. ` + (runs[3].R.trainLoss > 1.5 * runs[2].R.trainLoss && runs[3].R.trainLoss > 1.5 ? `At the higher rate post-norm stalls near chance where pre-norm learns.` : `At this scale the two wirings are close; the ceiling separates them at larger d and L.`);
  }
  ["pp-d", "pp-lr"].forEach(id => d3.select("#" + id).on("change", draw));
  ["pp-L", "pp-seed"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 13 · #ln-svg — LayerNorm against RMSNorm on real rows.  ═══════ */
(function () {
  const svg = d3.select("#ln-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const El = id => document.getElementById(id);
  function draw() {
    const mu = +El("ln-mu").value, d = +El("ln-d").value, gam = +El("ln-g").value, seed = +El("ln-seed").value;
    El("ln-muv").textContent = mu.toFixed(2); El("ln-gv").textContent = gam.toFixed(2); El("ln-seedv").textContent = seed;
    const base = TR.gauss(1, d, seed, 1)[0], m0 = DL.mean(base), x = base.map(v => v - m0 + mu);
    const gv = DL.zeros(d).map(() => gam), bv = DL.zeros(d);
    const ln = DL.layerNorm([x], gv, bv)[0], rm = DL.rmsNorm([x], gv)[0], diff = ln.map((v, j) => v - rm[j]);
    const gap = TR.rowNorm(diff);
    const gapAt = m => { const xx = base.map(v => v - m0 + m); const a = DL.layerNorm([xx], gv, bv)[0], b = DL.rmsNorm([xx], gv)[0]; return TR.rowNorm(a.map((v, j) => v - b[j])); };
    const f = DL.frame(svg, W, H, { l: 44, r: 12, t: 22, b: 30 }), g = f.g;
    const cw = Math.min(14, (f.iw - 160) / d), lim = Math.max(...x.map(Math.abs), ...ln.map(Math.abs), ...rm.map(Math.abs));
    const scl = TR.diverging(lim);
    const rows = [["x (row)", x], ["LayerNorm(x)", ln], ["RMSNorm(x)", rm], ["difference", diff]];
    rows.forEach((r, i) => { const yy = i * 22; TR.note(g, 0, yy + 11, r[0], DC.muted, 9.5); TR.strip(g, r[1], 90, yy, cw, scl, { h: 16 }); TR.note(g, 90 + cw * d + 8, yy + 11, `mean ${DL.fmt(DL.mean(r[1]), 3)} · ‖·‖ ${DL.fmt(TR.rowNorm(r[1]), 2)}`, DC.ink, 9); });
    TR.title(g, 0, -8, `one row, d = ${d}, μ = ${mu.toFixed(2)}, σ = ${DL.fmt(DL.std(x), 3)}, γ = ${gam.toFixed(2)}, β = 0`);
    /* lower: gap against mean */
    const ly = 110, lh = f.ih - ly, lw = f.iw;
    const gl = g.append("g").attr("transform", `translate(0,${ly})`);
    TR.title(gl, 0, -8, "‖LayerNorm(x) − RMSNorm(x)‖₂ against the row mean μ (σ held fixed)");
    const mus = DL.linspace(-3, 3, 121), gaps = mus.map(gapAt);
    const xs = d3.scaleLinear().domain([-3, 3]).range([0, lw]), ys = d3.scaleLinear().domain([0, Math.max(...gaps) * 1.1]).range([lh, 0]);
    DL.gridY(gl, ys, lw, 4); DL.axisB(gl, xs, lh, 7, "row mean μ"); DL.axisL(gl, ys, 4, "gap");
    DL.curve(gl, mus.map((m, i) => [xs(m), ys(gaps[i])]), { stroke: DC.violet, w: 2 });
    gl.append("circle").attr("cx", xs(mu)).attr("cy", ys(gap)).attr("r", 4).attr("fill", DC.lime);
    TR.note(gl, xs(mu) + 6, ys(gap) - 6, `gap ${DL.fmt(gap, 3)} at μ = ${mu.toFixed(2)}`, DC.lime, 9.5);
    const g0 = gapAt(0);
    TR.note(gl, xs(0) + 6, lh - 6, `at μ = 0: ${DL.fmtE(g0, 1)}`, DC.good, 9.5);
    El("ln-readout").innerHTML =
      `Row: mean ${DL.fmt(DL.mean(x), 3)}, std ${DL.fmt(DL.std(x), 3)}, d = ${d}. Gap between the two outputs: <b>${DL.fmt(gap, 3)}</b> now; <b>${DL.fmtE(g0, 1)}</b> at μ = 0 (round-off — the identity holds). ` +
      `Parameters at this d: LayerNorm <b>${2 * d}</b> (γ, β), RMSNorm <b>${d}</b> (γ). Reductions per row: LayerNorm two (mean, then variance), RMSNorm one (mean square). ` +
      `The RMSNorm output keeps the row's mean direction (its mean is ${DL.fmt(DL.mean(rm), 3)}); the LayerNorm output is exactly zero-mean.`;
  }
  d3.select("#ln-d").on("change", draw);
  ["ln-mu", "ln-g", "ln-seed"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 14 · #ff-svg — the FFN's share against the width ratio.  ══════ */
(function () {
  const svg = d3.select("#ff-svg"); if (svg.empty()) return;
  const W = 760, H = 360;
  const El = id => document.getElementById(id);
  function draw() {
    const d = +El("ff-d").value, h = +El("ff-h").value, gs = El("ff-g").value, gated = El("ff-k").value === "g", ratio = +El("ff-r").value;
    El("ff-rv").textContent = ratio;
    const g = gs === "h" ? h : +gs, dk = d / h;
    const share = (r, gg) => { const a = DL.mhaParams(d, dk, dk, h, gg, false).total, f = DL.ffnParams(d, Math.round(r * d), { gated: gated, bias: !gated }); return f / (a + f); };
    const f = DL.frame(svg, W, H, { l: 44, r: 12, t: 22, b: 34 }), G = f.g;
    const lw = 360, lh = f.ih;
    TR.title(G, 0, -8, `FFN share of a block's parameters against d_ff/d, d = ${d}, h = ${h}`);
    const rs = DL.linspace(1, 8, 57);
    const x = d3.scaleLinear().domain([1, 8]).range([0, lw]), y = d3.scaleLinear().domain([0, 1]).range([lh, 0]);
    DL.gridY(G, y, lw, 4); DL.axisB(G, x, lh, 7, "d_ff / d"); DL.axisL(G, y, 4, "share", v => Math.round(100 * v) + "%");
    DL.curve(G, rs.map(r => [x(r), y(share(r, h))]), { stroke: DC.violet, w: 2 });
    if (g !== h) DL.curve(G, rs.map(r => [x(r), y(share(r, g))]), { stroke: DC.teal, w: 2, dash: "5 3" });
    G.append("line").attr("x1", x(ratio)).attr("x2", x(ratio)).attr("y1", 0).attr("y2", lh).attr("stroke", DC.ink).attr("stroke-dasharray", "3 3");
    G.append("line").attr("x1", 0).attr("x2", lw).attr("y1", y(2 / 3)).attr("y2", y(2 / 3)).attr("stroke", DC.muted).attr("stroke-dasharray", "2 3");
    TR.note(G, 2, y(2 / 3) - 4, "2/3", DC.muted, 9);
    const sh = share(ratio, g);
    TR.note(G, x(ratio) + 5, 12, `${TR.pct(sh)} at ratio ${ratio}`, DC.ink, 10);
    DL.legend(G, [{ label: "MHA (g = h)", color: DC.violet }].concat(g !== h ? [{ label: `g = ${g}`, color: DC.teal, dash: "5 3" }] : []), 4, lh - 30, { gap: 12, font: 9.5 });
    /* right: stacked bar of parameters at the chosen ratio */
    const rx = lw + 40, rw = f.iw - rx, P = DL.mhaParams(d, dk, dk, h, g, false), dff = Math.round(ratio * d);
    const parts = gated
      ? [["W_Q", P.Wq, DC.a2], ["W_K", P.Wk, DC.a2], ["W_V", P.Wv, DC.a2], ["W_O", P.Wo, DC.a2], ["W_g", d * dff, DC.violet], ["W_u", d * dff, DC.violet], ["W_d", d * dff, DC.violet]]
      : [["W_Q", P.Wq, DC.a2], ["W_K", P.Wk, DC.a2], ["W_V", P.Wv, DC.a2], ["W_O", P.Wo, DC.a2], ["W₁ (+b)", d * dff + dff, DC.violet], ["W₂ (+b)", d * dff + d, DC.violet]];
    const tot = parts.reduce((s, p) => s + p[1], 0);
    const gr = G.append("g").attr("transform", `translate(${rx},0)`);
    TR.title(gr, 0, -8, `one block at d_ff = ${DL.commas(dff)}: ${DL.big(tot)} parameters`);
    let yy = 0; const bh = lh - 40;
    parts.forEach(p => {
      const hh = bh * p[1] / tot;
      gr.append("rect").attr("x", 0).attr("y", yy).attr("width", 70).attr("height", Math.max(hh - 1, 0.5)).attr("fill", p[2]).attr("fill-opacity", 0.85);
      TR.note(gr, 78, yy + Math.min(hh / 2 + 4, hh + 3), `${p[0]}  ${DL.big(p[1])}  (${TR.pct(p[1] / tot)})`, DC.ink, 9);
      yy += hh;
    });
    TR.note(gr, 0, bh + 16, "MACs per token split identically:", DC.muted, 9.5);
    TR.note(gr, 0, bh + 28, "one MAC per parameter per token, for every matrix here", DC.muted, 9.5);
    El("ff-readout").innerHTML =
      `d = ${d}, h = ${h}, g = ${g}, d_ff = ${DL.commas(dff)} (ratio ${ratio}), ${gated ? "gated" : "ungated"}. Attention projections <b>${DL.commas(P.total)}</b>, FFN <b>${DL.commas(DL.ffnParams(d, dff, { gated: gated, bias: !gated }))}</b> → FFN share <b>${TR.pct(sh)}</b>` +
      (g !== h ? ` (it would be ${TR.pct(share(ratio, h))} with g = h: grouped-query attention shrank W_K and W_V by ${h / g}× and moved the share).` : `.`) +
      ` Per-token weight MACs of the block: ${DL.big(P.total + (gated ? 3 : 2) * d * dff)}, in the same proportions.`;
  }
  ["ff-d", "ff-h", "ff-g", "ff-k"].forEach(id => d3.select("#" + id).on("change", draw));
  d3.select("#ff-r").on("input", draw);
  draw();
})();

/* ═════════ 15 · #gl-svg — activations, the gated surface, the width rule. ═ */
(function () {
  const svg = d3.select("#gl-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const El = id => document.getElementById(id);
  function draw() {
    const a = El("gl-a").value, d = +El("gl-d").value, m = +El("gl-m").value, mult = +El("gl-mu").value;
    const f = DL.frame(svg, W, H, { l: 40, r: 10, t: 22, b: 34 }), g = f.g;
    /* left: activations */
    const lw = 230, lh = f.ih;
    TR.title(g, 0, -8, "ReLU, GELU (exact), SiLU and their derivatives");
    const xs = DL.linspace(-4, 4, 161);
    const x = d3.scaleLinear().domain([-4, 4]).range([0, lw]), y = d3.scaleLinear().domain([-1, 4]).range([lh, 0]);
    DL.gridY(g, y, lw, 5); DL.axisB(g, x, lh, 4, "z"); DL.axisL(g, y, 5);
    const acts = [["relu", DC.muted], ["gelu", DC.a2], ["silu", DC.good]];
    acts.forEach(([k, c]) => {
      DL.curve(g, xs.map(v => [x(v), y(DL.act(k).f(v))]), { stroke: c, w: 2 });
      DL.curve(g, xs.map(v => [x(v), y(DL.act(k).df(v))]), { stroke: c, w: 1, dash: "3 2", op: 0.8 });
    });
    const minOf = k => { let best = [0, 0]; xs.forEach(v => { const fv = DL.act(k).f(v); if (fv < best[1]) best = [v, fv]; }); return best; };
    const mg = minOf("gelu"), ms = minOf("silu");
    DL.legend(g, acts.map(([k, c]) => ({ label: DL.act(k).label, color: c })), 4, 8, { gap: 12, font: 9.5 });
    TR.note(g, 4, lh - 18, `GELU min ${DL.fmt(mg[1], 3)} at z ≈ ${DL.fmt(mg[0], 2)}`, DC.a2, 9);
    TR.note(g, 4, lh - 6, `SiLU min ${DL.fmt(ms[1], 3)} at z ≈ ${DL.fmt(ms[0], 2)}`, DC.good, 9);
    /* middle: gated surface act(zg)·zu */
    const mx = lw + 44, mw = 200, mh = 200, K = 40;
    const gm = g.append("g").attr("transform", `translate(${mx},0)`);
    TR.title(gm, 0, -8, `gated unit: ${DL.act(a).label.split(" ")[0]}(z_g) · z_u`);
    const Z = []; let lim = 0;
    for (let i = 0; i < K; i++) { const row = []; for (let j = 0; j < K; j++) { const zg = -4 + 8 * j / (K - 1), zu = 4 - 8 * i / (K - 1); const v = DL.act(a).f(zg) * zu; row.push(v); lim = Math.max(lim, Math.abs(v)); } Z.push(row); }
    TR.heat(gm, Z, 0, 0, mw / K, TR.diverging(lim), { ch: mh / K });
    TR.note(gm, 0, mh + 12, "z_g = x·w_g →  (−4 … 4)", DC.muted, 9); TR.note(gm, mw + 4, 8, "z_u", DC.muted, 9); TR.note(gm, mw + 4, mh, "= x·w_u", DC.muted, 9);
    TR.note(gm, 0, mh + 26, "gate off (left): ≈ 0 for any z_u", DC.muted, 9);
    TR.note(gm, 0, mh + 38, "gate on (right): ± linear in z_u", DC.muted, 9);
    /* right: the rule */
    const rx = mx + mw + 50;
    const gr = g.append("g").attr("transform", `translate(${rx},0)`);
    const raw = (2 / 3) * 4 * d, rawM = raw * m, rounded = DL.ffnWidthGated(d, { ratio: 4, multiple: mult, mult: m });
    const pg = 3 * d * rounded, pu = 2 * d * 4 * d;
    TR.title(gr, 0, -8, "the width rule, evaluated");
    const kv = DL.kv(gr, 0, 14, { lead: 16, keyW: 118, size: 10 });
    kv("d", DL.commas(d)); kv("(2/3)·4d", DL.fmt(raw, 1)); kv("× m = " + m, DL.fmt(rawM, 1)); kv("⌈·/" + mult + "⌉·" + mult, DL.commas(rounded), DC.lime, true);
    kv("gated params 3·d·d_ff′", DL.commas(pg)); kv("ungated 2·d·4d", DL.commas(pu)); kv("ratio", DL.fmt(pg / pu, 3), pg / pu > 1.05 ? DC.a2 : DC.good);
    TR.note(gr, 0, 14 + 7 * 16 + 8, "11 008 · 14 336 · 28 672 are (4096, 1, 256),", DC.muted, 9);
    TR.note(gr, 0, 14 + 7 * 16 + 20, "(4096, 1.3, 1024) and (8192, 1.3, 4096)", DC.muted, 9);
    El("gl-readout").innerHTML =
      `Width rule at d = ${DL.commas(d)}, m = ${m}, multiple ${mult}: (2/3)·4d = ${DL.fmt(raw, 1)} → × m = ${DL.fmt(rawM, 1)} → rounded up to <b>${DL.commas(rounded)}</b>. ` +
      `Gated FFN parameters 3·d·d_ff′ = ${DL.commas(pg)} against the ungated 2·d·4d = ${DL.commas(pu)}: ratio <b>${DL.fmt(pg / pu, 3)}</b>${m > 1 ? " — the multiplier deliberately spends more on the FFN" : " — the same count to within the rounding"}. ` +
      `GELU's minimum is ${DL.fmt(mg[1], 3)} at z ≈ ${DL.fmt(mg[0], 2)}; SiLU's is ${DL.fmt(ms[1], 3)} at z ≈ ${DL.fmt(ms[0], 2)}; ReLU has none.`;
  }
  ["gl-a", "gl-d", "gl-m", "gl-mu"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 16 · #km-svg — the trained FFN read as a memory.  ═════════════ */
(function () {
  const svg = d3.select("#km-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const El = id => document.getElementById(id);
  function draw() {
    const l = +El("km-l").value, pos = +El("km-p").value, taskName = El("km-t").value;
    const R = TR.train({ task: taskName, steps: 300, seed: 1, cfg: { pos: "learned" } }), m = R.m, cfg = m.cfg;
    const dff = cfg.dff, V = cfg.V, iSel = Math.min(+El("km-i").value, dff - 1);
    El("km-i").max = dff - 1; El("km-pv").textContent = pos; El("km-iv").textContent = iSel;
    /* a fixed context from the task; place each token type at `pos` and read the FFN input of block l */
    const ex = TR.task(taskName, V, cfg.nCtx, DL.rng(31));
    const Act = DL.zeros2(V, dff);
    for (let t = 0; t < V; t++) {
      const idx = ex.idx.slice(); idx[pos] = t;
      const f = DL.tfForward(m, idx), st = f.st.blocks[l];
      Act[t] = st.f.H[pos].map(v => DL.act(m.blocks[l].ffn.act).f(v));
    }
    /* value logits: v_i (row i of W₂) through the tied unembedding, ignoring the final norm's rescaling */
    const W2 = m.blocks[l].ffn.W2, VL = DL.matmul(W2, DL.transpose(m.E));
    const f = DL.frame(svg, W, H, { l: 40, r: 10, t: 22, b: 10 }), g = f.g;
    const cw = 9.5, ch = 16;
    TR.title(g, 0, -8, `act(x·kᵢ + bᵢ): token type (rows) × slot (columns), block ${l + 1}, position ${pos}`);
    const la = Math.max(...Act.flat().map(Math.abs)) || 1;
    TR.heat(g, Act, 0, 0, cw, TR.diverging(la), { ch: ch });
    for (let t = 0; t < V; t++) TR.note(g, -4, t * ch + 11, TR.tokenLabel(t, V), DC.muted, 8.5, "end");
    g.append("rect").attr("x", iSel * cw).attr("y", 0).attr("width", cw).attr("height", V * ch).attr("fill", "none").attr("stroke", DC.lime).attr("stroke-width", 1.5);
    TR.note(g, 0, V * ch + 12, `slots 0 … ${dff - 1}; colour scale ±${DL.fmt(la, 2)}`, DC.muted, 9);
    /* right: value logits */
    const rx = dff * cw + 60, cw2 = 22, ch2 = 5.6;
    const gr = g.append("g").attr("transform", `translate(${rx},0)`);
    TR.title(gr, 0, -8, `vᵢ·Eᵀ: slot (rows) × vocabulary (columns)`);
    const lv = Math.max(...VL.flat().map(Math.abs)) || 1;
    TR.heat(gr, VL, 0, 0, cw2, TR.diverging(lv), { ch: ch2 });
    for (let t = 0; t < V; t++) TR.note(gr, t * cw2 + cw2 / 2, dff * ch2 + 11, TR.tokenLabel(t, V), DC.muted, 8.5, "middle");
    gr.append("rect").attr("x", 0).attr("y", iSel * ch2).attr("width", V * cw2).attr("height", ch2).attr("fill", "none").attr("stroke", DC.lime).attr("stroke-width", 1.5);
    /* read the selected slot */
    const col = Act.map(r => r[iSel]), topK = TR.argmax(col), topV = TR.argmax(VL[iSel]);
    const selective = d3.range(dff).filter(i => { const c = Act.map(r => r[i]); const mx = Math.max(...c); return c.filter(v => v > 0.5 * mx).length <= V / 2 && mx > 0.05 * la; }).length;
    const ky = V * ch + 40;
    TR.title(g, 0, ky, `slot ${iSel}`, DC.lime);
    TR.note(g, 0, ky + 16, `fires most on token ${TR.tokenLabel(topK, V)} (act ${DL.fmt(col[topK], 2)}), least on ${TR.tokenLabel(col.indexOf(Math.min(...col)), V)} (${DL.fmt(Math.min(...col), 2)})`, DC.ink, 9.5);
    TR.note(g, 0, ky + 30, `its value most favours token ${TR.tokenLabel(topV, V)} (+${DL.fmt(VL[iSel][topV], 2)} logit-units) and most suppresses ${TR.tokenLabel(VL[iSel].indexOf(Math.min(...VL[iSel])), V)}`, DC.ink, 9.5);
    TR.note(g, 0, ky + 44, `${selective} of ${dff} slots fire (≥ half their peak) on at most half the token types`, DC.muted, 9.5);
    TR.note(g, 0, ky + 56, "value logits ignore the final norm's per-row rescaling — a direction, not a calibrated number", DC.muted, 9);
    El("km-readout").innerHTML =
      `Block ${l + 1}, position ${pos}, task "${taskName}" (300 steps). Slot ${iSel}'s key fires most on <b>${TR.tokenLabel(topK, V)}</b> and its value most favours <b>${TR.tokenLabel(topV, V)}</b>${topK === topV ? " — a \"when I see it, say it\" slot" : ""}. ` +
      `<b>${selective}</b> of ${dff} slots are selective (fire at ≥ half their peak on at most ${V / 2} token types); mean activation over the grid ${DL.fmt(DL.mean(Act.flat()), 3)}. ` +
      `Keys are the ${dff} columns of W₁ (${cfg.d} × ${dff}); values are the ${dff} rows of W₂ (${dff} × ${cfg.d}).`;
  }
  ["km-l", "km-t"].forEach(id => d3.select("#" + id).on("change", draw));
  ["km-p", "km-i"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 17 · #fd-svg — the audit, live.  ══════════════════════════════ */
(function () {
  const svg = d3.select("#fd-svg"); if (svg.empty()) return;
  const W = 760, H = 420;
  const El = id => document.getElementById(id);
  function relErr(A, B) { let num = 0, den = 0; const a = A.flat(2), b = B.flat(2); for (let i = 0; i < a.length; i++) { num = Math.max(num, Math.abs(a[i] - b[i])); den = Math.max(den, Math.abs(b[i])); } return num / (den || 1); }
  const flat2 = M => Array.isArray(M[0]) ? M : [M];
  function audit(lossFn, mats, analytic, eps) {
    const num = DL.numGradMats(lossFn, mats, eps); let worst = 0;
    Object.keys(mats).forEach(k => worst = Math.max(worst, relErr(flat2(analytic[k]), flat2(num[k]))));
    return worst;
  }
  function run(eps, seed) {
    const r = DL.rng(seed * 101 + 7), n = 4, d = 8;
    const G = (a, b, s) => { const M = DL.zeros2(a, b); for (let i = 0; i < a; i++) for (let j = 0; j < b; j++) M[i][j] = DL.randn(r) * (s || 1); return M; };
    const inner = (Y, R) => { let s = 0; for (let i = 0; i < Y.length; i++) for (let j = 0; j < Y[0].length; j++) s += Y[i][j] * R[i][j]; return s; };
    const out = [];
    { const X = G(n, d, 2), gm = G(1, d)[0].map(v => 1 + 0.3 * v), bt = G(1, d)[0].map(v => 0.2 * v), R = G(n, d);
      const ln = DL.lnForward(X, gm, bt), bw = DL.lnBackward(R, ln.st);
      out.push(["LayerNorm (X, γ, β)", audit(() => inner(DL.lnForward(X, gm, bt).Y, R), { X: X, g: [gm], b: [bt] }, { X: bw.X, g: bw.g, b: bw.b }, eps)]);
      const rm = DL.rmsForward(X, gm), bw2 = DL.rmsBackward(R, rm.st);
      out.push(["RMSNorm (X, γ)", audit(() => inner(DL.rmsForward(X, gm).Y, R), { X: X, g: [gm] }, { X: bw2.X, g: bw2.g }, eps)]); }
    ["relu", "gelu", "silu"].forEach(a => [false, true].forEach(gated => {
      const P = DL.ffnInit(d, 12, { act: a, gated: gated, seed: seed + 4 }), X = G(n, d), R = G(n, d);
      const f = DL.ffnForward(X, P), bw = DL.ffnBackward(R, f.st, P);
      const mats = gated ? { Wg: P.Wg, Wu: P.Wu, Wd: P.Wd, X: X } : { W1: P.W1, b1: [P.b1], W2: P.W2, b2: [P.b2], X: X };
      const an = gated ? { Wg: bw.Wg, Wu: bw.Wu, Wd: bw.Wd, X: bw.X } : { W1: bw.W1, b1: bw.b1, W2: bw.W2, b2: bw.b2, X: bw.X };
      out.push([`FFN ${gated ? "gated" : "ungated"} ${a}`, audit(() => inner(DL.ffnForward(X, P).Y, R), mats, an, eps)]);
    }));
    { const P = DL.mhaInit(d, 4, 4, 2, { seed: seed + 9 }), X = G(n, d), R = G(n, d), M = DL.causalMask(n);
      ["adjacent", "half"].forEach(pair => {
        const masks = DL.alibiSlopes(2).map(sl => DL.addMasks(M, DL.alibiBias(n, sl))), ro = { base: 100, pairing: pair, offset: 3 };
        const f = DL.attnForward(X, P, { mask: masks, rope: ro }), bw = DL.attnBackward(R, f.st, P);
        out.push([`attention, RoPE (${pair}) + ALiBi`, audit(() => inner(DL.attnForward(X, P, { mask: masks, rope: ro }).Y, R), { Wq: P.Wq, Wk: P.Wk, Wv: P.Wv, Wo: P.Wo, X: X }, bw, eps)]);
      });
      const Pg = DL.mhaInit(d, 4, 4, 4, { seed: seed + 9, g: 2 }), ro = { base: 100, pairing: "adjacent", offset: 0 };
      const f = DL.attnForward(X, Pg, { mask: M, rope: ro }), bw = DL.attnBackward(R, f.st, Pg);
      out.push(["attention, RoPE + GQA (g = 2)", audit(() => inner(DL.attnForward(X, Pg, { mask: M, rope: ro }).Y, R), { Wq: Pg.Wq, Wk: Pg.Wk, Wv: Pg.Wv, Wo: Pg.Wo, X: X }, bw, eps)]); }
    [["ln", true, false], ["ln", false, false], ["rms", true, true], ["rms", false, true]].forEach(([norm, pre, gated]) => {
      const P = DL.blockInit(d, 2, 12, { norm: norm, pre: pre, gated: gated, seed: seed + 3 }), X = G(n, d), R = G(n, d), M = DL.causalMask(n), ro = { base: 100, pairing: "adjacent", offset: 0 };
      const f = DL.blockForward(X, P, { mask: M, rope: ro }), bw = DL.blockBackward(R, f.st, P);
      const mats = { Wq: P.attn.Wq, Wk: P.attn.Wk, Wv: P.attn.Wv, Wo: P.attn.Wo, n1g: [P.n1.g], n2g: [P.n2.g], X: X };
      const an = { Wq: bw.attn.Wq, Wk: bw.attn.Wk, Wv: bw.attn.Wv, Wo: bw.attn.Wo, n1g: bw.n1.g, n2g: bw.n2.g, X: bw.X };
      if (norm === "ln") { mats.n1b = [P.n1.b]; mats.n2b = [P.n2.b]; an.n1b = bw.n1.b; an.n2b = bw.n2.b; }
      if (gated) { mats.Wg = P.ffn.Wg; mats.Wu = P.ffn.Wu; mats.Wd = P.ffn.Wd; an.Wg = bw.ffn.Wg; an.Wu = bw.ffn.Wu; an.Wd = bw.ffn.Wd; }
      else { mats.W1 = P.ffn.W1; mats.b1 = [P.ffn.b1]; mats.W2 = P.ffn.W2; mats.b2 = [P.ffn.b2]; an.W1 = bw.ffn.W1; an.b1 = bw.ffn.b1; an.W2 = bw.ffn.W2; an.b2 = bw.ffn.b2; }
      out.push([`block ${pre ? "pre" : "post"}-norm, ${norm === "rms" ? "RMSNorm" : "LayerNorm"}, ${gated ? "gated" : "ungated"}`, audit(() => inner(DL.blockForward(X, P, { mask: M, rope: ro }).Y, R), mats, an, eps)]);
    });
    ["learned", "sin", "rope", "alibi", "none"].forEach(pos => {
      const m = DL.tfInit({ V: 6, d: 8, L: 2, h: 2, dff: 12, nCtx: 6, pos: pos, tie: pos !== "sin", scaleEmb: true, seed: seed + 2, norm: pos === "rope" ? "rms" : "ln", pre: pos !== "alibi" });
      const idx = [1, 4, 2, 5, 0, 3], tgt = [4, 2, 5, 0, 3, -1];
      const bw = DL.tfBackward(m, idx, tgt), pl = DL.tfParamList(m), gl = DL.tfGradList(bw.g, m), mats = {}, an = {};
      pl.forEach((p, i) => { mats[p.k] = Array.isArray(p.M[0]) ? p.M : [p.M]; an[p.k] = gl[i].M; });
      out.push([`whole model, pos = ${pos}${pos === "sin" ? ", untied" : ""}`, audit(() => DL.tfLoss(m, idx, tgt), mats, an, eps)]);
    });
    /* equalities */
    const X = G(n, d), P = DL.mhaInit(d, 4, 4, 2, { seed: seed + 9 }), M = DL.causalMask(n);
    const eqA = relErr(DL.attnForward(X, P, { mask: M }).Y, DL.mhaForward(X, P, { mask: M }).Y);
    const gm = G(1, d)[0], eqN = Math.max(relErr(DL.lnForward(X, gm, DL.zeros(d)).Y, DL.layerNorm(X, gm, DL.zeros(d))), relErr(DL.rmsForward(X, gm).Y, DL.rmsNorm(X, gm)));
    return { rows: out, eqA: eqA, eqN: eqN };
  }
  function draw() {
    const eps = +El("fd-e").value, seed = +El("fd-seed").value;
    El("fd-seedv").textContent = seed;
    const A = run(eps, seed);
    const f = DL.frame(svg, W, H, { l: 250, r: 20, t: 22, b: 30 }), g = f.g;
    const rows = A.rows, rh = (f.ih - 24) / (rows.length + 2);
    const x = d3.scaleLog().domain([1e-12, 1e-4]).range([0, f.iw]);
    TR.title(g, 0, -8, `worst relative error, analytic against central differences (ε = ${DL.fmtE(eps, 0)})`);
    DL.axisB(g, x, f.ih - 24, 8, "relative error", v => DL.fmtE(v, 0));
    [[1e-9, "expected (erf-free)", DC.good], [1e-6, "tolerance", DC.bad]].forEach(([v, lab, c]) => {
      g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", 0).attr("y2", f.ih - 24).attr("stroke", c).attr("stroke-dasharray", "4 3");
      TR.note(g, x(v) + 3, -1, lab, c, 9);
    });
    let worst = [null, 0];
    rows.forEach((r, i) => {
      const yy = i * rh, v = Math.max(r[1], 1e-12), bad = r[1] > 1e-6;
      g.append("rect").attr("x", 0).attr("y", yy + 2).attr("width", x(v)).attr("height", rh - 4).attr("fill", bad ? DC.bad : (r[0].indexOf("gelu") >= 0 || r[0].indexOf("whole") >= 0 || r[0].indexOf("block") >= 0 ? DC.a2 : DC.accent)).attr("fill-opacity", 0.8);
      TR.note(g, -6, yy + rh / 2 + 3, r[0], DC.ink, 9.5, "end");
      TR.note(g, x(v) + 4, yy + rh / 2 + 3, DL.fmtE(r[1], 1), bad ? DC.bad : DC.muted, 9);
      if (r[1] > worst[1]) worst = [r[0], r[1]];
    });
    const ey = rows.length * rh + 4;
    TR.note(g, -6, ey + 10, "attnForward vs mhaForward (no RoPE)", DC.ink, 9.5, "end"); TR.note(g, 4, ey + 10, `identical: ${DL.fmtE(A.eqA, 1)}`, DC.good, 9.5);
    TR.note(g, -6, ey + 24, "lnForward/rmsForward vs part 2's norms", DC.ink, 9.5, "end"); TR.note(g, 4, ey + 24, `identical: ${DL.fmtE(A.eqN, 1)}`, DC.good, 9.5);
    El("fd-readout").innerHTML =
      `${rows.length} backward passes audited at ε = ${DL.fmtE(eps, 0)}, seed ${seed}. Worst: <b>${DL.fmtE(worst[1], 1)}</b> (${worst[0]}). ` +
      `${rows.filter(r => r[1] > 1e-6).length === 0 ? "Every pass is below the 10⁻⁶ tolerance." : "<b>A pass exceeds the tolerance.</b>"} ` +
      `Passes with the exact GELU on the path sit near 10⁻⁷ to 10⁻⁸ because the error function itself is evaluated to about that accuracy; the erf-free passes sit at 10⁻⁹ to 10⁻¹¹. ` +
      `The attention block with RoPE off reproduces part 5's DL.mhaForward to ${DL.fmtE(A.eqA, 1)}, and the cached-norm forward passes match DL.layerNorm / DL.rmsNorm to ${DL.fmtE(A.eqN, 1)}.`;
  }
  d3.select("#fd-e").on("change", draw);
  d3.select("#fd-seed").on("input", draw);
  d3.select("#fd-go").on("click", draw);
  draw();
})();

/* ═════════ 18 · #st-svg — three masks and their reachability.  ═══════════ */
(function () {
  const svg = d3.select("#st-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const El = id => document.getElementById(id);
  function draw() {
    const kind = El("st-k").value, n = +El("st-n").value, pad = +El("st-pad").value, L = +El("st-L").value;
    El("st-p").max = n - 1; const p = Math.min(+El("st-p").value, n - 1);
    El("st-nv").textContent = n; El("st-pv").textContent = p; El("st-padv").textContent = pad; El("st-Lv").textContent = L;
    const valid = d3.range(n).map(j => j < n - pad);
    let M;
    if (kind === "enc") M = DL.zeros2(n, n);
    else if (kind === "dec") M = DL.causalMask(n);
    else { M = DL.causalMask(n); for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) M[i][j] = 0; }
    M = DL.addMasks(M, DL.padMask(n, valid));
    const B = M.map(r => r.map(v => isFinite(v) ? 1 : 0));
    /* reachability: R_L = (B + I)^L in Boolean arithmetic; the residual gives the +I */
    let R = B.map((r, i) => r.map((v, j) => (v || i === j) ? 1 : 0));
    const step = (A, C) => A.map((r, i) => r.map((_, j) => { for (let k = 0; k < n; k++) if (A[i][k] && C[k][j]) return 1; return 0; }));
    const R1 = R.map(r => r.slice()); for (let l = 1; l < L; l++) R = step(R, R1);
    const f = DL.frame(svg, W, H, { l: 30, r: 10, t: 22, b: 30 }), g = f.g;
    const cw = Math.min(14, 220 / n);
    TR.title(g, 0, -8, `mask, ${kind === "enc" ? "encoder" : kind === "dec" ? "decoder" : "prefix-LM (p = " + p + ")"}, ${pad} padded`);
    TR.heat(g, B, 0, 0, cw, v => v ? DC.accent : DC.grid, { stroke: DC.bg });
    TR.note(g, 0, n * cw + 12, "rows: query i · columns: key j · light = allowed", DC.muted, 9);
    const rx = 260;
    TR.title(g, rx, -8, `reachability after L = ${L} layer${L > 1 ? "s" : ""}`);
    TR.heat(g, R, rx, 0, cw, v => v ? DC.good : DC.grid, { stroke: DC.bg });
    TR.note(g, rx, n * cw + 12, "(i, j) lit = input j can influence output i", DC.muted, 9);
    const reach = R.map(r => r.reduce((s, v) => s + v, 0));
    const bx = 520, bw = f.iw - bx, bh = n * cw;
    const gb = g.append("g").attr("transform", `translate(${bx},0)`);
    TR.title(gb, 0, -8, "inputs reaching each output");
    const x = d3.scaleLinear().domain([0, n]).range([0, bw]);
    reach.forEach((v, i) => { gb.append("rect").attr("x", 0).attr("y", i * cw).attr("width", x(v)).attr("height", cw - 1).attr("fill", valid[i] ? DC.good : DC.muted).attr("fill-opacity", 0.8); TR.note(gb, x(v) + 3, i * cw + cw - 3, String(v), DC.ink, 8.5); });
    DL.axisB(gb, x, bh, 4, "count");
    const allowed = B.flat().reduce((s, v) => s + v, 0), causalCut = kind === "enc" ? 0 : B.length * B.length - B.flat().length + (DL.causalMask(n).flat().filter(v => !isFinite(v)).length - (kind === "prefix" ? p * (p - 1) / 2 : 0));
    const padCut = n * pad;
    El("st-readout").innerHTML =
      `n = ${n}, ${pad} padded keys, L = ${L}. Allowed (query, key) pairs: <b>${allowed}</b> of ${n * n}; removed by the ${kind === "enc" ? "(absent) causal" : "causal"} mask ${kind === "enc" ? 0 : n * (n - 1) / 2 - (kind === "prefix" ? p * (p - 1) / 2 : 0)}, by padding ${padCut} (overlap counted once in the total). ` +
      `Mean number of inputs that can influence an output: <b>${DL.fmt(DL.mean(reach.filter((_, i) => valid[i])), 2)}</b> over the unpadded positions — ` +
      (kind === "dec" ? `it grows by one per position and does not grow with depth: the lower triangle at L = 1 is the lower triangle at L = ${L}.` : kind === "enc" ? `every unpadded input, at every depth from L = 1 on.` : `the whole prefix for every position, plus the causal past for the suffix.`);
  }
  d3.select("#st-k").on("change", draw);
  ["st-n", "st-p", "st-pad", "st-L"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 19 · #xa-svg — cross-attention shapes and cost.  ══════════════ */
(function () {
  const svg = d3.select("#xa-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const El = id => document.getElementById(id);
  function draw() {
    const ns = +El("xa-s").value, nt = +El("xa-t").value, d = +El("xa-d").value, pad = +El("xa-pad").value;
    El("xa-sv").textContent = ns; El("xa-tv").textContent = nt; El("xa-padv").textContent = pad;
    const dm = 16, dk = 8;
    const He = TR.gauss(ns, dm, 4, 1), U = TR.gauss(nt, dm, 5, 1);
    const Wq = TR.gauss(dm, dk, 6, 1 / 4), Wk = TR.gauss(dm, dk, 7, 1 / 4), Wv = TR.gauss(dm, dk, 8, 1 / 4);
    const valid = d3.range(ns).map(j => j < ns - pad);
    const out = DL.sdpa(DL.matmul(U, Wq), DL.matmul(He, Wk), DL.matmul(He, Wv), { mask: DL.padMask(nt, valid) });
    const rowSums = out.A.map(r => r.reduce((s, v) => s + v, 0)), padMass = out.A.map(r => r.slice(ns - pad).reduce((s, v) => s + v, 0));
    const f = DL.frame(svg, W, H, { l: 40, r: 10, t: 22, b: 30 }), g = f.g;
    const cw = Math.min(14, 300 / Math.max(ns, nt));
    TR.title(g, 0, -8, `A = softmax(QKᵀ/√dₖ + pad): (${nt} × ${ns}), rows = target, columns = source`);
    TR.heat(g, out.A, 0, 0, cw, d3.scaleSequential(d3.interpolateViridis).domain([0, Math.max(...out.A.flat())]), { stroke: DC.bg });
    TR.note(g, 0, nt * cw + 12, `Q = N(X_tgt)·W_Q (${nt} × dₖ) · K, V = Hₑ·W_K, Hₑ·W_V (${ns} × dₖ) · Z = A·V (${nt} × dᵥ)`, DC.muted, 9);
    TR.note(g, 0, nt * cw + 24, `row sums: min ${DL.fmt(Math.min(...rowSums), 6)}, max ${DL.fmt(Math.max(...rowSums), 6)} · mass on the ${pad} padded columns: ${DL.fmtE(Math.max(...padMass), 1)}`, DC.ink, 9);
    /* right: cost per decoded token against n_src */
    const rx = 360, rw = f.iw - rx, rh = f.ih - 20;
    const gr = g.append("g").attr("transform", `translate(${rx},0)`);
    TR.title(gr, 0, -8, `MACs per decoded token, one cross-attention layer, d = ${d}`);
    const srcs = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    const qproj = d * d, blend = s => 2 * s * d, kvproj = s => 2 * s * d * d / nt;
    const x = d3.scaleLog().domain([8, 4096]).range([0, rw]), y = d3.scaleLog().domain([1e3, Math.max(kvproj(4096), qproj) * 2]).range([rh, 0]);
    DL.gridY(gr, y, rw, 4); DL.axisB(gr, x, rh, 5, "n_src", v => Number.isInteger(Math.log2(v)) && Math.log2(v) % 2 === 1 ? DL.big(v) : ""); DL.axisL(gr, y, 4, "MACs", v => Math.abs(Math.log10(v) % 1) < 1e-9 ? DL.big(v) : "");
    DL.curve(gr, srcs.map(s => [x(s), y(qproj)]), { stroke: DC.a2, w: 1.8 });
    DL.curve(gr, srcs.map(s => [x(s), y(blend(s))]), { stroke: DC.good, w: 1.8 });
    DL.curve(gr, srcs.map(s => [x(s), y(kvproj(s))]), { stroke: DC.violet, w: 1.8, dash: "4 3" });
    gr.append("line").attr("x1", x(Math.max(8, ns))).attr("x2", x(Math.max(8, ns))).attr("y1", 0).attr("y2", rh).attr("stroke", DC.ink).attr("stroke-dasharray", "3 3");
    DL.legend(gr, [{ label: "Q projection d² (every step)", color: DC.a2 }, { label: "scores + blend 2·n_src·d (every step)", color: DC.good }, { label: `K, V projections 2·n_src·d² ÷ n_tgt = ${nt} (once per source)`, color: DC.violet, dash: "4 3" }], 4, 10, { gap: 12, font: 9 });
    const s0 = ns;
    El("xa-readout").innerHTML =
      `n_src = ${ns} (${pad} padded), n_tgt = ${nt}. Every row of A sums to one over the unpadded source (min ${DL.fmt(Math.min(...rowSums), 6)}, max ${DL.fmt(Math.max(...rowSums), 6)}); the padded columns carry <b>${DL.fmtE(Math.max(...padMass), 1)}</b>. ` +
      `At d = ${d} one decoded token's cross-attention costs: Q projection <b>${DL.big(qproj)}</b>, scores + blend <b>${DL.big(blend(s0))}</b>, and the once-per-source K, V projections amortised over ${nt} target tokens <b>${DL.big(kvproj(s0))}</b>; the scores term overtakes the Q projection only at n_src = ${DL.commas(d / 2)}.`;
  }
  d3.select("#xa-d").on("change", draw);
  ["xa-s", "xa-t", "xa-pad"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 20 · #em-svg — the √d embedding scale, measured.  ═════════════ */
(function () {
  const svg = d3.select("#em-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const El = id => document.getElementById(id);
  function draw() {
    const init = El("em-i").value, scaled = El("em-s").checked, posKind = El("em-p").value, d = +El("em-d").value;
    const V = 40, n = 16, r = DL.rng(11);
    const std = init === "inv" ? 1 / Math.sqrt(d) : 0.02;
    const E = TR.gauss(V, d, 21, std), P = posKind === "sin" ? DL.sinusoidalPE(n, d) : TR.gauss(n, d, 22, 0.02);
    const toks = d3.range(n).map(() => Math.floor(r() * V)), toks2 = toks.map(t => (t + 7 + Math.floor(r() * (V - 1))) % V);
    const fac = scaled ? Math.sqrt(d) : 1;
    const build = tk => tk.map((t, i) => E[t].map((v, j) => v * fac + P[i][j]));
    const X = build(toks), X2 = build(toks2);
    const nE = DL.mean(toks.map(t => TR.rowNorm(E[t]) * fac)), nP = DL.mean(P.slice(0, n).map(TR.rowNorm));
    const cosE = DL.mean(X.map((x, i) => DL.dot(x, E[toks[i]]) / (TR.rowNorm(x) * TR.rowNorm(E[toks[i]]))));
    const cosP = DL.mean(X.map((x, i) => DL.dot(x, P[i]) / (TR.rowNorm(x) * TR.rowNorm(P[i]))));
    /* first-layer scores with random Wq, Wk */
    const dk = 16, Wq = TR.gauss(d, dk, 31, 1 / Math.sqrt(d)), Wk = TR.gauss(d, dk, 32, 1 / Math.sqrt(d));
    const S1 = DL.sdpa(DL.matmul(X, Wq), DL.matmul(X, Wk), X, {}).S, S2 = DL.sdpa(DL.matmul(X2, Wq), DL.matmul(X2, Wk), X2, {}).S;
    const a = S1.flat(), b = S2.flat(), ma = DL.mean(a), mb = DL.mean(b);
    let num = 0, da = 0, db = 0; a.forEach((v, i) => { num += (v - ma) * (b[i] - mb); da += (v - ma) ** 2; db += (b[i] - mb) ** 2; });
    const corr = num / Math.sqrt(da * db);
    const f = DL.frame(svg, W, H, { l: 44, r: 10, t: 22, b: 30 }), g = f.g;
    /* left bars */
    const lw = 180, lh = f.ih;
    TR.title(g, 0, -8, "mean row norms");
    const y = d3.scaleLinear().domain([0, Math.max(nE, nP) * 1.15]).range([lh, 0]);
    DL.gridY(g, y, lw, 4); DL.axisL(g, y, 4);
    [[`${scaled ? "√d·" : ""}E[token]`, nE, DC.accent], ["P[i]", nP, DC.a2]].forEach((bb, i) => {
      g.append("rect").attr("x", 20 + i * 80).attr("y", y(bb[1])).attr("width", 56).attr("height", lh - y(bb[1])).attr("fill", bb[2]).attr("fill-opacity", 0.85);
      TR.note(g, 48 + i * 80, y(bb[1]) - 5, DL.fmt(bb[1], 2), DC.ink, 10, "middle"); TR.note(g, 48 + i * 80, lh + 14, bb[0], DC.muted, 9.5, "middle");
    });
    TR.note(g, 0, lh + 27, `ratio token/position = ${DL.fmt(nE / nP, 3)}`, DC.ink, 9.5);
    /* middle: cosines */
    const mx = 240, mw = 150;
    const gm = g.append("g").attr("transform", `translate(${mx},0)`);
    TR.title(gm, 0, -8, "cosine of x⁰ with each part");
    const yc = d3.scaleLinear().domain([0, 1]).range([lh, 0]);
    DL.gridY(gm, yc, mw, 4); DL.axisL(gm, yc, 4);
    [["with E[token]", cosE, DC.accent], ["with P[i]", cosP, DC.a2]].forEach((bb, i) => {
      gm.append("rect").attr("x", 20 + i * 70).attr("y", yc(bb[1])).attr("width", 50).attr("height", lh - yc(bb[1])).attr("fill", bb[2]).attr("fill-opacity", 0.85);
      TR.note(gm, 45 + i * 70, yc(bb[1]) - 5, DL.fmt(bb[1], 3), DC.ink, 10, "middle"); TR.note(gm, 45 + i * 70, lh + 14, bb[0], DC.muted, 9.5, "middle");
    });
    /* right: two score matrices */
    const rx = 430, cw = Math.min(9, (f.iw - rx) / (2 * n + 2));
    const gr = g.append("g").attr("transform", `translate(${rx},0)`);
    TR.title(gr, 0, -8, "first-layer scores: real tokens · tokens replaced");
    const lim = Math.max(TR.maxAbs(S1), TR.maxAbs(S2)), scl = TR.diverging(lim);
    TR.heat(gr, S1, 0, 0, cw, scl); TR.heat(gr, S2, n * cw + 12, 0, cw, scl);
    TR.note(gr, 0, n * cw + 14, `correlation between the two: ${DL.fmt(corr, 3)}`, corr > 0.9 ? DC.bad : DC.good, 10);
    TR.note(gr, 0, n * cw + 27, corr > 0.9 ? "the scores barely depend on the tokens" : "the scores depend on the tokens", DC.muted, 9.5);
    El("em-readout").innerHTML =
      `d = ${d}, E ~ N(0, ${init === "inv" ? "1/d" : "0.02²"})${scaled ? " × √d" : ""}, ${posKind} positions. Mean norms: token term <b>${DL.fmt(nE, 2)}</b>, position term <b>${DL.fmt(nP, 2)}</b>, ratio <b>${DL.fmt(nE / nP, 3)}</b>. ` +
      `cos(x⁰, token part) = ${DL.fmt(cosE, 3)}, cos(x⁰, position part) = ${DL.fmt(cosP, 3)}. Replacing every token and recomputing a random first layer's scores gives a correlation of <b>${DL.fmt(corr, 3)}</b> with the original — ` +
      (corr > 0.9 ? `the first layer's attention pattern is a function of position alone.` : `the first layer's attention pattern carries the tokens.`);
  }
  ["em-i", "em-s", "em-p", "em-d"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 21 · #un-svg — the head's share, and the initial loss.  ═══════ */
(function () {
  const svg = d3.select("#un-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const El = id => document.getElementById(id);
  const FAM = [[768, 12, 12], [1024, 24, 16], [2048, 24, 16], [4096, 32, 32], [8192, 80, 64]];
  function draw() {
    const V = +El("un-V").value, nmax = +El("un-n").value, tv = +El("un-tv").value, seed = +El("un-seed").value;
    El("un-seedv").textContent = seed;
    const rows = FAM.map(([d, L, h]) => {
      const Pt = DL.modelParams({ V: V, d: d, L: L, h: h, dff: 4 * d, nCtx: nmax, tie: true }), Pu = DL.modelParams({ V: V, d: d, L: L, h: h, dff: 4 * d, nCtx: nmax, tie: false });
      const macs = Pt.nonEmbed - Pt.norms + V * d;
      return { d: d, L: L, tied: Pt.embed / Pt.total, untied: (Pu.embed + Pu.unembed) / Pu.total, mac: V * d / macs, total: Pu.total };
    });
    const f = DL.frame(svg, W, H, { l: 44, r: 10, t: 22, b: 30 }), g = f.g;
    const lw = 380, lh = f.ih;
    TR.title(g, 0, -8, `share of the vocabulary head, V = ${DL.commas(V)}, n_max = ${nmax}`);
    const x = d3.scaleLog().domain([768, 8192]).range([0, lw]), y = d3.scaleLinear().domain([0, Math.max(...rows.map(r => r.untied)) * 1.15]).range([lh, 0]);
    DL.gridY(g, y, lw, 4); DL.axisB(g, x, lh, 5, "d_model (with the depth of the family)", v => [768, 1024, 2048, 4096, 8192].indexOf(v) >= 0 ? String(v) : ""); DL.axisL(g, y, 4, "share", v => Math.round(100 * v) + "%");
    DL.curve(g, rows.map(r => [x(r.d), y(r.untied)]), { stroke: DC.a2, w: 2 });
    DL.curve(g, rows.map(r => [x(r.d), y(r.tied)]), { stroke: DC.accent, w: 2 });
    DL.curve(g, rows.map(r => [x(r.d), y(r.mac)]), { stroke: DC.good, w: 2, dash: "4 3" });
    rows.forEach(r => TR.note(g, x(r.d), lh - 4, DL.big(r.total), DC.muted, 8.5, "middle"));
    DL.legend(g, [{ label: "parameters, untied (E and W_out)", color: DC.a2 }, { label: "parameters, tied (E only)", color: DC.accent }, { label: "per-token MACs (d·V of the total)", color: DC.good, dash: "4 3" }], lw - 200, 8, { gap: 12, font: 9 });
    /* right: initial loss under four conventions */
    const rx = lw + 44, rw = f.iw - rx;
    const gr = g.append("g").attr("transform", `translate(${rx},0)`);
    const d = tv === 50 ? 64 : 256;
    const convs = [["untied, N(0,1/d)·√d", { tie: false, scaleEmb: true }, DC.good], ["tied, N(0,0.02²)", { tie: true, scaleEmb: false, embStd: 0.02 }, DC.good], ["tied, N(0,1/d)·√d", { tie: true, scaleEmb: true }, DC.a2], ["tied, N(0,1), no √d", { tie: true, scaleEmb: false }, DC.bad]];
    const losses = convs.map(c => {
      const m = DL.tfInit(Object.assign({ V: tv, d: d, L: 2, h: 2, dff: 4 * d, nCtx: 8, pos: "learned", seed: seed }, c[1]));
      const r = DL.rng(seed + 3); let L = 0;
      for (let t = 0; t < 8; t++) { const idx = []; for (let i = 0; i < 8; i++) idx.push(Math.floor(r() * tv)); L += DL.tfLoss(m, idx, idx.slice(1).concat([-1])) / 8; }
      return L;
    });
    TR.title(gr, 0, -8, `initial loss, V = ${tv}, d = ${d}`);
    const yl = d3.scaleLog().domain([1, Math.max(...losses) * 1.5]).range([lh, 0]);
    DL.gridY(gr, yl, rw, 4); DL.axisL(gr, yl, 4, "loss", v => Math.abs(Math.log10(v) % 1) < 1e-9 ? String(v) : "");
    const bwid = rw / 4 - 8;
    losses.forEach((L, i) => {
      gr.append("rect").attr("x", i * (bwid + 8)).attr("y", yl(L)).attr("width", bwid).attr("height", lh - yl(L)).attr("fill", convs[i][2]).attr("fill-opacity", 0.85);
      TR.note(gr, i * (bwid + 8) + bwid / 2, yl(L) - 4, DL.fmt(L, 2), DC.ink, 9.5, "middle");
      convs[i][0].split(", ").forEach((t, k) => TR.note(gr, i * (bwid + 8) + bwid / 2, lh + 12 + k * 10, t, DC.muted, 8, "middle"));
    });
    gr.append("line").attr("x1", 0).attr("x2", rw).attr("y1", yl(Math.log(tv))).attr("y2", yl(Math.log(tv))).attr("stroke", DC.ink).attr("stroke-dasharray", "4 3");
    TR.note(gr, rw, yl(Math.log(tv)) - 4, `ln V = ${DL.fmt(Math.log(tv), 2)}`, DC.ink, 9.5, "end");
    El("un-readout").innerHTML =
      `Head share at V = ${DL.commas(V)}: smallest configuration (d = 768, 12 blocks, ${DL.big(rows[0].total)} untied) <b>${TR.pct(rows[0].untied)}</b> of parameters untied, ${TR.pct(rows[0].tied)} tied, <b>${TR.pct(rows[0].mac)}</b> of per-token MACs; largest (d = 8192, 80 blocks, ${DL.big(rows[4].total)}) ${TR.pct(rows[4].untied)}, ${TR.pct(rows[4].tied)} and ${TR.pct(rows[4].mac)}. ` +
      `Initial loss of untrained two-layer models against ln V = ${DL.fmt(Math.log(tv), 2)}: ` + convs.map((c, i) => `${c[0]} <b>${DL.fmt(losses[i], 2)}</b>`).join(", ") +
      `. The last is the bug (logits √d = ${Math.sqrt(d).toFixed(0)}× too large); the third is tying's identity leak, ${DL.fmt(losses[2] - Math.log(tv), 2)} nats above ln V.`;
  }
  ["un-V", "un-n", "un-tv"].forEach(id => d3.select("#" + id).on("change", draw));
  d3.select("#un-seed").on("input", draw);
  draw();
})();

/* ═════════ 22 · #pc-svg — parameters by component, five configurations. ══ */
(function () {
  const svg = d3.select("#pc-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const El = id => document.getElementById(id);
  const PARTS = [["embed", "token embedding", DC.accent], ["pos", "positions", DC.teal], ["attn", "attention", DC.a2], ["ffn", "FFN", DC.violet], ["norms", "norms", DC.muted], ["unembed", "unembedding", DC.rose]];
  function draw() {
    const ci = +El("pc-c").value, gs = El("pc-g").value, ts = El("pc-t").value, fs = El("pc-f").value;
    const base = TR.CFGS[ci], mod = Object.assign({}, base);
    if (gs === "h") mod.g = null; else if (gs !== "cfg") mod.g = Math.min(+gs, base.h);
    if (ts !== "cfg") mod.tie = ts === "tied";
    if (fs === "u") { mod.gated = false; mod.dff = 4 * base.d; mod.bias = true; } else if (fs === "g") { mod.gated = true; mod.dff = DL.ffnWidthGated(base.d, { multiple: 256 }); mod.bias = false; }
    const all = TR.CFGS.map(c => DL.modelParams(c)), P0 = DL.modelParams(base), P1 = DL.modelParams(mod);
    const f = DL.frame(svg, W, H, { l: 90, r: 10, t: 22, b: 30 }), g = f.g;
    const lw = 400, rh = 34;
    TR.title(g, 0, -8, "parameters by component (log axis)");
    const x = d3.scaleLog().domain([1e6, 1e11]).range([0, lw]);
    DL.axisB(g, x, 5 * rh + 10, 5, "parameters", v => DL.big(v));
    all.forEach((P, i) => {
      let acc = 0; const yy = i * rh + 6;
      TR.note(g, -6, yy + 15, TR.CFGS[i].name, i === ci ? DC.ink : DC.muted, 9.5, "end");
      PARTS.forEach(([k, lab, c]) => { const v = P[k]; if (!v) return; const x0 = x(Math.max(1e6, acc)), x1 = x(Math.max(1e6, acc + v)); g.append("rect").attr("x", x0).attr("y", yy).attr("width", Math.max(0, x1 - x0)).attr("height", rh - 10).attr("fill", c).attr("fill-opacity", 0.85); acc += v; });
      TR.note(g, x(P.total) + 4, yy + 15, DL.big(P.total), DC.ink, 9);
    });
    DL.legend(g, PARTS.map(p => ({ label: p[1], color: p[2] })), 0, 5 * rh + 40, { vertical: false, step: 92, gap: 0, font: 9 });
    /* right: the chosen configuration, original against modified */
    const rx = lw + 40, rw = f.iw - rx, bh = f.ih - 60;
    const gr = g.append("g").attr("transform", `translate(${rx},0)`);
    TR.title(gr, 0, -8, `${base.name}: as configured · modified`);
    const tot = Math.max(P0.total, P1.total), y = d3.scaleLinear().domain([0, tot]).range([bh, 0]);
    [P0, P1].forEach((P, k) => {
      let acc = 0; const bx = 10 + k * 90;
      PARTS.forEach(([key, lab, c]) => { const v = P[key]; if (!v) return; gr.append("rect").attr("x", bx).attr("y", y(acc + v)).attr("width", 60).attr("height", y(acc) - y(acc + v)).attr("fill", c).attr("fill-opacity", 0.85); acc += v; });
      TR.note(gr, bx + 30, y(P.total) - 5, DL.big(P.total), DC.ink, 9.5, "middle");
      TR.note(gr, bx + 30, bh + 13, k ? "modified" : "configured", DC.muted, 9, "middle");
    });
    const deltas = PARTS.map(([k, lab]) => [lab, P1[k] - P0[k]]).filter(d => d[1] !== 0);
    deltas.forEach((d, i) => TR.note(gr, 190, 14 + i * 13, `${d[0]}: ${d[1] > 0 ? "+" : "−"}${DL.big(Math.abs(d[1]))}`, d[1] > 0 ? DC.a2 : DC.good, 9.5));
    if (!deltas.length) TR.note(gr, 190, 14, "no change", DC.muted, 9.5);
    El("pc-readout").innerHTML =
      `${base.name} as configured: <b>${DL.commas(P0.total)}</b> parameters — embedding ${DL.big(P0.embed)}, positions ${DL.big(P0.pos)}, attention ${DL.big(P0.attn)}, FFN ${DL.big(P0.ffn)}, norms ${DL.big(P0.norms)}, unembedding ${DL.big(P0.unembed)}; ${DL.commas(P0.perBlock.total)} per block, blocks are ${TR.pct(P0.blocks / P0.total)} of the total. ` +
      (deltas.length ? `Modified (g = ${mod.g === null || mod.g === undefined ? "h" : mod.g}, ${mod.tie ? "tied" : "untied"}, ${mod.gated ? "gated d_ff = " + DL.commas(mod.dff) : "ungated d_ff = " + DL.commas(mod.dff)}): <b>${DL.commas(P1.total)}</b>, a change of ${P1.total > P0.total ? "+" : "−"}${DL.commas(Math.abs(P1.total - P0.total))} (${TR.pct(Math.abs(P1.total - P0.total) / P0.total)}).` : `No modification selected.`);
  }
  ["pc-c", "pc-g", "pc-t", "pc-f"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 23 · #fl-svg — 6·N·D and the attention term.  ════════════════ */
(function () {
  const svg = d3.select("#fl-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const El = id => document.getElementById(id);
  function draw() {
    const n = Math.pow(2, +El("fl-n").value), unit = El("fl-u").value, causal = El("fl-c").value === "causal";
    El("fl-nv").textContent = DL.commas(n);
    const mul = unit === "flops" ? 6 : 3, uname = unit === "flops" ? "FLOPs" : "MACs";
    const rows = TR.CFGS.map(c => { const F = DL.modelFlops(c, n, { causal: causal }); return { name: c.name, w: mul * F.weightMacs, a: mul * F.attnMacs, share: F.attnMacs / F.macs, d: c.d }; });
    const f = DL.frame(svg, W, H, { l: 90, r: 10, t: 22, b: 30 }), g = f.g;
    const lw = 380, rh = 34;
    TR.title(g, 0, -8, `training ${uname} per token at n = ${DL.commas(n)}: weight term · attention term`);
    const x = d3.scaleLog().domain([1e8, 1e12]).range([0, lw]);
    DL.axisB(g, x, 5 * rh + 6, 5, uname + " per token", v => Math.abs(Math.log10(v) % 1) < 1e-9 ? DL.big(v) : "");
    rows.forEach((r, i) => {
      const yy = i * rh + 4;
      TR.note(g, -6, yy + 15, r.name, DC.ink, 9.5, "end");
      g.append("rect").attr("x", x(1e8)).attr("y", yy).attr("width", x(r.w) - x(1e8)).attr("height", rh - 10).attr("fill", DC.violet).attr("fill-opacity", 0.85);
      g.append("rect").attr("x", x(r.w)).attr("y", yy).attr("width", Math.max(0, x(r.w + r.a) - x(r.w))).attr("height", rh - 10).attr("fill", DC.a2).attr("fill-opacity", 0.9);
      TR.note(g, x(r.w + r.a) + 4, yy + 15, `${DL.big(r.w + r.a)} · attention ${TR.pct(r.share)}`, DC.ink, 9);
    });
    DL.legend(g, [{ label: `${mul}·N_weights (flat in n)`, color: DC.violet }, { label: `${mul}·L·n·d${causal ? "" : "·2"} (linear in n)`, color: DC.a2 }], 0, 5 * rh + 34, { vertical: false, step: 190, font: 9.5 });
    /* right: the two terms against n for the 7B-class */
    const rx = lw + 40, rw = f.iw - rx, rh2 = f.ih - 20, c7 = TR.CFGS[3];
    const gr = g.append("g").attr("transform", `translate(${rx},0)`);
    TR.title(gr, 0, -8, `${c7.name}: both terms against n`);
    const ns = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
    const term = nn => { const F = DL.modelFlops(c7, nn, { causal: causal }); return [mul * F.weightMacs, mul * F.attnMacs]; };
    const xs = d3.scaleLog().domain([256, 131072]).range([0, rw]), ys = d3.scaleLog().domain([1e8, 1e12]).range([rh2, 0]);
    DL.gridY(gr, ys, rw, 4); DL.axisB(gr, xs, rh2, 4, "n", v => Number.isInteger(Math.log2(v)) && Math.log2(v) % 2 === 0 ? DL.big(v) : ""); DL.axisL(gr, ys, 4, uname, v => Math.abs(Math.log10(v) % 1) < 1e-9 ? DL.big(v) : "");
    DL.curve(gr, ns.map(nn => [xs(nn), ys(term(nn)[0])]), { stroke: DC.violet, w: 2 });
    DL.curve(gr, ns.map(nn => [xs(nn), ys(term(nn)[1])]), { stroke: DC.a2, w: 2 });
    gr.append("line").attr("x1", xs(n)).attr("x2", xs(n)).attr("y1", 0).attr("y2", rh2).attr("stroke", DC.ink).attr("stroke-dasharray", "3 3");
    const cross = 12 * c7.d * (causal ? 1 : 0.5) * (1 + c7.V / (12 * c7.L * c7.d));
    TR.note(gr, 2, 12, `equal at n ≈ ${DL.commas(Math.round(cross))}`, DC.ink, 9.5);
    const peak = 1e15, r7 = rows[3];
    El("fl-readout").innerHTML =
      `Per token at n = ${DL.commas(n)}, ${causal ? "causal kernel" : "full n² kernel"}, in ${uname}: ` + rows.map(r => `${r.name} <b>${DL.big(r.w + r.a)}</b> (attention ${TR.pct(r.share)})`).join(", ") +
      `. For the 7B-class the weight term is ${DL.big(r7.w)} and the attention term ${DL.big(r7.a)}; the two are equal at n ≈ ${DL.commas(Math.round(cross))} ≈ 12·d. ` +
      `A device sustaining 10¹⁵ FLOP/s would train the 7B-class model at <b>${DL.commas(Math.round(peak / (unit === "flops" ? (r7.w + r7.a) : 2 * (r7.w + r7.a))))}</b> tokens per second at full utilisation.`;
  }
  ["fl-u", "fl-c"].forEach(id => d3.select("#" + id).on("change", draw));
  d3.select("#fl-n").on("input", draw);
  draw();
})();

/* ═════════ 24 · #tf-svg — teacher forcing, on the trained model.  ════════ */
(function () {
  const svg = d3.select("#tf-svg"); if (svg.empty()) return;
  const W = 760, H = 400;
  const El = id => document.getElementById(id);
  function freeRun(m, idx, tgt) {
    /* feed the model its own argmax on every supervised position */
    const seq = idx.slice(), out = [];
    const first = tgt.findIndex(t => t >= 0);
    for (let i = first; i < idx.length; i++) {
      if (tgt[i] < 0) { out.push(null); continue; }
      const f = DL.tfForward(m, seq.slice(0, i + 1)), a = TR.argmax(f.logits[i]);
      out.push(a);
      if (i + 1 < seq.length) seq[i + 1] = a;
    }
    return out;
  }
  function draw() {
    const taskName = El("tf-t").value, ei = +El("tf-e").value, steps = +El("tf-s").value;
    El("tf-ev").textContent = ei;
    const R = TR.train({ task: taskName, steps: steps, seed: 1, cfg: { pos: "learned" } }), m = R.m, V = m.cfg.V, n = m.cfg.nCtx;
    const ex = R.evalSet[ei], fw = DL.tfForward(m, ex.idx);
    const losses = ex.tgt.map((t, i) => t < 0 ? null : DL.xent(fw.logits[i], t));
    const sup = losses.filter(v => v !== null), meanL = DL.mean(sup);
    const f = DL.frame(svg, W, H, { l: 60, r: 10, t: 22, b: 10 }), g = f.g;
    const cw = 34;
    const rowCells = (y, vals, lab, color) => {
      TR.note(g, -6, y + 15, lab, DC.muted, 9.5, "end");
      vals.forEach((v, i) => {
        const on = v !== null && v >= 0;
        g.append("rect").attr("x", i * cw).attr("y", y).attr("width", cw - 3).attr("height", 22).attr("rx", 3).attr("fill", on ? color : DC.grid).attr("fill-opacity", on ? 0.8 : 1);
        TR.note(g, i * cw + (cw - 3) / 2, y + 15, on ? TR.tokenLabel(v, V) : "—", on ? DC.bg : DC.muted, 9.5, "middle");
      });
    };
    TR.title(g, 0, -8, `task "${taskName}", example ${ei}, model trained ${steps} steps`);
    rowCells(0, ex.idx, "input tᵢ", DC.accent);
    rowCells(30, ex.tgt, "target tᵢ₊₁", DC.a2);
    TR.note(g, 0, 68, "grey = no target at this position (the prompt half); the answer half is supervised", DC.muted, 9);
    /* mask */
    const my = 86, mc = 9;
    TR.note(g, -6, my + 10, "mask", DC.muted, 9.5, "end");
    TR.heat(g, DL.causalMask(n).map(r => r.map(v => isFinite(v) ? 1 : 0)), 0, my, mc, v => v ? DC.accent : DC.grid, { stroke: DC.bg });
    /* per-position loss */
    const bx = n * mc + 30, bw = 250, bh = 100;
    const gb = g.append("g").attr("transform", `translate(${bx},${my})`);
    TR.title(gb, 0, -6, "loss per supervised position, one forward pass");
    const yl = d3.scaleLinear().domain([0, Math.max(0.5, ...sup) * 1.15]).range([bh, 0]);
    DL.axisL(gb, yl, 3, "−log p");
    losses.forEach((v, i) => { if (v === null) return; gb.append("rect").attr("x", i * (bw / n)).attr("y", yl(v)).attr("width", bw / n - 3).attr("height", bh - yl(v)).attr("fill", DC.a2).attr("fill-opacity", 0.85); TR.note(gb, i * (bw / n) + (bw / n - 3) / 2, bh + 11, String(i), DC.muted, 8.5, "middle"); });
    gb.append("line").attr("x1", 0).attr("x2", bw).attr("y1", yl(meanL)).attr("y2", yl(meanL)).attr("stroke", DC.ink).attr("stroke-dasharray", "3 3");
    TR.note(gb, bw, yl(meanL) - 4, `mean ${DL.fmt(meanL, 3)}`, DC.ink, 9, "end");
    /* free-running vs teacher-forced on the answer half */
    const fy = my + n * mc + 40;
    const tfOut = ex.tgt.map((t, i) => t < 0 ? null : TR.argmax(fw.logits[i])), frOut = freeRun(m, ex.idx, ex.tgt);
    const pad = arr => arr.concat(new Array(n - arr.length).fill(null));
    TR.title(g, 0, fy - 6, "the answer half, predicted two ways (mismatches with the target outlined)");
    const rowPred = (y, vals, lab) => {
      TR.note(g, -6, y + 15, lab, DC.muted, 9.5, "end");
      vals.forEach((v, i) => {
        if (v === null || v === undefined) { g.append("rect").attr("x", i * cw).attr("y", y).attr("width", cw - 3).attr("height", 22).attr("rx", 3).attr("fill", DC.grid); return; }
        const ok = v === ex.tgt[i];
        g.append("rect").attr("x", i * cw).attr("y", y).attr("width", cw - 3).attr("height", 22).attr("rx", 3).attr("fill", ok ? DC.good : DC.bad).attr("fill-opacity", 0.75);
        TR.note(g, i * cw + (cw - 3) / 2, y + 15, TR.tokenLabel(v, V), DC.bg, 9.5, "middle");
      });
    };
    rowPred(fy, tfOut, "teacher-forced");
    const frFull = new Array(n).fill(null); const first = ex.tgt.findIndex(t => t >= 0); frOut.forEach((v, k) => frFull[first + k] = v);
    rowPred(fy + 30, frFull, "free-running");
    /* accuracies over the held-out set */
    let okTF = 0, okFR = 0, cnt = 0;
    R.evalSet.slice(0, 20).forEach(e => {
      const fo = DL.tfForward(m, e.idx), fr = freeRun(m, e.idx, e.tgt), fi = e.tgt.findIndex(t => t >= 0);
      e.tgt.forEach((t, i) => { if (t < 0) return; cnt++; if (TR.argmax(fo.logits[i]) === t) okTF++; if (fr[i - fi] === t) okFR++; });
    });
    TR.note(g, 0, fy + 68, `over 20 held-out examples: teacher-forced accuracy ${Math.round(100 * okTF / cnt)}%, free-running ${Math.round(100 * okFR / cnt)}%`, DC.ink, 10);
    El("tf-readout").innerHTML =
      `${sup.length} of ${n} positions carry a target; their mean loss is <b>${DL.fmt(meanL, 4)}</b>, every term from one forward pass through the causal mask. ` +
      `Over 20 held-out examples the model is right at <b>${Math.round(100 * okTF / cnt)}%</b> of supervised positions when teacher-forced and <b>${Math.round(100 * okFR / cnt)}%</b> when fed its own predictions` +
      (okFR < okTF ? ` — the gap is exposure bias: one wrong token becomes the context for the next.` : ` — no gap on this example set: the model's own outputs are the true tokens.`);
  }
  ["tf-t", "tf-s"].forEach(id => d3.select("#" + id).on("change", draw));
  d3.select("#tf-e").on("input", draw);
  draw();
})();

/* ═════════ 25 · #ls-svg — label smoothing.  ══════════════════════════════ */
(function () {
  const svg = d3.select("#ls-svg"); if (svg.empty()) return;
  const W = 760, H = 360;
  const El = id => document.getElementById(id);
  function draw() {
    const eps = +El("ls-e").value, V = +El("ls-V").value, gap = +El("ls-g").value;
    El("ls-ev").textContent = eps.toFixed(2); El("ls-gv").textContent = gap.toFixed(1);
    const q = DL.smoothTarget(V, 0, eps), floor = DL.smoothFloor(V, eps);
    const optGap = eps > 0 ? Math.log(1 + (1 - eps) * V / eps) : Infinity;
    /* loss as a function of the gap with all other logits equal: logits = [gap, 0, 0, …] */
    const lossAt = gp => { const lse = Math.log(Math.exp(gp) + (V - 1)); return (1 - eps + eps / V) * (lse - gp) + (V - 1) * (eps / V) * lse; };
    const gradAt = gp => { const p0 = Math.exp(gp) / (Math.exp(gp) + (V - 1)); return p0 - (1 - eps + eps / V); };
    const f = DL.frame(svg, W, H, { l: 44, r: 12, t: 22, b: 30 }), g = f.g;
    const pw = (f.iw - 60) / 3, ph = f.ih;
    /* 1: target */
    const g1 = g.append("g");
    TR.title(g1, 0, -8, `q_ε for V = ${DL.commas(V)}`);
    const y1 = d3.scaleLog().domain([Math.min(1e-7, q[1] / 2) || 1e-7, 1]).range([ph, 0]);
    DL.axisL(g1, y1, 4, "probability", v => Math.abs(Math.log10(v) % 1) < 1e-9 ? DL.fmtE(v, 0) : "");
    [["correct", q[0], DC.good], ["each other", q[1], DC.muted]].forEach((b, i) => {
      const yy = y1(Math.max(b[1], 1e-7));
      g1.append("rect").attr("x", 20 + i * 70).attr("y", yy).attr("width", 50).attr("height", ph - yy).attr("fill", b[2]).attr("fill-opacity", 0.85);
      TR.note(g1, 45 + i * 70, yy - 4, DL.sig(b[1], 3), DC.ink, 9.5, "middle"); TR.note(g1, 45 + i * 70, ph + 13, b[0], DC.muted, 9, "middle");
    });
    /* 2: floor against eps */
    const g2 = g.append("g").attr("transform", `translate(${pw + 30},0)`);
    TR.title(g2, 0, -8, "loss floor H(q_ε) against ε");
    const es = DL.linspace(0, 0.3, 61), fl = es.map(e => DL.smoothFloor(V, e));
    const x2 = d3.scaleLinear().domain([0, 0.3]).range([0, pw]), y2 = d3.scaleLinear().domain([0, Math.max(...fl) * 1.1]).range([ph, 0]);
    DL.gridY(g2, y2, pw, 4); DL.axisB(g2, x2, ph, 4, "ε"); DL.axisL(g2, y2, 4, "nats");
    DL.curve(g2, es.map((e, i) => [x2(e), y2(fl[i])]), { stroke: DC.violet, w: 2 });
    g2.append("circle").attr("cx", x2(eps)).attr("cy", y2(floor)).attr("r", 4).attr("fill", DC.lime);
    TR.note(g2, x2(eps) + 6, y2(floor) - 6, `floor ${DL.fmt(floor, 3)}`, DC.lime, 9.5);
    /* 3: loss against the gap */
    const g3 = g.append("g").attr("transform", `translate(${2 * (pw + 30)},0)`);
    TR.title(g3, 0, -8, "L_ε against the logit gap z_k − z_j");
    const gs = DL.linspace(0, 20, 81), ls = gs.map(lossAt);
    const x3 = d3.scaleLinear().domain([0, 20]).range([0, pw]), y3 = d3.scaleLinear().domain([0, Math.max(...ls) * 1.05]).range([ph, 0]);
    DL.gridY(g3, y3, pw, 4); DL.axisB(g3, x3, ph, 4, "gap"); DL.axisL(g3, y3, 4, "loss");
    DL.curve(g3, gs.map((gp, i) => [x3(gp), y3(ls[i])]), { stroke: DC.a2, w: 2 });
    if (isFinite(optGap) && optGap <= 20) { g3.append("line").attr("x1", x3(optGap)).attr("x2", x3(optGap)).attr("y1", 0).attr("y2", ph).attr("stroke", DC.good).attr("stroke-dasharray", "3 3"); TR.note(g3, x3(optGap) + 4, 12, `optimum ${DL.fmt(optGap, 2)}`, DC.good, 9.5); }
    else TR.note(g3, pw, 12, eps === 0 ? "optimum at +∞ (ε = 0)" : `optimum ${DL.fmt(optGap, 2)}, off scale`, DC.good, 9.5, "end");
    g3.append("circle").attr("cx", x3(gap)).attr("cy", y3(lossAt(gap))).attr("r", 4).attr("fill", DC.lime);
    TR.note(g3, x3(gap) + 6, y3(lossAt(gap)) + 14, `∂L/∂z_k = ${DL.fmt(gradAt(gap), 4)}`, DC.lime, 9.5);
    El("ls-readout").innerHTML =
      `ε = ${eps.toFixed(2)}, V = ${DL.commas(V)}: the target puts <b>${DL.sig(q[0], 4)}</b> on the correct token and ${DL.sig(q[1], 3)} on each of the other ${DL.commas(V - 1)}. ` +
      `Loss floor H(q_ε) = <b>${DL.fmt(floor, 4)}</b> nats${eps === 0 ? " (no smoothing: zero)" : ""}. Optimal logit gap <b>${isFinite(optGap) ? DL.fmt(optGap, 3) : "+∞"}</b>; at a gap of ${gap.toFixed(1)} the loss is ${DL.fmt(lossAt(gap), 4)} and ∂L/∂z_k = ${DL.fmt(gradAt(gap), 4)} ` +
      `(${gradAt(gap) < -1e-4 ? "still pushing the correct logit up" : gradAt(gap) > 1e-4 ? "pushing the correct logit DOWN — past the optimum" : "at the optimum"}).`;
  }
  d3.select("#ls-V").on("change", draw);
  ["ls-e", "ls-g"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 26 · #sc-svg — the schedules, from DL.lrAt and DL.noamLR.  ═══ */
(function () {
  const svg = d3.select("#sc-svg"); if (svg.empty()) return;
  const W = 760, H = 360;
  const El = id => document.getElementById(id);
  function draw() {
    const d = +El("sc-d").value, w = +El("sc-w").value, T = +El("sc-T").value, fl = +El("sc-f").value, logY = El("sc-log").checked;
    const peak = Math.pow(d * w, -0.5);
    const noam = t => DL.noamLR(t, d, w), cos = t => DL.lrAt("cosine", t, { peak: peak, warm: w, total: T, floor: fl * peak });
    const wsd = t => t < w ? peak * t / w : (t < 0.8 * T ? peak : peak * Math.max(0, (T - t) / (0.2 * T)) + fl * peak * Math.min(1, (t - 0.8 * T) / (0.2 * T)));
    const f = DL.frame(svg, W, H, { l: 60, r: 12, t: 22, b: 34 }), g = f.g;
    const lw = 440, lh = f.ih;
    TR.title(g, 0, -8, `learning rate against step, d = ${d}, warmup ${DL.commas(w)}, total ${DL.commas(T)}`);
    const K = 400, ts = DL.linspace(1, T, K).map(Math.round);
    const x = d3.scaleLinear().domain([0, T]).range([0, lw]);
    const y = logY ? d3.scaleLog().domain([peak * 1e-3, peak * 1.3]).range([lh, 0]).clamp(true) : d3.scaleLinear().domain([0, peak * 1.15]).range([lh, 0]);
    DL.gridY(g, y, lw, 4); DL.axisB(g, x, lh, 5, "step", v => DL.big(v)); DL.axisL(g, y, 4, "η", v => DL.fmtE(v, 0));
    const curves = [["2017: (d·warmup)^(−½), then step^(−½)", noam, DC.a2], ["warmup + cosine to the floor", cos, DC.accent], ["warmup – stable – decay (last 20%)", wsd, DC.good]];
    curves.forEach(c => DL.curve(g, ts.map(t => [x(t), y(Math.max(c[1](t), peak * 1e-3))]), { stroke: c[2], w: 1.8 }));
    DL.legend(g, curves.map(c => ({ label: c[0], color: c[2] })), lw - 250, lh - 46, { gap: 12, font: 9 });
    /* right: the warmup zoom and the equivalence check */
    const rx = lw + 40, rw = f.iw - rx;
    const gr = g.append("g").attr("transform", `translate(${rx},0)`);
    TR.title(gr, 0, -8, "the first 2 × warmup steps");
    const ts2 = DL.linspace(1, 2 * w, 200).map(Math.round), x2 = d3.scaleLinear().domain([0, 2 * w]).range([0, rw]), y2 = d3.scaleLinear().domain([0, peak * 1.15]).range([lh, 0]);
    DL.axisB(gr, x2, lh, 3, "step", v => DL.big(v)); DL.axisL(gr, y2, 3, "", v => DL.fmtE(v, 0));
    curves.forEach(c => DL.curve(gr, ts2.map(t => [x2(t), y2(c[1](t))]), { stroke: c[2], w: 1.6 }));
    let worst = 0; for (let t = 1; t <= 2 * w; t += 7) worst = Math.max(worst, Math.abs(noam(t) - DL.lrAt("invsqrt", t, { peak: peak, warm: w })) / noam(t));
    TR.note(gr, 0, lh + 30, `noamLR vs lrAt("invsqrt"): max rel. diff ${DL.fmtE(worst, 1)}`, DC.good, 9);
    const meanOf = fn => { let s = 0; for (let t = 1; t <= T; t += Math.max(1, Math.floor(T / 2000))) s += fn(t); return s / Math.ceil(T / Math.max(1, Math.floor(T / 2000))); };
    El("sc-readout").innerHTML =
      `Peak of the 2017 rule: (d·warmup)^(−½) = (${d}·${DL.commas(w)})^(−½) = <b>${DL.fmtE(peak, 2)}</b>, reached at step ${DL.commas(w)}. At the last step: 2017 rule ${DL.fmtE(noam(T), 2)} (= peak × √(warmup/T) = ${DL.fmt(Math.sqrt(w / T), 3)} × peak), cosine ${DL.fmtE(cos(T), 2)}, warmup–stable–decay ${DL.fmtE(wsd(T), 2)}. ` +
      `Mean rate over the run: 2017 <b>${DL.fmtE(meanOf(noam), 2)}</b>, cosine <b>${DL.fmtE(meanOf(cos), 2)}</b>, stable–decay <b>${DL.fmtE(meanOf(wsd), 2)}</b> — the cosine schedule delivers ${DL.fmt(meanOf(cos) / meanOf(noam), 2)}× the 2017 rule's total step at the same peak.`;
  }
  ["sc-d", "sc-w", "sc-T", "sc-f", "sc-log"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 27 · #ad-svg — β₂ under a spike, through DL.OPT.adam.  ════════ */
(function () {
  const svg = d3.select("#ad-svg"); if (svg.empty()) return;
  const W = 760, H = 360, N = 5000, SPIKE = 3000;
  const El = id => document.getElementById(id);
  function draw() {
    const spike = +El("ad-s").value, b2 = +El("ad-b").value, clipMul = +El("ad-c").value, seed = +El("ad-seed").value;
    El("ad-sv").textContent = spike; El("ad-seedv").textContent = seed;
    const r = DL.rng(seed * 13), typ = 0.02;
    const gs = []; for (let t = 0; t < N; t++) gs.push((t === SPIKE ? spike : 1) * (typ + 0.3 * typ * DL.randn(r)));
    const clipped = clipMul > 0 ? gs.map(v => DL.clipNorm([v], clipMul * typ).g[0]) : gs;
    const run = beta2 => { const st = DL.OPT.adam.init(1); return clipped.map(v => -DL.OPT.adam.step(st, [v], { lr: 1, beta1: 0.9, beta2: beta2 })[0]); };
    const A = run(b2), B = run(0.95);
    const f = DL.frame(svg, W, H, { l: 50, r: 12, t: 22, b: 30 }), g = f.g;
    const uh = 100, x = d3.scaleLinear().domain([0, N]).range([0, f.iw]);
    TR.title(g, 0, -8, `gradient stream: typical ${typ}, spike ×${spike} at step ${SPIKE}${clipMul ? ", clipped at " + clipMul + "× typical" : ""}`);
    const yg = d3.scaleLog().domain([typ / 10, typ * 120]).range([uh, 0]).clamp(true);
    DL.axisL(g, yg, 3, "g", v => DL.fmtE(v, 0));
    DL.curve(g, gs.map((v, t) => [x(t), yg(Math.max(v, typ / 10))]), { stroke: DC.muted, w: 1 });
    if (clipMul) DL.curve(g, clipped.map((v, t) => [x(t), yg(Math.max(v, typ / 10))]), { stroke: DC.rose, w: 1.4 });
    const ly = uh + 40, lh = f.ih - ly;
    const gl = g.append("g").attr("transform", `translate(0,${ly})`);
    TR.title(gl, 0, -8, "Adam step |m̂/(√v̂ + ε)| in units of the learning rate");
    const ys = d3.scaleLinear().domain([0, Math.max(1.5, ...A, ...B) * 1.1]).range([lh, 0]);
    DL.gridY(gl, ys, f.iw, 4); DL.axisB(gl, x, lh, 6, "step", v => DL.big(v)); DL.axisL(gl, ys, 4);
    gl.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", ys(1)).attr("y2", ys(1)).attr("stroke", DC.line);
    DL.curve(gl, A.map((v, t) => [x(t), ys(v)]), { stroke: DC.a2, w: 1.6 });
    DL.curve(gl, B.map((v, t) => [x(t), ys(v)]), { stroke: DC.good, w: 1.6 });
    DL.legend(gl, [{ label: `β₂ = ${b2}`, color: DC.a2 }, { label: "β₂ = 0.95", color: DC.good }], f.iw - 100, 8, { gap: 12, font: 9.5 });
    const pre = arr => DL.mean(arr.slice(SPIKE - 60, SPIKE - 1));
    const recover = arr => { const base = pre(arr); for (let t = SPIKE + 1; t < N - 20; t++) { let ok = true; for (let u = t; u < t + 20; u++) if (Math.abs(arr[u] - base) / base > 0.15) { ok = false; break; } if (ok) return t - SPIKE; } return N - SPIKE; };
    El("ad-readout").innerHTML =
      `Step size at the spike: β₂ = ${b2} <b>${DL.fmt(A[SPIKE], 2)}</b>× (from ${DL.fmt(pre(A), 2)} before), β₂ = 0.95 <b>${DL.fmt(B[SPIKE], 2)}</b>× (from ${DL.fmt(pre(B), 2)}). ` +
      `Ten steps later: ${DL.fmt(A[SPIKE + 10], 2)} and ${DL.fmt(B[SPIKE + 10], 2)}; a hundred steps later: ${DL.fmt(A[SPIKE + 100], 2)} and ${DL.fmt(B[SPIKE + 100], 2)}. ` +
      `Back within 15% of the pre-spike step for 20 consecutive steps: β₂ = ${b2} after <b>${recover(A)}</b> steps, β₂ = 0.95 after <b>${recover(B)}</b>` +
      (clipMul ? `. Clipping capped the spike at ${DL.fmt(clipMul * typ, 3)} before the optimiser saw it.` : `.`);
  }
  ["ad-b", "ad-c"].forEach(id => d3.select("#" + id).on("change", draw));
  ["ad-s", "ad-seed"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 28 · #mp-svg — fp16 against bf16 on a gradient histogram.  ═══ */
(function () {
  const svg = d3.select("#mp-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const El = id => document.getElementById(id);
  const FP16 = { minSub: Math.pow(2, -24), minNorm: Math.pow(2, -14), max: 65504 }, BF16 = { minSub: Math.pow(2, -133), minNorm: Math.pow(2, -126), max: (2 - Math.pow(2, -7)) * Math.pow(2, 127) };
  function draw() {
    const med = +El("mp-m").value, spread = +El("mp-s").value, k = +El("mp-k").value, val = +El("mp-v").value;
    El("mp-sv").textContent = spread.toFixed(1); El("mp-kv").textContent = k;
    const scale = Math.pow(2, k), r = DL.rng(5), M = 4000, vals = [];
    for (let i = 0; i < M; i++) vals.push(med * Math.pow(10, spread / 2 * DL.randn(r)) * scale);
    const frac = (arr, fmt) => { let u = 0, o = 0; arr.forEach(v => { const q = fmt(v); if (q === 0) u++; else if (!isFinite(q)) o++; }); return { under: u / arr.length, over: o / arr.length }; };
    const f16 = frac(vals, TR.fp16), b16 = frac(vals, TR.bf16);
    const f = DL.frame(svg, W, H, { l: 44, r: 12, t: 22, b: 34 }), g = f.g;
    const lw = 480, lh = f.ih - 30;
    TR.title(g, 0, -8, `|gradient| histogram, log-normal (median ${DL.fmtE(med, 0)}, ${spread} decades wide)${k ? ", × loss scale 2^" + k : ""}`);
    const x = d3.scaleLog().domain([1e-12, 1e6]).range([0, lw]);
    const nb = 54, edges = DL.linspace(-12, 6, nb + 1), bins = new Array(nb).fill(0);
    vals.forEach(v => { const b = Math.floor((Math.log10(v) + 12) / 18 * nb); if (b >= 0 && b < nb) bins[b]++; });
    const y = d3.scaleLinear().domain([0, Math.max(...bins) * 1.1]).range([lh, 0]);
    /* fp16 range shading */
    g.append("rect").attr("x", x(FP16.minNorm)).attr("y", 0).attr("width", x(FP16.max) - x(FP16.minNorm)).attr("height", lh).attr("fill", DC.accent).attr("fill-opacity", 0.1);
    g.append("rect").attr("x", x(FP16.minSub)).attr("y", 0).attr("width", x(FP16.minNorm) - x(FP16.minSub)).attr("height", lh).attr("fill", DC.accent).attr("fill-opacity", 0.05);
    DL.axisB(g, x, lh, 6, "|g|", v => Math.abs(Math.log10(v) % 1) < 1e-9 ? DL.fmtE(v, 0) : ""); DL.axisL(g, y, 3, "count");
    bins.forEach((c, i) => { const lo = Math.pow(10, edges[i]), hi = Math.pow(10, edges[i + 1]); const mid = Math.sqrt(lo * hi); const col = mid < FP16.minSub ? DC.bad : (mid > FP16.max ? DC.rose : (mid < FP16.minNorm ? DC.a2 : DC.accent)); g.append("rect").attr("x", x(lo)).attr("y", y(c)).attr("width", Math.max(0.5, x(hi) - x(lo) - 0.6)).attr("height", lh - y(c)).attr("fill", col).attr("fill-opacity", 0.85); });
    [[FP16.minSub, "fp16 min subnormal 6.0e−8"], [FP16.minNorm, "fp16 min normal 6.1e−5"], [FP16.max, "fp16 max 65 504"]].forEach(([v, lab], i) => { g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", 0).attr("y2", lh).attr("stroke", DC.ink).attr("stroke-dasharray", "3 3"); TR.note(g, x(v) + 3, 12 + i * 12, lab, DC.ink, 8.5); });
    TR.note(g, 0, lh + 28, `bf16 range: 9.2e−41 … 3.4e38 — the whole axis and beyond`, DC.good, 9.5);
    DL.legend(g, [{ label: "fp16 normal", color: DC.accent }, { label: "fp16 subnormal (coarse)", color: DC.a2 }, { label: "fp16 underflow → 0", color: DC.bad }, { label: "fp16 overflow → ∞", color: DC.rose }], lw - 150, lh - 60, { gap: 12, font: 9 });
    /* right: one value rounded */
    const rx = lw + 30, gr = g.append("g").attr("transform", `translate(${rx},0)`);
    TR.title(gr, 0, -8, "one value, three formats");
    const kv = DL.kv(gr, 0, 14, { lead: 15, keyW: 48, size: 9.5 });
    const f32 = Math.fround(val), h16 = TR.fp16(val), hb = TR.bf16(val);
    kv("value", DL.sig(val, 7)); kv("fp32", DL.sig(f32, 8), DC.muted); kv("", "rel " + DL.fmtE(Math.abs(f32 - val) / val, 1), DC.muted);
    kv("fp16", isFinite(h16) ? DL.sig(h16, 8) : "overflow", DC.accent); kv("", isFinite(h16) ? "rel " + DL.fmtE(Math.abs(h16 - val) / val, 1) : "—", DC.accent);
    kv("bf16", DL.sig(hb, 8), DC.good); kv("", "rel " + DL.fmtE(Math.abs(hb - val) / val, 1), DC.good);
    kv("ratio", isFinite(h16) && h16 !== val ? DL.fmt(Math.abs(hb - val) / Math.abs(h16 - val), 1) + "× (bf16/fp16)" : "—", DC.ink);
    TR.note(gr, 0, 14 + 8 * 15 + 6, "bf16's mantissa is 3 bits shorter: its steps", DC.muted, 9);
    TR.note(gr, 0, 14 + 8 * 15 + 18, "are 8× coarser; the error on ONE value varies", DC.muted, 9);
    El("mp-readout").innerHTML =
      `Of ${M} gradients (median ${DL.fmtE(med * scale, 1)} after the loss scale): fp16 sends <b>${TR.pct(f16.under)}</b> to exactly zero and <b>${TR.pct(f16.over)}</b> to infinity; bf16 <b>${TR.pct(b16.under)}</b> and <b>${TR.pct(b16.over)}</b>. ` +
      (k === 0 ? `Raise the loss scale to move the histogram out of fp16's underflow. ` : `With 2^${k} the histogram sits ${k} binary orders higher. `) +
      `Rounding ${DL.sig(val, 6)}: fp16 error ${isFinite(h16) ? DL.fmtE(Math.abs(h16 - val) / val, 1) : "overflow"}, bf16 error ${DL.fmtE(Math.abs(hb - val) / val, 1)}, fp32 error ${DL.fmtE(Math.abs(f32 - val) / val, 1)}.`;
  }
  ["mp-m", "mp-v"].forEach(id => d3.select("#" + id).on("change", draw));
  ["mp-s", "mp-k"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 29 · #sp-svg — temperature, top-k, top-p on one distribution. ═ */
(function () {
  const svg = d3.select("#sp-svg"); if (svg.empty()) return;
  const W = 760, H = 360, V = 50;
  const El = id => document.getElementById(id);
  function draw() {
    const T = +El("sp-T").value, k = +El("sp-k").value, p = +El("sp-p").value, z = +El("sp-z").value;
    El("sp-Tv").textContent = T.toFixed(1); El("sp-kv").textContent = k ? String(k) : "off"; El("sp-pv").textContent = p.toFixed(2);
    /* Zipf-like logits: p_r ∝ r^(−z), then a little seeded jitter */
    const r = DL.rng(9), logits = d3.range(V).map(i => -z * Math.log(i + 1) + 0.15 * DL.randn(r));
    const p0 = DL.softmax(logits), pT = TR.truncate(logits, { T: T }), pK = TR.truncate(logits, { T: T, k: k }), pP = TR.truncate(logits, { T: T, k: k, p: p });
    const f = DL.frame(svg, W, H, { l: 40, r: 10, t: 22, b: 30 }), g = f.g;
    const pw = (f.iw - 60) / 4, ph = f.ih - 30;
    const panels = [["model p (T = 1)", p0, DC.muted], [`after T = ${T.toFixed(1)}`, pT, DC.accent], [`after top-k${k ? " = " + k : " (off)"}`, pK, DC.a2], [`after top-p = ${p.toFixed(2)}`, pP, DC.good]];
    const ymax = Math.max(...panels.flatMap(pp => pp[1])) * 1.1;
    panels.forEach((pp, i) => {
      const gg = g.append("g").attr("transform", `translate(${i * (pw + 20)},0)`);
      TR.title(gg, 0, -8, pp[0], pp[2]);
      const x = d3.scaleBand().domain(d3.range(V)).range([0, pw]).paddingInner(0.15), y = d3.scaleLinear().domain([0, ymax]).range([ph, 0]);
      if (i === 0) DL.axisL(gg, y, 4, "probability");
      DL.axisB(gg, d3.scaleLinear().domain([1, V]).range([0, pw]), ph, 3, "rank");
      pp[1].forEach((v, j) => { if (v > 0) gg.append("rect").attr("x", x(j)).attr("y", y(v)).attr("width", x.bandwidth()).attr("height", ph - y(v)).attr("fill", pp[2]).attr("fill-opacity", 0.85); });
      const H = DL.entropyOf(pp[1]), alive = pp[1].filter(v => v > 0).length;
      TR.note(gg, pw, 12, `H = ${DL.fmt(H, 2)} nats`, DC.ink, 9.5, "end"); TR.note(gg, pw, 24, `e^H = ${DL.fmt(Math.exp(H), 1)} candidates`, DC.ink, 9.5, "end"); TR.note(gg, pw, 36, `${alive} tokens with p > 0`, DC.muted, 9, "end");
    });
    const Hs = panels.map(pp => DL.entropyOf(pp[1])), alive = panels.map(pp => pp[1].filter(v => v > 0).length);
    El("sp-readout").innerHTML =
      `Zipf exponent ${z}, V = ${V}. Entropy: model <b>${DL.fmt(Hs[0], 3)}</b> → temperature ${T.toFixed(1)}: <b>${DL.fmt(Hs[1], 3)}</b> → top-k: <b>${DL.fmt(Hs[2], 3)}</b> → top-p: <b>${DL.fmt(Hs[3], 3)}</b> nats (effective candidates ${panels.map(pp => DL.fmt(Math.exp(DL.entropyOf(pp[1])), 1)).join(" → ")}). ` +
      `Tokens with non-zero probability: ${alive.join(" → ")}. Temperature ${T < 1 ? "lightened" : T > 1 ? "thickened" : "left"} the tail but ${alive[1] === V ? "removed nothing" : "removed some"}; top-p kept ${alive[3]} token${alive[3] === 1 ? "" : "s"} and gives the mode <b>${DL.fmt(Math.max(...pP), 3)}</b>.`;
  }
  d3.select("#sp-z").on("change", draw);
  ["sp-T", "sp-k", "sp-p"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 30 · #dc-svg — greedy, beam and sampling, live.  ══════════════ */
(function () {
  const svg = d3.select("#dc-svg"); if (svg.empty()) return;
  const W = 760, H = 420;
  const El = id => document.getElementById(id);
  function draw() {
    const taskName = El("dc-t").value, steps = +El("dc-s").value, ei = +El("dc-e").value, B = +El("dc-B").value, T = +El("dc-T").value, seed = +El("dc-seed").value;
    El("dc-ev").textContent = ei; El("dc-Tv").textContent = T.toFixed(1); El("dc-seedv").textContent = seed;
    const R = TR.train({ task: taskName, steps: steps, seed: 1, cfg: { pos: "learned" } }), m = R.m, V = m.cfg.V, n = m.cfg.nCtx;
    const ex = R.evalSet[ei], first = ex.tgt.findIndex(t => t >= 0), last = ex.tgt.length - 1 - ex.tgt.slice().reverse().findIndex(t => t >= 0);
    const truth = ex.tgt.slice(first, last + 1);
    /* greedy, with the distributions kept */
    const greedy = [], dists = []; let seq = ex.idx.slice(0, first + 1);
    for (let i = first; i <= last; i++) { const f = DL.tfForward(m, seq), pr = DL.softmax(f.logits[i]); dists.push(pr); const a = TR.argmax(pr); greedy.push(a); seq.push(a); }
    const beam = TR.beamSearch(m, ex.idx, ex.tgt, B);
    const r = DL.rng(seed * 77 + 1), samp = []; seq = ex.idx.slice(0, first + 1);
    for (let i = first; i <= last; i++) { const f = DL.tfForward(m, seq), pr = TR.truncate(f.logits[i], { T: T }); const a = TR.sampleFrom(pr, r); samp.push(a); seq.push(a); }
    const lp = toks => TR.seqLogProb(m, ex.idx, toks, first);
    const rows = [["greedy", greedy, DC.accent], [`beam (B = ${B})`, beam.best.toks, DC.a2], [`sample (T = ${T.toFixed(1)})`, samp, DC.violet]];
    const f = DL.frame(svg, W, H, { l: 100, r: 10, t: 22, b: 10 }), g = f.g;
    const cw = 30;
    const cellRow = (y, lab, toks, color, offset, truthArr) => {
      TR.note(g, -6, y + 15, lab, DC.muted, 9.5, "end");
      toks.forEach((t, k) => {
        const ok = truthArr ? t === truthArr[k] : true;
        g.append("rect").attr("x", (offset + k) * cw).attr("y", y).attr("width", cw - 3).attr("height", 22).attr("rx", 3).attr("fill", truthArr ? (ok ? DC.good : DC.bad) : color).attr("fill-opacity", 0.75);
        TR.note(g, (offset + k) * cw + (cw - 3) / 2, y + 15, TR.tokenLabel(t, V), DC.bg, 9.5, "middle");
      });
    };
    TR.title(g, 0, -8, `task "${taskName}", example ${ei}, model trained ${steps} steps`);
    cellRow(0, "prompt", ex.idx.slice(0, first + 1), DC.muted, 0, null);
    cellRow(26, "true answer", truth, DC.ink, first + 1, null);
    rows.forEach((rw, k) => { cellRow(60 + k * 26, rw[0], rw[1], rw[2], first + 1, truth); TR.note(g, (first + 1 + truth.length) * cw + 6, 60 + k * 26 + 15, `log p = ${DL.fmt(lp(rw[1]), 3)}`, DC.ink, 9); });
    TR.note(g, (first + 1 + truth.length) * cw + 6, 26 + 15, `log p = ${DL.fmt(lp(truth), 3)}`, DC.muted, 9);
    /* distributions per greedy step */
    const dy = 150, ch = 12, cwd = 22;
    TR.title(g, 0, dy - 6, "the distribution at each greedy step (rows), over the vocabulary (columns)");
    const scl = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);
    TR.heat(g, dists, 0, dy, cwd, scl, { ch: ch, stroke: DC.bg });
    dists.forEach((pr, k) => { const a = greedy[k]; g.append("rect").attr("x", a * cwd).attr("y", dy + k * ch).attr("width", cwd).attr("height", ch).attr("fill", "none").attr("stroke", DC.lime).attr("stroke-width", 1.2); TR.note(g, -6, dy + k * ch + 9, `step ${k + 1}`, DC.muted, 8.5, "end"); });
    for (let t = 0; t < V; t++) TR.note(g, t * cwd + cwd / 2, dy + dists.length * ch + 11, TR.tokenLabel(t, V), DC.muted, 8.5, "middle");
    /* beam history */
    const bx = 300, by = dy;
    const gb = g.append("g").attr("transform", `translate(${bx},${by})`);
    TR.title(gb, 0, -6, `beam hypotheses after each step (top ${Math.min(6, Math.max(B, 6))} shown, kept ones bright)`);
    beam.history.forEach((hs, k) => {
      const yy = k * 15;
      TR.note(gb, 0, yy + 10, `step ${k + 1}:`, DC.muted, 8.5);
      hs.forEach((h, j) => TR.note(gb, 48 + j * 68, yy + 10, `${h.toks.map(t => TR.tokenLabel(t, V)).join("")} ${DL.fmt(h.score, 2)}`, h.kept ? DC.a2 : DC.muted, 8.5));
    });
    const right = toks => toks.filter((t, k) => t === truth[k]).length;
    El("dc-readout").innerHTML =
      `Greedy: <b>${greedy.map(t => TR.tokenLabel(t, V)).join(" ")}</b> (log p ${DL.fmt(lp(greedy), 3)}, ${right(greedy)}/${truth.length} right). Beam B = ${B}: <b>${beam.best.toks.map(t => TR.tokenLabel(t, V)).join(" ")}</b> (log p ${DL.fmt(beam.best.score, 3)}, ${right(beam.best.toks)}/${truth.length}). Sample at T = ${T.toFixed(1)}, seed ${seed}: <b>${samp.map(t => TR.tokenLabel(t, V)).join(" ")}</b> (log p ${DL.fmt(lp(samp), 3)}, ${right(samp)}/${truth.length}). ` +
      `The true answer has log p ${DL.fmt(lp(truth), 3)}. ` + (beam.best.score > lp(greedy) + 1e-9 ? `The beam found a higher-scoring sequence than greedy.` : `Beam and greedy score the same: the distributions are peaked enough that the greedy path is the beam's best.`);
  }
  ["dc-t", "dc-s", "dc-B"].forEach(id => d3.select("#" + id).on("change", draw));
  ["dc-e", "dc-T", "dc-seed"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 31 · #dw-svg — depth against width at a fixed budget.  ════════ */
(function () {
  const svg = d3.select("#dw-svg"); if (svg.empty()) return;
  const W = 760, H = 400;
  const El = id => document.getElementById(id);
  function draw() {
    const ci = +El("dw-c").value, taskName = El("dw-t").value, steps = +El("dw-s").value;
    const f = DL.frame(svg, W, H, { l: 50, r: 10, t: 22, b: 34 }), g = f.g;
    const lw = 360, lh = f.ih;
    TR.title(g, 0, -8, "d = √(N_blocks / 12L): one curve per budget, real shapes as points");
    const x = d3.scaleLog().domain([4, 256]).range([0, lw]), y = d3.scaleLog().domain([256, 32768]).range([lh, 0]);
    DL.gridY(g, y, lw, 4); DL.axisB(g, x, lh, 5, "depth L", v => Number.isInteger(Math.log2(v)) ? String(v) : ""); DL.axisL(g, y, 4, "width d", v => Number.isInteger(Math.log2(v)) ? DL.big(v) : "");
    /* the aspect-ratio band 40 … 130 */
    const band = [[4, 40 * 4], [256, 40 * 256]], band2 = [[4, 130 * 4], [256, 130 * 256]];
    const clampY = v => Math.max(256, Math.min(32768, v));
    g.append("path").attr("d", `M${x(4)},${y(clampY(band[0][1]))} L${x(256)},${y(clampY(band[1][1]))} L${x(256)},${y(clampY(band2[1][1]))} L${x(4)},${y(clampY(band2[0][1]))} Z`).attr("fill", DC.a2).attr("fill-opacity", 0.08);
    TR.note(g, x(5), y(clampY(5 * 130)) - 4, "d/L = 130", DC.a2, 9); TR.note(g, x(60), y(60 * 40) + 12, "d/L = 40", DC.a2, 9);
    TR.CFGS.forEach((c, i) => {
      const N = DL.modelParams(c).blocks, Ls = DL.linspace(Math.log(4), Math.log(256), 60).map(Math.exp);
      DL.curve(g, Ls.map(L => [x(L), y(Math.max(256, Math.min(32768, Math.sqrt(N / (12 * L)))))]), { stroke: i === ci ? DC.accent : DC.line, w: i === ci ? 2.2 : 1.2 });
      g.append("circle").attr("cx", x(c.L)).attr("cy", y(c.d)).attr("r", i === ci ? 5 : 3.5).attr("fill", i === ci ? DC.accent : DC.muted);
      TR.note(g, x(c.L) + 7, y(c.d) + 4, `${c.name} d/L = ${Math.round(c.d / c.L)}`, i === ci ? DC.ink : DC.muted, 9);
    });
    /* right: the live triple */
    const rx = lw + 44, rw = f.iw - rx;
    const gr = g.append("g").attr("transform", `translate(${rx},0)`);
    const shapes = [[1, 32, "1 block, d = 32"], [2, 22, "2 blocks, d = 22"], [4, 16, "4 blocks, d = 16"]];
    const runs = shapes.map(([L, d, lab]) => { const R = TR.train({ task: taskName, steps: steps, B: 8, seed: 2, cfg: { d: d, L: L, h: 2, dff: 2 * d, pos: "learned" } }); let cnt = 0; DL.tfParamList(R.m).forEach(p => cnt += Array.isArray(p.M[0]) ? p.M.length * p.M[0].length : p.M.length); return { L: L, d: d, lab: lab, R: R, params: cnt }; });
    TR.title(gr, 0, -8, `three tiny decoders of ≈ equal size, task "${taskName}"`);
    const allv = runs.flatMap(r => r.R.curve.map(c => c[1]));
    const xs = d3.scaleLinear().domain([0, steps]).range([0, rw]), ys = d3.scaleLog().domain([Math.max(1e-3, Math.min(...allv) / 2), Math.max(...allv) * 1.5]).range([lh, 0]).clamp(true);
    DL.gridY(gr, ys, rw, 4); DL.axisB(gr, xs, lh, 4, "step"); DL.axisL(gr, ys, 4, "loss", v => Math.abs(Math.log10(v) % 1) < 1e-9 ? DL.fmtE(v, 0) : "");
    const cols = [DC.a2, DC.accent, DC.good];
    runs.forEach((r, i) => DL.curve(gr, r.R.curve.map(c => [xs(c[0]), ys(Math.max(c[1], 1e-3))]), { stroke: cols[i], w: 1.8 }));
    DL.legend(gr, runs.map((r, i) => ({ label: `${r.lab} · ${DL.commas(r.params)} params · held-out ${DL.fmtE(r.R.evalLoss, 1)}`, color: cols[i] })), 4, 10, { gap: 12, font: 9 });
    const c = TR.CFGS[ci], best = runs.reduce((a, b) => b.R.evalLoss < a.R.evalLoss ? b : a, runs[0]);
    const spread = Math.max(...runs.map(r => r.R.evalLoss)) / Math.max(1e-6, Math.min(...runs.map(r => r.R.evalLoss)));
    El("dw-readout").innerHTML =
      `${c.name}: d = ${c.d}, L = ${c.L}, d/L = <b>${DL.fmt(c.d / c.L, 1)}</b>, N_blocks = ${DL.big(DL.modelParams(c).blocks)}; on its curve, halving L to ${c.L / 2} would need d = ${DL.commas(Math.round(Math.sqrt(DL.modelParams(c).blocks / (12 * c.L / 2))))}. ` +
      `Live triple (${steps} steps, seed 2): ` + runs.map(r => `${r.lab} → ${DL.commas(r.params)} parameters, held-out loss <b>${DL.fmtE(r.R.evalLoss, 2)}</b>`).join("; ") +
      `. Best: ${best.lab}. ` + (spread > 3 ? `The spread is a factor of ${DL.fmt(spread, 1)} — at this scale the result is the depth floor (whether the task's circuit fits in L blocks), not the fixed-budget plateau.` : `The three are within a factor of ${DL.fmt(spread, 1)} of each other — the toy-scale shadow of the plateau, on a task every shape can represent.`);
  }
  ["dw-c", "dw-t", "dw-s"].forEach(id => d3.select("#" + id).on("change", draw));
  draw();
})();

/* ═════════ 32 · #lv-svg — the signature: a decoder trained live.  ════════ */
(function () {
  const svg = d3.select("#lv-svg"); if (svg.empty()) return;
  const W = 760, H = 460;
  const El = id => document.getElementById(id);
  function draw() {
    const taskName = El("lv-t").value, steps = +El("lv-s").value, seed = +El("lv-seed").value, ei = +El("lv-e").value;
    El("lv-seedv").textContent = seed; El("lv-ev").textContent = ei;
    const R = TR.train({ task: taskName, steps: steps, seed: seed, cfg: { pos: "learned" } }), m = R.m, V = m.cfg.V, n = m.cfg.nCtx, L = m.cfg.L, h = m.cfg.h;
    let params = 0; DL.tfParamList(m).forEach(p => params += Array.isArray(p.M[0]) ? p.M.length * p.M[0].length : p.M.length);
    /* mean attention per head over the held-out set */
    const maps = []; for (let l = 0; l < L; l++) { maps.push([]); for (let j = 0; j < h; j++) maps[l].push(DL.zeros2(n, n)); }
    R.evalSet.forEach(ex => { const f = DL.tfForward(m, ex.idx); for (let l = 0; l < L; l++) for (let j = 0; j < h; j++) { const A = f.st.blocks[l].a.per[j].A; for (let i = 0; i < n; i++) for (let k = 0; k <= i; k++) maps[l][j][i][k] += A[i][k] / R.evalSet.length; } });
    const offsetOf = A => { const s = new Array(n).fill(0), c = new Array(n).fill(0); for (let i = 0; i < n; i++) for (let k = 0; k <= i; k++) { s[i - k] += A[i][k]; c[i - k]++; } const mean = s.map((v, k) => c[k] ? v / c[k] : 0); return { k: mean.indexOf(Math.max(...mean)), w: Math.max(...mean) }; };
    const f = DL.frame(svg, W, H, { l: 44, r: 10, t: 22, b: 10 }), g = f.g;
    const lw = 300, lh = 200;
    TR.title(g, 0, -8, `training loss, task "${taskName}", ${steps} steps, seed ${seed}`);
    const x = d3.scaleLinear().domain([0, steps]).range([0, lw]), allv = R.curve.map(c => c[1]);
    const y = d3.scaleLog().domain([Math.max(1e-3, Math.min(...allv) / 2), Math.max(...allv, Math.log(V - 2)) * 1.5]).range([lh, 0]).clamp(true);
    DL.gridY(g, y, lw, 4); DL.axisB(g, x, lh, 5, "step"); DL.axisL(g, y, 4, "loss", v => Math.abs(Math.log10(v) % 1) < 1e-9 ? DL.fmtE(v, 0) : "");
    DL.curve(g, R.curve.map(c => [x(c[0]), y(Math.max(c[1], 1e-3))]), { stroke: DC.accent, w: 2 });
    const gmax = Math.max(...R.gnorms.map(c => c[1])), yg = d3.scaleLinear().domain([0, gmax * 1.1]).range([lh, 0]);
    DL.curve(g, R.gnorms.map(c => [x(c[0]), yg(c[1])]), { stroke: DC.muted, w: 1, op: 0.7 });
    g.append("line").attr("x1", 0).attr("x2", lw).attr("y1", y(Math.log(V - 2))).attr("y2", y(Math.log(V - 2))).attr("stroke", DC.bad).attr("stroke-dasharray", "3 3");
    TR.note(g, lw, y(Math.log(V - 2)) - 4, `chance ln ${V - 2} = ${DL.fmt(Math.log(V - 2), 3)}`, DC.bad, 9, "end");
    DL.legend(g, [{ label: "training loss", color: DC.accent }, { label: `‖gradient‖ (max ${DL.fmt(gmax, 2)}, own scale)`, color: DC.muted }], 4, 10, { gap: 12, font: 9 });
    /* attention maps */
    const ax = lw + 44, cw = 11, offs = [];
    TR.title(g, ax, -8, "mean attention per head, held-out examples");
    for (let l = 0; l < L; l++) for (let j = 0; j < h; j++) {
      const A = maps[l][j], px = ax + j * (n * cw + 16), py = l * (n * cw + 30);
      TR.heat(g, A, px, py, cw, d3.scaleSequential(d3.interpolateViridis).domain([0, Math.max(...A.flat())]), { stroke: DC.bg });
      const o = offsetOf(A); offs.push({ l: l, j: j, k: o.k, w: o.w });
      TR.note(g, px, py + n * cw + 11, `block ${l + 1}, head ${j}: peak offset −${o.k} (${DL.fmt(o.w, 2)})`, DC.ink, 8.5);
    }
    /* example */
    const ex = R.evalSet[ei], fo = DL.tfForward(m, ex.idx), pred = ex.tgt.map((t, i) => t < 0 ? null : TR.argmax(fo.logits[i]));
    const ey = lh + 40, cc = 26;
    const rowCells = (yy, lab, vals, color, cmp) => {
      TR.note(g, -6, yy + 14, lab, DC.muted, 9, "end");
      vals.forEach((v, i) => { const on = v !== null && v >= 0; const ok = cmp ? v === ex.tgt[i] : true; g.append("rect").attr("x", i * cc).attr("y", yy).attr("width", cc - 3).attr("height", 20).attr("rx", 3).attr("fill", on ? (cmp ? (ok ? DC.good : DC.bad) : color) : DC.grid).attr("fill-opacity", 0.8); if (on) TR.note(g, i * cc + (cc - 3) / 2, yy + 14, TR.tokenLabel(v, V), DC.bg, 9, "middle"); });
    };
    TR.title(g, 0, ey - 6, `held-out example ${ei}`);
    rowCells(ey, "input", ex.idx, DC.accent, false); rowCells(ey + 24, "target", ex.tgt, DC.a2, false); rowCells(ey + 48, "predicted", pred, null, true);
    El("lv-readout").innerHTML =
      `${DL.commas(params)} parameters, ${steps} AdamW steps of batch 8, seed ${seed}, task "${TR.TASKS[taskName]}". Final training loss <b>${DL.fmtE(R.trainLoss, 2)}</b>; held-out loss <b>${DL.fmtE(R.evalLoss, 2)}</b>, accuracy <b>${Math.round(100 * R.evalAcc)}%</b> over ${R.evalSet.length} examples (chance loss ln 8 = 2.079). ` +
      `Heads by peak mean offset: ` + offs.map(o => `block ${o.l + 1} head ${o.j} → −${o.k} (weight ${DL.fmt(o.w, 2)})`).join(", ") + `.` +
      (taskName === "copy" && offs.some(o => o.k === 4 && o.w > 0.4) ? ` A head at offset −4 is the copy circuit: the token to predict is the one four back.` : "");
  }
  ["lv-t", "lv-s"].forEach(id => d3.select("#" + id).on("change", draw));
  ["lv-seed", "lv-e"].forEach(id => d3.select("#" + id).on("input", draw));
  d3.select("#lv-go").on("click", draw);
  draw();
})();

/* ═════════ 33 · #ll-svg — the logit lens on the trained model.  ══════════ */
(function () {
  const svg = d3.select("#ll-svg"); if (svg.empty()) return;
  const W = 760, H = 400;
  const El = id => document.getElementById(id);
  function draw() {
    const taskName = El("ll-t").value, steps = +El("ll-s").value, ei = +El("ll-e").value;
    El("ll-ev").textContent = ei;
    const R = TR.train({ task: taskName, steps: steps, seed: 1, cfg: { pos: "learned" } }), m = R.m, V = m.cfg.V, n = m.cfg.nCtx, L = m.cfg.L;
    const ex = R.evalSet[ei], fo = DL.tfForward(m, ex.idx);
    const lens = fo.st.xs.map(X => { const hN = DL.lnForward(X, m.nf.g, m.nf.b).Y; return DL.matmul(hN, DL.transpose(m.E)).map(z => DL.softmax(z)); });
    const sup = d3.range(n).filter(i => ex.tgt[i] >= 0);
    const pCorrect = lens.map(P => DL.mean(sup.map(i => P[i][ex.tgt[i]])));
    const kl = lens.map(P => DL.mean(sup.map(i => { let s = 0; for (let t = 0; t < V; t++) if (lens[L][i][t] > 0) s += lens[L][i][t] * Math.log(lens[L][i][t] / Math.max(P[i][t], 1e-12)); return s; })));
    const f = DL.frame(svg, W, H, { l: 44, r: 10, t: 22, b: 30 }), g = f.g;
    const cw = 20, ch = 14, labels = ["after the embedding (x⁰)", "after block 1 (x¹)", "after block 2 (x²) = output"];
    lens.forEach((P, l) => {
      const px = l * (V * cw + 40);
      TR.title(g, px, -8, labels[l]);
      TR.heat(g, P, px, 0, cw, d3.scaleSequential(d3.interpolateViridis).domain([0, 1]), { ch: ch, stroke: DC.bg });
      for (let i = 0; i < n; i++) { if (ex.tgt[i] >= 0) g.append("rect").attr("x", px + ex.tgt[i] * cw).attr("y", i * ch).attr("width", cw).attr("height", ch).attr("fill", "none").attr("stroke", DC.lime).attr("stroke-width", 1.3); if (l === 0) TR.note(g, -6, i * ch + 10, `${i}: ${TR.tokenLabel(ex.idx[i], V)}`, DC.muted, 8.5, "end"); }
      for (let t = 0; t < V; t++) TR.note(g, px + t * cw + cw / 2, n * ch + 11, TR.tokenLabel(t, V), DC.muted, 8, "middle");
      TR.note(g, px, n * ch + 26, `p(correct) = ${DL.fmt(pCorrect[l], 3)} · KL to output = ${DL.fmt(kl[l], 3)}`, DC.ink, 9);
    });
    /* lower: the two curves */
    const ly = n * ch + 50, lh = f.ih - ly, lw = 300;
    const gl = g.append("g").attr("transform", `translate(0,${ly})`);
    TR.title(gl, 0, -6, "mean over supervised positions, against depth");
    const x = d3.scalePoint().domain(d3.range(L + 1)).range([20, lw - 20]), y1 = d3.scaleLinear().domain([0, 1]).range([lh, 0]);
    DL.axisB(gl, d3.scaleLinear().domain([0, L]).range([20, lw - 20]), lh, L, "layer l"); DL.axisL(gl, y1, 3, "p(correct)");
    DL.curve(gl, pCorrect.map((v, l) => [x(l), y1(v)]), { stroke: DC.good, w: 2 });
    gl.selectAll(null).data(pCorrect).join("circle").attr("cx", (v, l) => x(l)).attr("cy", v => y1(v)).attr("r", 3.5).attr("fill", DC.good);
    const g2 = g.append("g").attr("transform", `translate(${lw + 60},${ly})`);
    const y2 = d3.scaleLinear().domain([0, Math.max(0.1, ...kl) * 1.1]).range([lh, 0]);
    DL.axisB(g2, d3.scaleLinear().domain([0, L]).range([20, lw - 20]), lh, L, "layer l"); DL.axisL(g2, y2, 3, "KL(output ‖ lens)");
    DL.curve(g2, kl.map((v, l) => [x(l), y2(v)]), { stroke: DC.a2, w: 2 });
    g2.selectAll(null).data(kl).join("circle").attr("cx", (v, l) => x(l)).attr("cy", v => y2(v)).attr("r", 3.5).attr("fill", DC.a2);
    const arrive = pCorrect.findIndex(v => v > 0.5);
    El("ll-readout").innerHTML =
      `Task "${taskName}", example ${ei}, ${steps} steps. Mean probability of the correct token through the lens: after the embedding <b>${DL.fmt(pCorrect[0], 3)}</b>, after block 1 <b>${DL.fmt(pCorrect[1], 3)}</b>, after block 2 <b>${DL.fmt(pCorrect[2], 3)}</b>. KL from the final output: ${kl.map(v => DL.fmt(v, 3)).join(", ")}. ` +
      (arrive < 0 ? `The answer never passes 0.5 through the lens on this example.` : arrive === 0 ? `The answer is readable already after the embedding.` : `The answer becomes readable (p > 0.5) after block ${arrive}.`) +
      ` After the embedding the lens reads the current token with mean probability ${DL.fmt(DL.mean(sup.map(i => lens[0][i][ex.idx[i]])), 3)} — the identity leak of §21.`;
  }
  ["ll-t", "ll-s"].forEach(id => d3.select("#" + id).on("change", draw));
  d3.select("#ll-e").on("input", draw);
  draw();
})();

/* ═════════ 34 · #f3-svg — three families at one budget.  ════════════════ */
(function () {
  const svg = d3.select("#f3-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const El = id => document.getElementById(id);
  function draw() {
    const d = +El("f3-d").value, L = +El("f3-L").value, ns = +El("f3-s").value, nt = +El("f3-t").value, mfrac = +El("f3-m").value;
    El("f3-sv").textContent = ns; El("f3-tv").textContent = nt;
    const fam = [
      { name: "encoder-only", col: DC.accent, params: 12 * d * d * L, wM: 12 * d * d * L * ns, aM: 2 * ns * ns * d * L, sup: Math.round(mfrac * ns), cache: 0 },
      { name: "decoder-only", col: DC.a2, params: 12 * d * d * L, wM: 12 * d * d * L * (ns + nt), aM: (ns + nt) * (ns + nt) * d * L, sup: ns + nt - 1, cache: 2 * d * L },
      { name: "encoder–decoder", col: DC.good, params: 12 * d * d * (L / 2) + 16 * d * d * (L / 2), wM: 12 * d * d * (L / 2) * ns + 16 * d * d * (L / 2) * nt, aM: 2 * ns * ns * d * (L / 2) + nt * nt * d * (L / 2) + 2 * ns * nt * d * (L / 2), sup: nt - 1, cache: 2 * d * (L / 2) }];
    const f = DL.frame(svg, W, H, { l: 10, r: 10, t: 22, b: 40 }), g = f.g;
    const groups = [["block parameters", r => r.params, v => DL.big(v)], ["forward MACs per sequence", r => r.wM + r.aM, v => DL.big(v)], ["supervised positions per sequence", r => r.sup, v => DL.commas(v)], ["KV cache per generated token", r => r.cache, v => v ? DL.big(v) : "none"]];
    const gw = f.iw / 4, gh = f.ih - 30;
    groups.forEach((gr, k) => {
      const gg = g.append("g").attr("transform", `translate(${k * gw},0)`);
      TR.title(gg, 0, -8, gr[0]);
      const vals = fam.map(gr[1]), y = d3.scaleLinear().domain([0, Math.max(...vals) * 1.15]).range([gh, 0]);
      fam.forEach((r, i) => {
        const v = gr[1](r), bx = 12 + i * ((gw - 24) / 3), bw = (gw - 24) / 3 - 8;
        gg.append("rect").attr("x", bx).attr("y", y(v)).attr("width", bw).attr("height", gh - y(v)).attr("fill", r.col).attr("fill-opacity", 0.85);
        if (k === 1) gg.append("rect").attr("x", bx).attr("y", y(v)).attr("width", bw).attr("height", gh - y(r.wM)).attr("fill", DC.rose).attr("fill-opacity", 0.9);
        TR.note(gg, bx + bw / 2, y(v) - 4, gr[2](v), DC.ink, 9, "middle");
        TR.note(gg, bx + bw / 2, gh + 12, ["enc", "dec", "enc–dec"][i], DC.muted, 9, "middle");
      });
      if (k === 1) TR.note(gg, 0, gh + 26, "red cap = the attention n² term", DC.rose, 8.5);
    });
    El("f3-readout").innerHTML = fam.map(r => `<b>${r.name}</b>: ${DL.big(r.params)} block parameters, ${DL.big(r.wM + r.aM)} MACs per sequence (${TR.pct(r.aM / (r.wM + r.aM))} attention), ${DL.commas(r.sup)} supervised positions, cache ${r.cache ? DL.big(r.cache) + " elements" : "none"} per generated token`).join("; ") +
      `. The decoder supervises <b>${DL.fmt(fam[1].sup / fam[0].sup, 1)}×</b> as many positions per sequence as the encoder and ${DL.fmt(fam[1].sup / fam[2].sup, 1)}× as many as the encoder–decoder, at d = ${d}, L = ${L}, n_src = ${ns}, n_tgt = ${nt}.`;
  }
  ["f3-d", "f3-L", "f3-m"].forEach(id => d3.select("#" + id).on("change", draw));
  ["f3-s", "f3-t"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();

/* ═════════ 35 · #bg-svg — four bugs, measured.  ══════════════════════════ */
(function () {
  const svg = d3.select("#bg-svg"); if (svg.empty()) return;
  const W = 760, H = 400;
  const El = id => document.getElementById(id);
  function draw() {
    const Lmax = +El("bg-L").value, pad = +El("bg-p").value, mc = El("bg-c").value === "-Infinity" ? -Infinity : -1e9, seed = +El("bg-seed").value;
    El("bg-Lv").textContent = Lmax; El("bg-pv").textContent = pad; El("bg-seedv").textContent = seed;
    const f = DL.frame(svg, W, H, { l: 44, r: 10, t: 22, b: 34 }), g = f.g;
    const pw = (f.iw - 90) / 4, ph = f.ih - 20;
    const panel = (k, ttl) => { const gg = g.append("g").attr("transform", `translate(${k * (pw + 30)},0)`); TR.title(gg, 0, -8, ttl); return gg; };
    /* 1: logit std against depth with / without the final norm */
    const n = 6, d = 16, V = 12, E = TR.gauss(V, d, seed + 40, 1 / Math.sqrt(d));
    const blocks = []; for (let l = 0; l < Lmax; l++) blocks.push(DL.blockInit(d, 2, 4 * d, { pre: true, seed: seed * 50 + l }));
    let X = TR.gauss(n, d, seed + 41, 1); const ones = DL.zeros(d).map(() => 1);
    const stdWith = [], stdWithout = [];
    const logitStd = M => { const Z = DL.matmul(M, DL.transpose(E)).flat(); return DL.std(Z); };
    stdWith.push(logitStd(DL.layerNorm(X, ones, DL.zeros(d)))); stdWithout.push(logitStd(X));
    for (let l = 0; l < Lmax; l++) { X = DL.blockForward(X, blocks[l], { mask: DL.causalMask(n) }).Y; stdWith.push(logitStd(DL.layerNorm(X, ones, DL.zeros(d)))); stdWithout.push(logitStd(X)); }
    const g1 = panel(0, "logit std against depth");
    const x1 = d3.scaleLinear().domain([0, Lmax]).range([0, pw]), y1 = d3.scaleLinear().domain([0, Math.max(...stdWithout, ...stdWith) * 1.1]).range([ph, 0]);
    DL.gridY(g1, y1, pw, 3); DL.axisB(g1, x1, ph, 4, "blocks"); DL.axisL(g1, y1, 3);
    DL.curve(g1, stdWithout.map((v, l) => [x1(l), y1(v)]), { stroke: DC.bad, w: 2 }); DL.curve(g1, stdWith.map((v, l) => [x1(l), y1(v)]), { stroke: DC.good, w: 2 });
    DL.legend(g1, [{ label: "no final norm", color: DC.bad }, { label: "with ln_f", color: DC.good }], 4, 10, { gap: 12, font: 9 });
    /* 2: padding mask before vs after the softmax */
    const nq = 8, valid = d3.range(nq).map(j => j < nq - pad);
    const Q = TR.gauss(nq, 8, seed + 42, 1), K = TR.gauss(nq, 8, seed + 43, 1), Vv = TR.gauss(nq, 8, seed + 44, 1);
    const good = DL.sdpa(Q, K, Vv, { mask: DL.padMask(nq, valid) }), raw = DL.sdpa(Q, K, Vv, {});
    const after = raw.A.map(r => r.map((v, j) => valid[j] ? v : 0)), sumsGood = good.A.map(r => r.reduce((s, v) => s + v, 0)), sumsBad = after.map(r => r.reduce((s, v) => s + v, 0));
    const g2 = panel(1, "row sums: mask before / after softmax");
    const x2 = d3.scaleBand().domain(d3.range(nq)).range([0, pw]).paddingInner(0.3), y2 = d3.scaleLinear().domain([0, 1.1]).range([ph, 0]);
    DL.gridY(g2, y2, pw, 3); DL.axisL(g2, y2, 3); DL.axisB(g2, d3.scaleLinear().domain([0, nq - 1]).range([x2.bandwidth() / 2, pw - x2.bandwidth() / 2]), ph, 4, "query row");
    sumsGood.forEach((v, i) => g2.append("rect").attr("x", x2(i)).attr("y", y2(v)).attr("width", x2.bandwidth() / 2).attr("height", ph - y2(v)).attr("fill", DC.good).attr("fill-opacity", 0.85));
    sumsBad.forEach((v, i) => g2.append("rect").attr("x", x2(i) + x2.bandwidth() / 2).attr("y", y2(v)).attr("width", x2.bandwidth() / 2).attr("height", ph - y2(v)).attr("fill", DC.bad).attr("fill-opacity", 0.85));
    DL.legend(g2, [{ label: "before (correct)", color: DC.good }, { label: "after (bug)", color: DC.bad }], 4, 10, { gap: 12, font: 9 });
    /* 3: mask on the query axis */
    const Mq = DL.zeros2(nq, nq); for (let i = 0; i < nq; i++) for (let j = 0; j < nq; j++) if (!valid[i]) Mq[i][j] = mc;
    const wrongAx = DL.sdpa(Q, K, Vv, { mask: Mq });
    const g3 = panel(2, `mask on the QUERY axis (${mc === -Infinity ? "−∞" : "−1e9"})`);
    const cw = Math.min(pw / nq, ph / nq);
    const nanRows = wrongAx.A.filter(r => r.some(v => Number.isNaN(v))).length;
    TR.heat(g3, wrongAx.A.map(r => r.map(v => Number.isNaN(v) ? -1 : v)), 0, 0, cw, v => v < 0 ? DC.bad : d3.interpolateViridis(Math.min(1, v * 2)), { stroke: DC.bg });
    TR.note(g3, 0, nq * cw + 12, nanRows ? `${nanRows} rows are not-a-number (red): loud` : `padded rows are UNIFORM (1/${nq} each): silent`, nanRows ? DC.a2 : DC.bad, 9);
    TR.note(g3, 0, nq * cw + 24, `row ${nq - 1} sums to ${Number.isNaN(wrongAx.A[nq - 1][0]) ? "not-a-number" : DL.fmt(wrongAx.A[nq - 1].reduce((s, v) => s + v, 0), 3)}`, DC.muted, 9);
    /* 4: RoPE on V */
    const P = DL.mhaInit(16, 8, 8, 2, { seed: seed + 45 }), Xr = TR.gauss(nq, 16, seed + 46, 1), shift = 37;
    const outQK = off => DL.attnForward(Xr, P, { mask: DL.causalMask(nq), rope: { base: 100, offset: off } }).Y;
    const outQKV = off => { /* rotate V too, by hand: V' = rope(V) per head */
      const Q0 = DL.splitHeads(DL.matmul(Xr, P.Wq), 2), K0 = DL.splitHeads(DL.matmul(Xr, P.Wk), 2), V0 = DL.splitHeads(DL.matmul(Xr, P.Wv), 2);
      const per = []; for (let j = 0; j < 2; j++) per.push(DL.sdpa(DL.rope(Q0[j], { base: 100, offset: off }), DL.rope(K0[j], { base: 100, offset: off }), DL.rope(V0[j], { base: 100, offset: off }), { mask: DL.causalMask(nq) }).Z);
      return DL.matmul(DL.mergeHeads(per), P.Wo);
    };
    const resQK = TR.maxAbs(outQK(0).map((r, i) => r.map((v, j) => v - outQK(shift)[i][j]))), resQKV = TR.maxAbs(outQKV(0).map((r, i) => r.map((v, j) => v - outQKV(shift)[i][j])));
    const scaleOut = TR.maxAbs(outQK(0));
    const g4 = panel(3, `shift every position by ${shift}: max |ΔY|`);
    const y4 = d3.scaleLog().domain([1e-17, 10]).range([ph, 0]);
    DL.axisL(g4, y4, 4, "", v => Math.abs(Math.log10(v) % 1) < 1e-9 && Math.round(Math.log10(v)) % 4 === 0 ? DL.fmtE(v, 0) : "");
    [["RoPE on Q, K", resQK, DC.good], ["RoPE on Q, K, V", resQKV, DC.bad]].forEach((b, i) => { const v = Math.max(b[1], 1e-17); g4.append("rect").attr("x", 14 + i * (pw / 2)).attr("y", y4(v)).attr("width", pw / 2 - 28).attr("height", ph - y4(v)).attr("fill", b[2]).attr("fill-opacity", 0.85); TR.note(g4, 14 + i * (pw / 2) + (pw / 2 - 28) / 2, y4(v) - 4, DL.fmtE(b[1], 1), DC.ink, 9, "middle"); TR.note(g4, 14 + i * (pw / 2) + (pw / 2 - 28) / 2, ph + 12, b[0], DC.muted, 8.5, "middle"); });
    El("bg-readout").innerHTML =
      `(1) Logit std after ${Lmax} pre-norm blocks: <b>${DL.fmt(stdWithout[Lmax], 2)}</b> without the final norm (from ${DL.fmt(stdWithout[0], 2)} at the input) against <b>${DL.fmt(stdWith[Lmax], 2)}</b> with it. ` +
      `(2) Padding mask with ${pad} padded keys: rows sum to 1.000000 before the softmax; after it they sum to <b>${sumsBad.map(v => DL.fmt(v, 3)).join(", ")}</b> — each output rescaled by its own factor. ` +
      `(3) The same mask on the query axis with ${mc === -Infinity ? "−∞" : "−1e9"}: ${nanRows ? `<b>${nanRows} not-a-number rows</b> — the report you want` : `<b>no error</b>, the padded query rows attend uniformly to every key`}. ` +
      `(4) Shifting every position by ${shift} changes the attention output by <b>${DL.fmtE(resQK, 1)}</b> with RoPE on Q and K (output scale ${DL.fmt(scaleOut, 2)}) and by <b>${DL.fmtE(resQKV, 1)}</b> with RoPE also on V.`;
  }
  d3.select("#bg-c").on("change", draw);
  ["bg-L", "bg-p", "bg-seed"].forEach(id => d3.select("#" + id).on("input", draw));
  draw();
})();
