/* rnns-lstms.viz.js — the visualizations on deep-learning/rnns-lstms.html
   (Deep Learning · part 4). Loaded after ../data.js → ../notes.js → dl-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the
   page, so a section can be reordered or removed without breaking the rest.

   Every numeric primitive — the recurrent cells and their backward passes,
   the truncated forms, the state-to-state Jacobian, the eigenvalues and the
   spectral radius, the parameter counts, the activations, the stable losses,
   the optimisers and the clipping — comes from dl-viz.js (DL.*). NOTHING
   here forks those. Read the SEQUENCE LAYOUT comment above DL.CELL before
   touching any of it: a sequence is TIME-MAJOR, (T × d), one step per ROW,
   and W is (dₕ × dₕ) acting on the RIGHT.
   Page-specific numerics live in RN below.

     1  #uf-svg    the recurrence, and the same thing unfolded
     2  #wn-svg    the fixed window against the recurrent cell
     3  #sh-svg    four ways to move information a distance d
     4  #tr-svg    the state, step by step
     5  #sp-svg    one step, with every shape and every count
     6  #wr-svg    the five wirings and their gradient paths
     7  #ls-svg    where the loss enters, and what each step receives
     8  #dv-svg    ∂L/∂V as a sum of outer products, audited
     9  #pt-svg    every path from a loss to a use of W
    10  #du-svg    the three parameter gradients and their shared factor
    11  #rc-svg    the double sum against the O(T) recursion
    12  #wk-svg    the worked example, live
    13  #au-svg    every parameter, analytic against numeric
    14  #ct-svg    time, memory and latency against length
    15  #tc-svg    which terms each truncation form keeps
    16  #tb-svg    the truncation bias, measured
    17  #pw-svg    a matrix raised to a power
    18  #sr-svg    sweeping the spectral radius
    19  #cr-svg    three candidate criteria, on separating matrices
    20  #td-svg    the same matrix every step against a fresh one
    21  #cl-svg    the cliff, and one step with and without a clip
    22  #rm-svg    four remedies, measured on gradient at distance
    23  #lk-svg    leaky units, delays and reservoirs
    24  #lm-svg    the LSTM cell, with real values flowing
    25  #gt-svg    ablate one gate and see what breaks
    26  #cs-svg    gradient at distance: vanilla, GRU, LSTM
    27  #fb-svg    sweeping the forget-gate bias
    28  #pm-svg    parameter counts and matched widths
    29  #vr-svg    the variants as edits to one diagram
    30  #gr-svg    GRU and LSTM side by side
    31  #cm-svg    the adding problem, fitted live
    32  #bi-svg    one direction against two, and the leak
    33  #sk-svg    stacking: path lengths and time constants
    34  #ss-svg    the encoder-decoder pair and its one edge
    35  #tf-svg    teacher forcing against free running
    36  #bn-svg    what survives in the context vector
    37  #dc-svg    greedy, beam and exhaustive search
    38  #tw-svg    chain against tree
    39  #nw-svg    recurrent against attention, four costs
    40  #dg-svg    the four diagnostics, with faults you can inject

   Every number these print is recomputed from the data they draw.          */

/* ══════════ page-local helpers (deliberately NOT in dl-viz.js) ══════════ */
const RN = (function () {

  /* ---- seeded data ------------------------------------------------- */
  function seq(T, dx, seed, kind) {
    const r = DL.rng(seed || 3), X = [];
    for (let t = 0; t < T; t++) {
      const x = new Array(dx).fill(0);
      if (kind === "pulse") { if (t === 0) for (let j = 0; j < dx; j++) x[j] = 2; }
      else if (kind === "two") { if (t === 0 || t === Math.min(T - 1, 19)) for (let j = 0; j < dx; j++) x[j] = 2; }
      else if (kind === "step") { const v = t < T / 2 ? -1 : 1; for (let j = 0; j < dx; j++) x[j] = v; }
      else if (kind === "zero") { /* leave zeros */ }
      else for (let j = 0; j < dx; j++) x[j] = DL.randn(r);
      X.push(x);
    }
    return X;
  }
  function targets(T, dy, kind, seed, where) {
    const r = DL.rng(seed || 5), Y = [];
    for (let t = 0; t < T; t++) {
      let has = true;
      if (where === "last") has = (t === T - 1);
      else if (where === "first") has = (t === 0);
      else if (where === "sparse") has = (t % 5 === 0);
      else if (where === "half") has = (t >= T / 2);
      if (!has) { Y.push(null); continue; }
      if (kind === "softmax") { const k = Math.floor(r() * dy); Y.push(new Array(dy).fill(0).map((_, j) => (j === k ? 1 : 0))); }
      else if (kind === "linear") Y.push(new Array(dy).fill(0).map(() => DL.randn(r) * 0.5));
      else Y.push(new Array(dy).fill(0).map(() => (r() < 0.5 ? 0 : 1)));
    }
    return Y;
  }
  /* a network with a known spectral radius, for the figures that sweep it */
  function net(cell, dx, dh, dy, o) {
    return DL.seqInit(cell, dx, dh, dy, Object.assign({ seed: 7, out: "sigmoid" }, o || {}));
  }
  /* a seeded square matrix, optionally rescaled to a target spectral radius */
  function mat(n, seed, rho) {
    const r = DL.rng(seed || 9), A = DL.zeros2(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i][j] = DL.randn(r);
    return (rho === undefined || rho === null) ? A : DL.scaleToRho(A, rho);
  }
  /* a random orthogonal matrix by Gram–Schmidt: every |λ| and every σ is 1 */
  function ortho(n, seed) {
    const r = DL.rng(seed || 4), Q = DL.zeros2(n, n);
    for (let j = 0; j < n; j++) {
      const v = []; for (let i = 0; i < n; i++) v.push(DL.randn(r));
      for (let p = 0; p < j; p++) {
        let d = 0; for (let i = 0; i < n; i++) d += v[i] * Q[i][p];
        for (let i = 0; i < n; i++) v[i] -= d * Q[i][p];
      }
      let nn = 0; for (let i = 0; i < n; i++) nn += v[i] * v[i];
      nn = Math.sqrt(nn) || 1;
      for (let i = 0; i < n; i++) Q[i][j] = v[i] / nn;
    }
    return Q;
  }
  /* ---- small numeric utilities the figures share -------------------- */
  const norm = a => Math.sqrt(a.reduce((s, x) => s + x * x, 0));
  const flat = A => { const o = []; for (const r of A) for (const v of r) o.push(v); return o; };
  const fnorm = A => norm(flat(A));
  const cosv = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0) / Math.max(1e-300, norm(a) * norm(b));
  /* least-squares slope of log y against x — the measured per-step rate  */
  function logSlope(xs, ys) {
    const pts = [];
    for (let i = 0; i < xs.length; i++) if (ys[i] > 0 && isFinite(ys[i])) pts.push([xs[i], Math.log(ys[i])]);
    if (pts.length < 2) return NaN;
    const n = pts.length;
    const mx = pts.reduce((s, p) => s + p[0], 0) / n, my = pts.reduce((s, p) => s + p[1], 0) / n;
    let num = 0, den = 0;
    pts.forEach(p => { num += (p[0] - mx) * (p[1] - my); den += (p[0] - mx) * (p[0] - mx); });
    return den === 0 ? NaN : Math.exp(num / den);
  }
  /* ---- colour ------------------------------------------------------- */
  const div = d3.scaleLinear().domain([-1, 0, 1]).range([DC.rose, DC.panel2, DC.accent]).clamp(true);
  const signColor = (v, m) => div(m > 0 ? v / m : 0);
  const heat = d3.scaleSequential(d3.interpolateViridis);
  const CELLC = { vanilla: DC.bad, gru: DC.a2, lstm: DC.good, attn: DC.violet, linrec: DC.teal };
  /* ---- drawing helpers ---------------------------------------------- */
  function title(g, x, y, t, col) {
    g.append("text").attr("x", x).attr("y", y).attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", col || DC.ink).text(t);
  }
  function note(g, x, y, t, col, size) {
    g.append("text").attr("x", x).attr("y", y).attr("font-size", size || 10)
      .attr("fill", col || DC.muted).text(t);
  }
  function chip(g, x, y, t, col) {
    const w = t.length * 5.6 + 12;
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    gg.append("rect").attr("x", 0).attr("y", -9).attr("width", w).attr("height", 15).attr("rx", 4)
      .attr("fill", col).attr("fill-opacity", 0.16).attr("stroke", col).attr("stroke-opacity", 0.55);
    gg.append("text").attr("x", w / 2).attr("y", 2).attr("text-anchor", "middle")
      .attr("font-size", 9.5).attr("fill", col).text(t);
    return w;
  }
  /* a log scale that survives zeros and negatives by flooring */
  function logY(vals, ih, floorAt) {
    const pos = vals.filter(v => v > 0 && isFinite(v));
    const lo = pos.length ? Math.min(...pos) : 1e-12, hi = pos.length ? Math.max(...pos) : 1;
    const f = floorAt === undefined ? Math.min(lo, hi * 1e-12) : floorAt;
    return d3.scaleLog().domain([Math.max(f, 1e-300), Math.max(hi * 1.6, f * 10)]).range([ih, 0]).clamp(true);
  }
  const sup = { 1: "¹", 2: "²", 3: "³" };
  /* pretty a value for a readout */
  const P = (x, d) => (Math.abs(x) >= 1e-3 && Math.abs(x) < 1e5) ? DL.fmt(x, d === undefined ? 4 : d) : DL.fmtE(x, 2);

  /* ---- an Adam trainer for the figures that fit something live ------ */
  function packer(net) {
    const nc = DL.cellFlatten(net.p).v.length, dh = net.dh, dy = net.dy;
    return {
      n: nc + dh * dy + dy,
      get() { return DL.cellFlatten(net.p).v.concat(flat(net.V)).concat(net.c); },
      set(v) {
        DL.cellUnflatten(net.p, v.slice(0, nc));
        let k = nc;
        for (let i = 0; i < dh; i++) for (let j = 0; j < dy; j++) net.V[i][j] = v[k++];
        for (let j = 0; j < dy; j++) net.c[j] = v[k++];
      },
      grad(g) { return DL.cellGradFlatten(net.p, g).concat(flat(g.dV)).concat(g.dc); }
    };
  }
  function trainer(net, lr, clipAt) {
    const pk = packer(net), o = DL.OPT.adam, hp = Object.assign({}, o.hp, { lr: lr });
    const st = o.init(pk.n);
    let v = pk.get();
    return {
      step(batch) {                       // batch = [{X, Y}, …]
        const G = new Array(pk.n).fill(0);
        let L = 0;
        for (const b of batch) {
          const g = DL.seqBPTT(net, b.X, b.Y);
          const gf = pk.grad(g);
          for (let i = 0; i < G.length; i++) G[i] += gf[i] / batch.length;
          L += g.loss / batch.length;
        }
        const cl = DL.clipNorm(G, clipAt === undefined ? 1 : clipAt);
        const d = o.step(st, cl.g, hp);
        for (let i = 0; i < v.length; i++) v[i] += d[i];
        pk.set(v);
        return { loss: L, norm: cl.norm, clipped: cl.scaled };
      }
    };
  }
  /* the adding problem, used by §31 and §25 and §40 */
  function addBatch(T, n, r) {
    const B = [];
    for (let i = 0; i < n; i++) {
      const X = [], Y = new Array(T).fill(null);
      const i1 = Math.floor(r() * Math.max(1, Math.floor(T / 10)));
      let i2 = Math.floor(r() * T); if (i2 === i1) i2 = (i1 + 1) % T;
      let s = 0;
      for (let t = 0; t < T; t++) {
        const v = r(), m = (t === i1 || t === i2) ? 1 : 0;
        if (m) s += v;
        X.push([v, m]);
      }
      Y[T - 1] = [s];
      B.push({ X, Y, s, i1, i2 });
    }
    return B;
  }
  /* the three memory tasks used by §25 and §40. Input width is K + 1: K value
     channels and one MARKER channel. Read out at the last step only.
       hold    the cue is at step 0; everything after is distraction
       reset   the same, but half the sequences carry a marker that resets the
               answer to class 0 — so "always answer 0" is not a winning constant
       ignore  the cue is at a RANDOM step and is the only step whose marker
               channel is on; every other step carries plausible-looking values
     Chance is 1 − 1/K. */
  function memBatch(kind, T, K, n, r) {
    const B = [];
    for (let i = 0; i < n; i++) {
      const X = [], Y = new Array(T).fill(null);
      let cur = 0;
      if (kind === "ignore") {
        const m = 1 + Math.floor(r() * Math.max(1, T - 2)), k = Math.floor(r() * K);
        cur = k;
        for (let t = 0; t < T; t++) {
          const x = new Array(K + 1).fill(0);
          if (t === m) { x[k] = 1; x[K] = 1; }
          else for (let j = 0; j < K; j++) x[j] = (r() < 0.35 ? 1 : 0) * (r() < 0.5 ? 1 : -1);
          X.push(x);
        }
      } else {
        const k = Math.floor(r() * K);
        cur = k;
        const reset = (kind === "reset" && r() < 0.5) ? (1 + Math.floor(r() * Math.max(1, T - 2))) : -1;
        for (let t = 0; t < T; t++) {
          const x = new Array(K + 1).fill(0);
          if (t === 0) x[k] = 1;
          else if (t === reset) { x[K] = 1; cur = 0; }
          else for (let j = 0; j < K; j++) x[j] = 0.3 * DL.randn(r);
          X.push(x);
        }
      }
      Y[T - 1] = new Array(K).fill(0).map((_, j) => (j === cur ? 1 : 0));
      B.push({ X, Y, k: cur });
    }
    return B;
  }

  return { seq, targets, net, mat, ortho, norm, flat, fnorm, cosv, logSlope,
           div, signColor, heat, CELLC, title, note, chip, logY, P, sup,
           packer, trainer, addBatch, memBatch };
})();

/* ═══════════ 1 · #uf-svg — the recurrence, and the unfolding ═══════════ */
(function () {
  const svg = d3.select("#uf-svg");
  if (svg.empty()) return;
  const W = 760, H = 330;
  const Te = document.getElementById("uf-T"), Me = document.getElementById("uf-mode"),
    Pe = document.getElementById("uf-pat"), Le = document.getElementById("uf-lbl");

  function draw() {
    const T = +Te.value, mode = Me.value, pat = Pe.value, lab = Le.checked;
    document.getElementById("uf-Tv").textContent = T;
    const f = DL.frame(svg, W, H, { l: 14, r: 14, t: 16, b: 14 });
    const g = f.g;

    const showCycle = mode !== "unroll", showRoll = mode !== "cycle";
    const cyW = 190, gap = 34;
    let x0 = 0;

    if (showCycle) {
      RN.title(g, 0, 6, "the recurrent layer");
      const cx = 66, ys = { x: 190, h: 118, o: 46 };
      const box = (yy, t, col) => {
        g.append("rect").attr("x", cx - 26).attr("y", yy - 15).attr("width", 52).attr("height", 30)
          .attr("rx", 6).attr("fill", DC.panel2).attr("stroke", col).attr("stroke-width", 1.4);
        g.append("text").attr("x", cx).attr("y", yy + 4).attr("text-anchor", "middle")
          .attr("font-size", 12).attr("fill", DC.ink).text(t);
      };
      box(ys.x, "x", DC.accent);
      box(ys.h, "h", DC.a2);
      if (pat !== "none") box(ys.o, "o", DC.good);
      DL.arrow(g, cx, ys.x - 16, cx, ys.h + 17, { color: DC.accent, w: 1.6 });
      if (lab) RN.note(g, cx + 6, (ys.x + ys.h) / 2 + 4, "U", DC.accent);
      if (pat !== "none") {
        DL.arrow(g, cx, ys.h - 16, cx, ys.o + 17, { color: DC.good, w: 1.6 });
        if (lab) RN.note(g, cx + 6, (ys.h + ys.o) / 2 + 4, "V", DC.good);
      }
      /* the self loop, with the unit-delay square */
      const loop = `M${cx + 27},${ys.h - 6} C${cx + 78},${ys.h - 34} ${cx + 78},${ys.h + 34} ${cx + 27},${ys.h + 6}`;
      g.append("path").attr("d", loop).attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 1.8);
      g.append("rect").attr("x", cx + 56).attr("y", ys.h - 8).attr("width", 16).attr("height", 16)
        .attr("fill", DC.a2).attr("fill-opacity", 0.9);
      if (lab) RN.note(g, cx + 44, ys.h - 16, "W", DC.a2);
      RN.note(g, cx - 30, ys.h + 46, "the black square is a", DC.muted, 9.5);
      RN.note(g, cx - 30, ys.h + 57, "one-step delay", DC.muted, 9.5);
      x0 = cyW;
      if (showRoll) DL.arrow(g, x0 - 22, 118, x0 + 4, 118, { color: DC.muted, w: 1.4 });
      x0 += gap - 20;
    }

    if (showRoll) {
      RN.title(g, x0, 6, "unfolded along time — one column per step, the SAME three matrices");
      const avail = f.iw - x0 - 8;
      const step = Math.min(66, avail / Math.max(1, T));
      const r = 13;
      const yy = { x: 190, h: 118, o: 46, L: 8 };
      const px = i => x0 + 24 + i * step;
      let nodes = 0, edges = 0;
      for (let t = 0; t < T; t++) {
        const X = px(t);
        const hasOut = pat === "many" || (pat === "last" && t === T - 1);
        /* input */
        g.append("circle").attr("cx", X).attr("cy", yy.x).attr("r", r)
          .attr("fill", DC.panel2).attr("stroke", DC.accent).attr("stroke-width", 1.3);
        g.append("text").attr("x", X).attr("y", yy.x + 4).attr("text-anchor", "middle")
          .attr("font-size", 9.5).attr("fill", DC.muted).text("x" + (t + 1));
        nodes++;
        /* state */
        g.append("circle").attr("cx", X).attr("cy", yy.h).attr("r", r)
          .attr("fill", DC.panel2).attr("stroke", DC.a2).attr("stroke-width", 1.6);
        g.append("text").attr("x", X).attr("y", yy.h + 4).attr("text-anchor", "middle")
          .attr("font-size", 9.5).attr("fill", DC.ink).text("h" + (t + 1));
        nodes++;
        DL.arrow(g, X, yy.x - r - 1, X, yy.h + r + 1, { color: DC.accent, w: 1.2, head: 4 });
        edges++;
        if (lab && step > 40) RN.note(g, X + 4, (yy.x + yy.h) / 2 + 3, "U", DC.accent, 9);
        if (t > 0) {
          DL.arrow(g, px(t - 1) + r + 1, yy.h, X - r - 1, yy.h, { color: DC.a2, w: 1.4, head: 5 });
          edges++;
          if (lab && step > 34) RN.note(g, (px(t - 1) + X) / 2 - 3, yy.h - 6, "W", DC.a2, 9);
        }
        if (hasOut) {
          g.append("circle").attr("cx", X).attr("cy", yy.o).attr("r", r)
            .attr("fill", DC.panel2).attr("stroke", DC.good).attr("stroke-width", 1.3);
          g.append("text").attr("x", X).attr("y", yy.o + 4).attr("text-anchor", "middle")
            .attr("font-size", 9.5).attr("fill", DC.muted).text("o" + (t + 1));
          nodes++;
          DL.arrow(g, X, yy.h - r - 1, X, yy.o + r + 1, { color: DC.good, w: 1.2, head: 4 });
          edges++;
          if (lab && step > 40) RN.note(g, X + 4, (yy.h + yy.o) / 2 + 3, "V", DC.good, 9);
          g.append("text").attr("x", X).attr("y", yy.L + 4).attr("text-anchor", "middle")
            .attr("font-size", 9.5).attr("fill", DC.bad).text("L" + (t + 1));
          nodes++; edges++;
        }
      }
      /* h0 stub */
      g.append("text").attr("x", px(0) - r - 12).attr("y", yy.h + 4).attr("text-anchor", "end")
        .attr("font-size", 9.5).attr("fill", DC.muted).text("h₀");
      DL.arrow(g, px(0) - r - 10, yy.h, px(0) - r - 1, yy.h, { color: DC.line, w: 1.2, head: 4 });

      /* the counts */
      const dx = 8, dh = 16, dy = 4;
      const par = dx * dh + dh * dh + dh + dh * dy + dy;
      const nOut = pat === "many" ? T : (pat === "last" ? 1 : 0);
      RN.note(g, x0, 240, `nodes in the unfolded graph  ${nodes}`, DC.muted);
      RN.note(g, x0, 254, `edges                        ${edges}`, DC.muted);
      RN.note(g, x0, 268, `DISTINCT weight matrices     3   ← does not change with T`, DC.good);
      RN.note(g, x0, 282, `parameters (dₓ=8, dₕ=16, d_y=4)  ${DL.commas(par)}   ← does not change with T`, DC.good);

      document.getElementById("uf-readout").innerHTML =
        `T = <b>${T}</b> · unfolded graph: <b>${nodes}</b> nodes, <b>${edges}</b> edges, <b>${nOut}</b> loss term${nOut === 1 ? "" : "s"} · ` +
        `distinct parameter matrices <b>3</b> · parameters <b>${DL.commas(par)}</b> at dₓ=8, dₕ=16, d_y=4 — ` +
        `the graph grows with T and the parameter count does not, which is the entire point of sharing.`;
    } else {
      document.getElementById("uf-readout").innerHTML =
        `The cyclic diagram alone. It has <b>3</b> edges and <b>3</b> matrices whatever the sequence length is; ` +
        `switch the view to see what it becomes when the loop is written out.`;
    }
  }
  [Te, Me, Pe, Le].forEach(e => e.addEventListener(e.type === "checkbox" || e.tagName === "SELECT" ? "change" : "input", draw));
  Le.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 2 · #wn-svg — the window against the recurrence ═══════════ */
(function () {
  const svg = d3.select("#wn-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Te = document.getElementById("wn-T"), Xe = document.getElementById("wn-dx"),
    He = document.getElementById("wn-dh"), Re = document.getElementById("wn-tr");

  /* the measured half: a linear detector fitted at some positions only.
     window model   — one weight vector per POSITION  (T·dₓ weights)
     shared model   — ONE weight vector, used at every position (dₓ weights)
     both fitted by least squares on the same data, then tested everywhere. */
  function generalisation(T, dx, trainKind) {
    const r = DL.rng(21), n = 160;
    const patt = []; for (let j = 0; j < dx; j++) patt.push(DL.randn(r));
    const trainPos = trainKind === "1" ? [0]
      : trainKind === "2" ? [0, Math.floor(T / 2)]
        : trainKind === "half" ? d3.range(0, T, 2)
          : d3.range(T);
    /* data: at each position, either the pattern (+noise) or noise */
    const make = () => {
      const X = [], y = [];
      for (let i = 0; i < n; i++) {
        const pos = trainPos[Math.floor(r() * trainPos.length)];
        const lab = r() < 0.5 ? 1 : 0;
        const row = DL.zeros2(T, dx).map(rr => rr.map(() => 0.6 * DL.randn(r)));
        if (lab) for (let j = 0; j < dx; j++) row[pos][j] += patt[j];
        X.push({ row, pos }); y.push(lab);
      }
      return { X, y };
    };
    const D = make();
    /* shared: features are the pattern-projection at the ACTIVE position */
    const Ash = D.X.map(d => d.row[d.pos].concat([1]));
    const wsh = DL.ridge(Ash, D.y.map(v => [v]), 1e-3);
    /* window: features are the full flattened window */
    const Awn = D.X.map(d => { const o = []; for (let t = 0; t < T; t++) for (let j = 0; j < dx; j++) o.push(d.row[t][j]); o.push(1); return o; });
    const wwn = DL.ridge(Awn, D.y.map(v => [v]), 1e-3);
    /* test at EVERY position */
    const accSh = [], accWn = [];
    for (let p = 0; p < T; p++) {
      let cs = 0, cw = 0, m = 60;
      for (let i = 0; i < m; i++) {
        const lab = i % 2;
        const row = DL.zeros2(T, dx).map(rr => rr.map(() => 0.6 * DL.randn(r)));
        if (lab) for (let j = 0; j < dx; j++) row[p][j] += patt[j];
        const fs = row[p].concat([1]);
        let ps = 0; for (let j = 0; j < fs.length; j++) ps += fs[j] * wsh[j][0];
        if ((ps > 0.5 ? 1 : 0) === lab) cs++;
        const fw = []; for (let t = 0; t < T; t++) for (let j = 0; j < dx; j++) fw.push(row[t][j]); fw.push(1);
        let pw = 0; for (let j = 0; j < fw.length; j++) pw += fw[j] * wwn[j][0];
        if ((pw > 0.5 ? 1 : 0) === lab) cw++;
      }
      accSh.push(cs / m); accWn.push(cw / m);
    }
    return { accSh, accWn, trainPos };
  }

  function draw() {
    const T = +Te.value, dx = +Xe.value, dh = +He.value, tk = Re.value;
    document.getElementById("wn-Tv").textContent = T;
    document.getElementById("wn-dxv").textContent = dx;
    document.getElementById("wn-dhv").textContent = dh;
    const f = DL.frame(svg, W, H, { l: 16, r: 14, t: 22, b: 16 }), g = f.g;

    const pWin = T * dx * dh + dh, pRec = dx * dh + dh * dh + dh;
    /* ---- left: the two parameter counts ---- */
    RN.title(g, 0, 0, "parameters");
    const bw = 150, y0 = 26;
    const sc = d3.scaleLog().domain([Math.min(pRec, pWin) / 3, Math.max(pRec, pWin) * 1.2]).range([4, bw]).clamp(true);
    [["flattened window", pWin, DC.bad], ["recurrent cell", pRec, DC.good]].forEach((d, i) => {
      const yy = y0 + i * 46;
      g.append("rect").attr("x", 0).attr("y", yy).attr("width", sc(d[1])).attr("height", 18)
        .attr("rx", 3).attr("fill", d[2]).attr("fill-opacity", 0.55).attr("stroke", d[2]);
      RN.note(g, 0, yy - 4, d[0], DC.muted, 10);
      g.append("text").attr("x", 4).attr("y", yy + 32).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(DL.commas(d[1]));
    });
    g.append("text").attr("x", 0).attr("y", y0 + 108).attr("font-size", 11).attr("fill", DC.a2)
      .attr("font-weight", 600).text("ratio  " + DL.fmt(pWin / pRec, 1) + "×");
    RN.note(g, 0, y0 + 124, "the window count is T·dₓ·dₕ + dₕ", DC.muted, 9.5);
    RN.note(g, 0, y0 + 136, "the recurrent count has no T in it", DC.muted, 9.5);

    /* ---- middle: the first weight matrix, as blocks ---- */
    const mx = 190;
    RN.title(g, mx, 0, "the first weight matrix, drawn as blocks");
    const bh = 12, bwid = 96, show = Math.min(T, 14);
    for (let t = 0; t < show; t++) {
      g.append("rect").attr("x", mx).attr("y", y0 + t * (bh + 2)).attr("width", bwid).attr("height", bh)
        .attr("rx", 2).attr("fill", d3.interpolateTurbo(t / Math.max(1, show - 1)))
        .attr("fill-opacity", 0.75).attr("stroke", DC.line);
    }
    if (T > show) RN.note(g, mx + bwid / 2 - 4, y0 + show * (bh + 2) + 10, "⋮", DC.muted, 12);
    RN.note(g, mx, y0 + Math.min(T, show) * (bh + 2) + (T > show ? 22 : 12), `${T} independent blocks`, DC.bad, 9.5);
    RN.note(g, mx, y0 + Math.min(T, show) * (bh + 2) + (T > show ? 34 : 24), "one colour = one free block", DC.muted, 9.5);

    const rx = mx + 130;
    g.append("rect").attr("x", rx).attr("y", y0).attr("width", bwid).attr("height", bh)
      .attr("rx", 2).attr("fill", DC.good).attr("fill-opacity", 0.75).attr("stroke", DC.line);
    for (let t = 1; t < show; t++) {
      g.append("rect").attr("x", rx).attr("y", y0 + t * (bh + 2)).attr("width", bwid).attr("height", bh)
        .attr("rx", 2).attr("fill", DC.good).attr("fill-opacity", 0.75).attr("stroke", DC.line)
        .attr("stroke-dasharray", "2 2");
    }
    RN.note(g, rx, y0 + Math.min(T, show) * (bh + 2) + (T > show ? 22 : 12), "ONE block", DC.good, 9.5);
    RN.note(g, rx, y0 + Math.min(T, show) * (bh + 2) + (T > show ? 34 : 24), "reused at every step", DC.muted, 9.5);

    /* ---- right: measured generalisation across positions ---- */
    const gx = 470, gw = f.iw - gx, gh = 200;
    RN.title(g, gx, 0, "accuracy at each position, after fitting at the marked ones");
    const gg = g.append("g").attr("transform", `translate(${gx},${y0})`);
    const res = generalisation(Math.min(T, 24), Math.min(dx, 8), tk);
    const TT = Math.min(T, 24);
    const x = d3.scaleLinear().domain([0, TT - 1]).range([0, gw - 10]);
    const y = d3.scaleLinear().domain([0.4, 1]).range([gh, 0]);
    DL.gridY(gg, y, gw - 10, 4);
    DL.axisB(gg, x, gh, Math.min(TT, 8), "position");
    DL.axisL(gg, y, 4, "accuracy");
    gg.append("line").attr("x1", 0).attr("x2", gw - 10).attr("y1", y(0.5)).attr("y2", y(0.5))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.6);
    res.trainPos.forEach(p => {
      if (p < TT) gg.append("line").attr("x1", x(p)).attr("x2", x(p)).attr("y1", 0).attr("y2", gh)
        .attr("stroke", DC.a2).attr("stroke-opacity", 0.3).attr("stroke-width", 2);
    });
    DL.curve(gg, res.accSh.map((v, i) => [x(i), y(v)]), { stroke: DC.good, w: 2 });
    DL.curve(gg, res.accWn.map((v, i) => [x(i), y(v)]), { stroke: DC.bad, w: 2 });
    DL.legend(gg, [{ label: "shared weights", color: DC.good }, { label: "position-specific", color: DC.bad },
    { label: "trained positions", color: DC.a2, op: 0.35 }], 6, gh + 34, { vertical: false, step: 118, font: 9.5 });

    const untr = d3.range(TT).filter(p => res.trainPos.indexOf(p) < 0);
    const mSh = untr.length ? d3.mean(untr.map(p => res.accSh[p])) : 1;
    const mWn = untr.length ? d3.mean(untr.map(p => res.accWn[p])) : 1;
    document.getElementById("wn-readout").innerHTML =
      `window <b>${DL.commas(pWin)}</b> parameters against recurrent <b>${DL.commas(pRec)}</b> — a factor of <b>${DL.fmt(pWin / pRec, 1)}×</b>, ` +
      `and the recurrent count contains no T at all. Measured at <b>${untr.length}</b> untrained positions: ` +
      `shared weights <b>${DL.fmt(100 * mSh, 1)}%</b>, position-specific weights <b>${DL.fmt(100 * mWn, 1)}%</b> ` +
      `(chance is 50%). The gap is <b>${DL.fmt(100 * (mSh - mWn), 1)}</b> points.`;
  }
  [Te, Xe, He].forEach(e => e.addEventListener("input", draw));
  Re.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 3 · #sh-svg — four ways to move information ═══════════ */
(function () {
  const svg = d3.select("#sh-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const De = document.getElementById("sh-d"), Fe = document.getElementById("sh-f"),
    Me = document.getElementById("sh-m");

  function draw() {
    const d = +De.value, fk = +Fe.value, which = Me.value;
    document.getElementById("sh-dv").textContent = d;
    document.getElementById("sh-fv").textContent = fk;
    const f = DL.frame(svg, W, H, { l: 130, r: 16, t: 22, b: 16 }), g = f.g;

    const rows = [
      { k: "dense", label: "dense window", path: 1, grows: "yes — T·dₓ·dₕ", col: DC.bad },
      { k: "conv", label: "1-D convolution", path: Math.ceil(d / (fk - 1 || 1)), grows: "no", col: DC.violet },
      { k: "rnn", label: "recurrence", path: d, grows: "no", col: DC.a2 },
      { k: "attn", label: "self-attention", path: 1, grows: "no (but the COST does)", col: DC.teal }
    ];
    RN.title(g, 0, 0, `non-linear transformations between two positions ${d} apart`);
    const bh = 22, y0 = 16;
    const mx = Math.max(...rows.map(r => r.path));
    const sc = d3.scaleLinear().domain([0, mx]).range([0, f.iw - 90]);
    rows.forEach((r, i) => {
      const yy = y0 + i * (bh + 8);
      g.append("text").attr("x", -8).attr("y", yy + 15).attr("text-anchor", "end")
        .attr("font-size", 11).attr("fill", which === r.k ? DC.ink : DC.muted)
        .attr("font-weight", which === r.k ? 600 : 400).text(r.label);
      g.append("rect").attr("x", 0).attr("y", yy).attr("width", Math.max(2, sc(r.path))).attr("height", bh)
        .attr("rx", 3).attr("fill", r.col).attr("fill-opacity", which === r.k ? 0.7 : 0.28)
        .attr("stroke", r.col).attr("stroke-opacity", which === r.k ? 1 : 0.5);
      g.append("text").attr("x", Math.max(2, sc(r.path)) + 8).attr("y", yy + 15)
        .attr("font-size", 11).attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", DC.ink).text(r.path);
    });

    /* the matrix picture */
    const n = 16, cw = 15, ox = 0, oy = 156;
    RN.title(g, 0, oy - 8, "which entries of a position-to-position matrix that architecture keeps");
    const sel = rows.find(r => r.k === which);
    let free = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let fill = "none", op = 1, stroke = DC.grid;
      if (which === "dense") { fill = d3.interpolateTurbo((i * n + j) / (n * n)); free++; }
      else if (which === "conv") {
        const off = j - i, rad = (fk - 1) >> 1;
        if (Math.abs(off) <= rad) { fill = d3.interpolateTurbo((off + rad) / Math.max(1, 2 * rad)); }
      }
      else if (which === "rnn") { fill = (i < 6 && j < 6) ? DC.a2 : "none"; op = 0.6; }
      else if (which === "attn") { fill = DC.teal; op = 0.30; free = 0; }
      const el = g.append("rect").attr("x", ox + j * cw).attr("y", oy + i * cw)
        .attr("width", cw - 1).attr("height", cw - 1).attr("rx", 1)
        .attr("stroke", stroke).attr("stroke-opacity", 0.5);
      if (fill === "none") el.attr("fill", DC.panel).attr("fill-opacity", 0.25);
      else el.attr("fill", fill).attr("fill-opacity", op);
    }
    if (which === "conv") free = fk;
    if (which === "rnn") free = 36;
    const labels = {
      dense: ["every entry is its own free parameter", "n² parameters, and they GROW with the sequence"],
      conv: [`a band of width ${fk}; cells of one colour are FORCED equal`, `${fk} free parameters, whatever n is`],
      rnn: ["not laid out over positions at all — this is the", "dₕ × dₕ transition matrix, APPLIED at every step"],
      attn: ["every entry is used, and none is stored —", "the mixing weights are computed from the data"]
    }[which];
    RN.note(g, ox + n * cw + 18, oy + 14, labels[0], DC.ink, 10.5);
    RN.note(g, ox + n * cw + 18, oy + 28, labels[1], DC.muted, 10);
    RN.note(g, ox + n * cw + 18, oy + 52, "path length at d = " + d + ":  " + sel.path, DC.a2, 11);
    RN.note(g, ox + n * cw + 18, oy + 68, "parameters grow with sequence length:  " + sel.grows, DC.muted, 10);

    document.getElementById("sh-readout").innerHTML =
      `<b>${sel.label}</b> · path length between positions ${d} apart: <b>${sel.path}</b> ` +
      `(recurrence needs <b>${d}</b>, a ${fk}-tap convolution needs <b>${Math.ceil(d / (fk - 1 || 1))}</b>, ` +
      `dense and attention need <b>1</b>) · free parameters in the drawn matrix: ` +
      `<b>${which === "attn" ? "0 — computed, not stored" : DL.commas(free)}</b> · ` +
      `does the parameter count grow with sequence length? <b>${sel.grows}</b>.`;
  }
  [De, Fe].forEach(e => e.addEventListener("input", draw));
  Me.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 4 · #tr-svg — the state, step by step ═══════════ */
(function () {
  const svg = d3.select("#tr-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Te = document.getElementById("tr-T"), He = document.getElementById("tr-dh"),
    Re = document.getElementById("tr-rho"), Ie = document.getElementById("tr-inp"),
    We = document.getElementById("tr-what");

  function draw() {
    const T = +Te.value, dh = +He.value, rho = +Re.value, inp = Ie.value, what = We.value;
    document.getElementById("tr-Tv").textContent = T;
    document.getElementById("tr-dhv").textContent = dh;
    document.getElementById("tr-rhov").textContent = DL.fmt(rho, 2);
    const f = DL.frame(svg, W, H, { l: 40, r: 16, t: 22, b: 20 }), g = f.g;

    const dx = 3;
    const p = DL.CELL.vanilla.init(dx, dh, { seed: 12, rho: rho, uScale: 0.7 });
    const rhoAch = DL.specRad(p.W);
    let h0 = null;
    if (inp === "zero") { const r = DL.rng(4); h0 = []; for (let i = 0; i < dh; i++) h0.push(0.8 * DL.randn(r)); }
    const X = RN.seq(T, dx, 31, inp);
    const st = DL.CELL.vanilla.forward(p, X, h0);

    /* the field being drawn */
    const M = [];
    for (let t = 0; t < T; t++) {
      const h = st.H[t];
      if (what === "h") M.push(h.slice());
      else if (what === "sat") M.push(h.map(v => 1 - v * v));
      else {
        /* z recovered exactly: z = atanh(h) is safe because |h| < 1 strictly */
        M.push(h.map(v => Math.atanh(Math.min(0.999999999, Math.max(-0.999999999, v)))));
      }
    }
    const mx = Math.max(1e-9, Math.max(...M.map(r => Math.max(...r.map(Math.abs)))));

    /* ---- heat map ---- */
    const hw = f.iw - 150, cw = Math.min(16, hw / T), ch = Math.min(15, 150 / dh);
    RN.title(g, 0, 0, what === "h" ? "the state hₜ, one row per unit" : what === "z" ? "the pre-activation zₜ" : "the tanh derivative 1 − hₜ²");
    for (let t = 0; t < T; t++) for (let i = 0; i < dh; i++) {
      const v = M[t][i];
      g.append("rect").attr("x", t * cw).attr("y", 14 + i * ch).attr("width", cw + 0.4).attr("height", ch + 0.4)
        .attr("shape-rendering", "crispEdges")
        .attr("fill", what === "sat" ? RN.heat(Math.min(1, v)) : RN.signColor(v, mx));
    }
    g.append("rect").attr("x", 0).attr("y", 14).attr("width", T * cw).attr("height", dh * ch)
      .attr("fill", "none").attr("stroke", DC.line);
    RN.note(g, -6, 14 + dh * ch / 2, "units", DC.muted, 9.5);

    /* ---- norm against time ---- */
    const y0 = 14 + dh * ch + 34, gh = H - y0 - 54;
    const norms = st.H.map(h => RN.norm(h));
    const x = d3.scaleLinear().domain([1, Math.max(2, T)]).range([0, T * cw]);
    const yy = RN.logY(norms.concat([rhoAch ? Math.pow(rhoAch, T) * (norms[0] || 1) : 1]), gh);
    const gg = g.append("g").attr("transform", `translate(0,${y0})`);
    DL.gridY(gg, yy, T * cw, 4);
    DL.axisB(gg, x, gh, 6, "time step");
    DL.axisL(gg, yy, 4, "‖hₜ‖", d => DL.fmtE(d, 0));
    DL.curve(gg, norms.map((v, i) => [x(i + 1), yy(Math.max(v, 1e-300))]), { stroke: DC.accent, w: 2 });
    /* the free-decay reference is only MEANINGFUL when nothing is driving the
       state; a continuously driven state is held up by its input. */
    const undriven = (inp === "pulse" || inp === "zero");
    if (undriven) {
      const ref = [];
      for (let t = 1; t <= T; t++) ref.push([x(t), yy(Math.max(1e-300, (norms[0] || 1) * Math.pow(rhoAch, t - 1)))]);
      DL.curve(gg, ref, { stroke: DC.a2, w: 1.4, dash: "4 3" });
    }
    DL.legend(gg, [{ label: "measured ‖hₜ‖", color: DC.accent }].concat(undriven
      ? [{ label: "ρ(W)ᵗ reference", color: DC.a2, dash: "4 3" }]
      : [{ label: "no ρᵗ reference: the state is DRIVEN", color: DC.muted }]),
      6, gh + 32, { vertical: false, step: 190, font: 9.5 });

    /* ---- saturation panel ---- */
    const px = T * cw + 26;
    const dlast = st.H[T - 1].map(v => 1 - v * v);
    const satFrac = dlast.filter(v => v < 0.1).length / dh;
    const meanGate = d3.mean(st.H.map(h => d3.mean(h.map(v => 1 - v * v))));
    const kv = DL.kv(g, px, 24, { keyW: 96, size: 10.5, lead: 15 });
    kv("target ρ(W)", DL.fmt(rho, 3));
    kv("achieved ρ(W)", DL.fmt(rhoAch, 6), DC.good);
    kv("‖h₁‖", RN.P(norms[0], 4));
    kv("‖h_T‖", RN.P(norms[T - 1], 4));
    kv("ratio", RN.P(norms[T - 1] / Math.max(1e-300, norms[0]), 4), DC.a2);
    kv("per-step factor", DL.fmt(Math.pow(norms[T - 1] / Math.max(1e-300, norms[0]), 1 / Math.max(1, T - 1)), 4));
    kv("mean 1 − h²", DL.fmt(meanGate, 4));
    kv("saturated units", DL.fmt(100 * satFrac, 1) + "%", satFrac > 0.5 ? DC.bad : DC.ink);
    RN.note(g, px, 24 + 8 * 15 + 12, "‘saturated’ = 1 − h² below 0.1", DC.muted, 9);
    RN.note(g, px, 24 + 8 * 15 + 24, "at the FINAL step", DC.muted, 9);

    const drivenNote = (inp === "pulse" || inp === "zero")
      ? `a measured per-step factor of <b>${DL.fmt(Math.pow(norms[T - 1] / Math.max(1e-300, norms[0]), 1 / Math.max(1, T - 1)), 4)}</b> ` +
        `against ρ(W) = ${DL.fmt(rhoAch, 3)} times the mean tanh gate ${DL.fmt(meanGate, 3)} = <b>${DL.fmt(rhoAch * meanGate, 4)}</b>`
      : `the state is <b>continuously driven</b>, so it does not decay at the free rate at all and no ρᵗ reference is drawn — ` +
        `its norm settles at whatever balances the input against the recurrence`;
    document.getElementById("tr-readout").innerHTML =
      `ρ(W) requested <b>${DL.fmt(rho, 2)}</b>, achieved <b>${DL.fmt(rhoAch, 6)}</b> · ` +
      `‖h₁‖ = <b>${RN.P(norms[0], 4)}</b> → ‖h_T‖ = <b>${RN.P(norms[T - 1], 4)}</b> · ` + drivenNote +
      ` · mean tanh derivative <b>${DL.fmt(meanGate, 4)}</b>, ` +
      `<b>${DL.fmt(100 * satFrac, 1)}%</b> of units saturated at the last step.`;
  }
  [Te, He, Re].forEach(e => e.addEventListener("input", draw));
  [Ie, We].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 5 · #sp-svg — one step, every shape and every count ═══════════ */
(function () {
  const svg = d3.select("#sp-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const ids = ["sp-dx", "sp-dh", "sp-dy", "sp-T"].map(i => document.getElementById(i));
  const Ce = document.getElementById("sp-cell");

  function draw() {
    const dx = +ids[0].value, dh = +ids[1].value, dy = +ids[2].value, T = +ids[3].value;
    const cell = Ce.value, gates = DL.CELL[cell].nGate;
    ["sp-dxv", "sp-dhv", "sp-dyv", "sp-Tv"].forEach((i, k) => document.getElementById(i).textContent = [dx, dh, dy, T][k]);
    const f = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 16 }), g = f.g;

    const cp = DL.cellParams(cell, dx, dh, 1);
    const pV = dh * dy, pc = dy, total = cp.total + pV + pc;
    const macCell = T * gates * (dx * dh + dh * dh);
    const macOut = T * dh * dy;
    const macs = macCell + macOut;

    /* ---- the shape chain ---- */
    RN.title(g, 0, 0, "one step, as a shape chain");
    const boxes = [
      ["hₜ₋₁", `1 × ${dh}`, DC.a2], ["W", `${dh} × ${dh}`, DC.a2],
      ["xₜ", `1 × ${dx}`, DC.accent], ["U", `${dx} × ${dh}`, DC.accent],
      ["zₜ", `1 × ${dh}`, DC.ink], ["hₜ", `1 × ${dh}`, DC.a2],
      ["V", `${dh} × ${dy}`, DC.good], ["sₜ", `1 × ${dy}`, DC.good]
    ];
    const bw = 82, bh = 30, gap = 8;
    boxes.forEach((b, i) => {
      const col = i % 4, row = Math.floor(i / 4);
      const X = col * (bw + gap + 12), Y = 18 + row * (bh + 22);
      g.append("rect").attr("x", X).attr("y", Y).attr("width", bw).attr("height", bh)
        .attr("rx", 5).attr("fill", DC.panel2).attr("stroke", b[2]).attr("stroke-opacity", 0.7);
      g.append("text").attr("x", X + bw / 2).attr("y", Y + 13).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", DC.ink).text(b[0]);
      g.append("text").attr("x", X + bw / 2).attr("y", Y + 25).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", DC.muted).text(b[1]);
      if (col < 3) DL.arrow(g, X + bw + 1, Y + bh / 2, X + bw + gap + 10, Y + bh / 2, { color: DC.line, w: 1.2, head: 4 });
    });
    if (gates > 1) RN.note(g, 0, 122, `the ${cell === "lstm" ? "LSTM" : "GRU"} has ${gates} such (U, W) pairs, one per gate-like computation`, DC.a2, 10);

    /* ---- parameter split ---- */
    const py = 148, pw = 330;
    RN.title(g, 0, py - 8, "where the parameters are");
    const parts = [
      ["input U", cp.U, DC.accent], ["recurrent W", cp.W, DC.a2],
      ["biases", cp.b, DC.muted], ["read-out V, c", pV + pc, DC.good]
    ];
    let acc = 0;
    parts.forEach(pt => {
      const w = pw * pt[1] / total;
      g.append("rect").attr("x", acc).attr("y", py).attr("width", Math.max(0.5, w)).attr("height", 22)
        .attr("fill", pt[2]).attr("fill-opacity", 0.6).attr("stroke", pt[2]).attr("stroke-opacity", 0.9);
      acc += w;
    });
    parts.forEach((pt, i) => {
      const yy = py + 36 + i * 15;
      g.append("rect").attr("x", 0).attr("y", yy - 8).attr("width", 9).attr("height", 9).attr("rx", 2)
        .attr("fill", pt[2]).attr("fill-opacity", 0.75);
      RN.note(g, 14, yy, pt[0], DC.muted, 10);
      g.append("text").attr("x", 190).attr("y", yy).attr("text-anchor", "end").attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(DL.commas(pt[1]));
      g.append("text").attr("x", 240).attr("y", yy).attr("text-anchor", "end").attr("font-size", 10)
        .attr("fill", DC.muted).text(DL.fmt(100 * pt[1] / total, 1) + "%");
    });
    g.append("text").attr("x", 190).attr("y", py + 36 + 4 * 15 + 4).attr("text-anchor", "end")
      .attr("font-size", 11).attr("font-family", "SF Mono, Menlo, monospace")
      .attr("fill", DC.a2).attr("font-weight", 600).text(DL.commas(total));
    RN.note(g, 0, py + 36 + 4 * 15 + 4, "TOTAL", DC.a2, 10);

    /* ---- arithmetic split ---- */
    const ax = 400;
    RN.title(g, ax, py - 8, `arithmetic for one sequence of T = ${T}`);
    const aparts = [["recurrence, all steps", macCell, DC.a2], ["read-out, all steps", macOut, DC.good]];
    let acc2 = 0;
    aparts.forEach(pt => {
      const w = 300 * pt[1] / macs;
      g.append("rect").attr("x", ax + acc2).attr("y", py).attr("width", Math.max(0.5, w)).attr("height", 22)
        .attr("fill", pt[2]).attr("fill-opacity", 0.6).attr("stroke", pt[2]);
      acc2 += w;
    });
    const kv = DL.kv(g, ax, py + 46, { keyW: 176, size: 10.5, lead: 15 });
    kv("forward MACs", DL.big(macs));
    kv("backward MACs (≈ 2×)", DL.big(2 * macs));
    kv("read-out share of params", DL.fmt(100 * (pV + pc) / total, 1) + "%", (pV + pc) / total > 0.5 ? DC.bad : DC.ink);
    kv("states kept for backward", DL.big(T * dh * (cell === "lstm" ? 6 : cell === "gru" ? 5 : 1)));
    const bytes = T * dh * (cell === "lstm" ? 6 : cell === "gru" ? 5 : 1) * 4;
    kv("that, in float32", DL.fmt(bytes / 1048576, 3) + " MB / sequence");

    document.getElementById("sp-readout").innerHTML =
      `<b>${DL.CELL[cell].label}</b>, dₓ=${dx} dₕ=${dh} d_y=${dy} T=${T} · parameters <b>${DL.commas(total)}</b> ` +
      `(cell ${DL.commas(cp.total)}, read-out ${DL.commas(pV + pc)} = <b>${DL.fmt(100 * (pV + pc) / total, 1)}%</b>) · ` +
      `forward <b>${DL.big(macs)}</b> MACs per sequence, backward about <b>${DL.big(2 * macs)}</b> · ` +
      `activations to keep for the backward pass <b>${DL.fmt(bytes / 1048576, 3)} MB</b> per sequence.`;
  }
  ids.forEach(e => e.addEventListener("input", draw));
  Ce.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 6 · #wr-svg — the five wirings ═══════════ */
(function () {
  const svg = d3.select("#wr-svg");
  if (svg.empty()) return;
  const W = 760, H = 390;
  const Ke = document.getElementById("wr-k"), Te = document.getElementById("wr-T"),
    Se = document.getElementById("wr-k2");

  function draw() {
    const kind = Ke.value, T = +Te.value;
    Se.max = String(T);
    const kstep = Math.min(+Se.value, T);
    document.getElementById("wr-Tv").textContent = T;
    document.getElementById("wr-k2v").textContent = kstep;
    const f = DL.frame(svg, W, H, { l: 20, r: 200, t: 24, b: 20 }), g = f.g;

    const encdec = kind === "encdec";
    const nEnc = encdec ? Math.ceil(T / 2) : T, nDec = encdec ? T - nEnc : 0;
    const N = T;
    const step = Math.min(58, (f.iw - 30) / Math.max(1, N));
    const px = i => 16 + i * step + (encdec && i >= nEnc ? 22 : 0);
    const yy = { x: 250, h: 172, o: 90, L: 40 };
    const r = 12;

    const hasOut = t => {
      if (kind === "many2many" || kind === "out2hid") return true;
      if (kind === "many2one") return t === T - 1;
      if (kind === "one2many") return true;
      if (kind === "encdec") return t >= nEnc;
      return false;
    };
    const hasIn = t => {
      if (kind === "one2many") return t === 0;
      if (kind === "encdec") return t < nEnc;
      return true;
    };

    RN.title(g, 0, 0, {
      many2many: "sequence → sequence, hidden-to-hidden recurrence",
      out2hid: "sequence → sequence, recurrence only from the OUTPUT",
      many2one: "sequence → vector",
      one2many: "vector → sequence",
      encdec: "encoder → decoder, different lengths"
    }[kind]);

    /* edges first */
    for (let t = 0; t < N; t++) {
      const X = px(t);
      if (hasIn(t)) DL.arrow(g, X, yy.x - r - 1, X, yy.h + r + 1, { color: DC.accent, w: 1.2, head: 4, op: 0.9 });
      if (t > 0) {
        if (kind === "out2hid") {
          DL.arrow(g, px(t - 1), yy.o + r + 1, X - 4, yy.h - r - 2, { color: DC.violet, w: 1.4, head: 5 });
        } else if (encdec && t === nEnc) {
          DL.arrow(g, px(t - 1) + r + 1, yy.h, X - r - 1, yy.h, { color: DC.teal, w: 3, head: 7 });
          RN.note(g, (px(t - 1) + X) / 2 - 8, yy.h - 10, "C", DC.teal, 11);
        } else {
          DL.arrow(g, px(t - 1) + r + 1, yy.h, X - r - 1, yy.h, { color: DC.a2, w: 1.4, head: 5 });
        }
      }
      if (kind === "one2many" && t > 0) {
        DL.arrow(g, px(0), yy.x, X, yy.h + r + 4, { color: DC.accent, w: 0.8, head: 3, op: 0.35, dash: "2 2" });
      }
      if (hasOut(t)) {
        DL.arrow(g, X, yy.h - r - 1, X, yy.o + r + 1, { color: DC.good, w: 1.2, head: 4 });
        DL.arrow(g, X, yy.o - r - 1, X, yy.L + 8, { color: DC.bad, w: 1.1, head: 4 });
      }
    }
    /* nodes */
    for (let t = 0; t < N; t++) {
      const X = px(t);
      if (hasIn(t)) {
        g.append("circle").attr("cx", X).attr("cy", yy.x).attr("r", r)
          .attr("fill", DC.panel2).attr("stroke", DC.accent);
        g.append("text").attr("x", X).attr("y", yy.x + 3.5).attr("text-anchor", "middle")
          .attr("font-size", 8.5).attr("fill", DC.muted).text("x" + (t + 1));
      }
      g.append("circle").attr("cx", X).attr("cy", yy.h).attr("r", r + (t + 1 === kstep ? 3 : 0))
        .attr("fill", t + 1 === kstep ? DC.a2 : DC.panel2).attr("fill-opacity", t + 1 === kstep ? 0.35 : 1)
        .attr("stroke", DC.a2).attr("stroke-width", t + 1 === kstep ? 2.4 : 1.3);
      g.append("text").attr("x", X).attr("y", yy.h + 3.5).attr("text-anchor", "middle")
        .attr("font-size", 8.5).attr("fill", DC.ink).text("h" + (t + 1));
      if (hasOut(t)) {
        g.append("circle").attr("cx", X).attr("cy", yy.o).attr("r", r)
          .attr("fill", DC.panel2).attr("stroke", DC.good);
        g.append("text").attr("x", X).attr("y", yy.o + 3.5).attr("text-anchor", "middle")
          .attr("font-size", 8.5).attr("fill", DC.muted).text("o" + (t + 1));
        g.append("text").attr("x", X).attr("y", yy.L).attr("text-anchor", "middle")
          .attr("font-size", 9).attr("fill", DC.bad).text("L" + (t + 1));
      }
    }
    /* highlight the paths into kstep */
    const lossSteps = [];
    for (let t = 0; t < N; t++) if (hasOut(t)) lossSteps.push(t);
    const reach = lossSteps.filter(t => t >= kstep - 1);
    reach.forEach(t => {
      for (let u = kstep - 1; u < t; u++) {
        g.append("line").attr("x1", px(u)).attr("x2", px(u + 1)).attr("y1", yy.h).attr("y2", yy.h)
          .attr("stroke", DC.violet).attr("stroke-width", 3.2).attr("stroke-opacity", 0.35);
      }
      g.append("line").attr("x1", px(t)).attr("x2", px(t)).attr("y1", yy.o).attr("y2", yy.h)
        .attr("stroke", DC.violet).attr("stroke-width", 3.2).attr("stroke-opacity", 0.35);
    });

    /* side panel */
    const sx = f.iw + 14;
    const longest = reach.length ? (Math.max(...reach) - (kstep - 1) + 1) : 0;
    const shortest = reach.length ? (Math.min(...reach) - (kstep - 1) + 1) : 0;
    const par = kind === "out2hid";
    const kv = DL.kv(g, sx, 26, { keyW: 128, size: 10.5, lead: 16 });
    kv("loss terms", String(lossSteps.length));
    kv("paths into h" + kstep, String(reach.length), reach.length === 0 ? DC.bad : DC.ink);
    kv("shortest path", reach.length ? String(shortest) : "—");
    kv("longest path", reach.length ? String(longest) : "—", longest > 12 ? DC.bad : DC.ink);
    kv("parallel over time?", par ? "yes" : "no", par ? DC.good : DC.bad);
    RN.note(g, sx, 130, kind === "many2one"
      ? "one loss, at the very end: the ONLY" : "", DC.muted, 9.5);
    RN.note(g, sx, 142, kind === "many2one"
      ? "route to early steps is the chain." : "", DC.muted, 9.5);
    RN.note(g, sx, 154, kind === "out2hid"
      ? "under teacher forcing the steps" : "", DC.muted, 9.5);
    RN.note(g, sx, 166, kind === "out2hid"
      ? "decouple entirely." : "", DC.muted, 9.5);

    document.getElementById("wr-readout").innerHTML =
      `<b>${Ke.options[Ke.selectedIndex].text}</b> · ${lossSteps.length} loss term${lossSteps.length === 1 ? "" : "s"} · ` +
      `gradient paths into h${kstep}: <b>${reach.length}</b>` +
      (reach.length ? `, the longest travelling <b>${longest}</b> recurrent step${longest === 1 ? "" : "s"}` : " — no loss can reach it") +
      ` · trainable in parallel over time: <b>${par ? "yes" : "no"}</b>.`;
  }
  [Te, Se].forEach(e => e.addEventListener("input", draw));
  Ke.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 7 · #ls-svg — where the loss enters ═══════════ */
(function () {
  const svg = d3.select("#ls-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const Te = document.getElementById("ls-T"), We = document.getElementById("ls-w"),
    Re = document.getElementById("ls-rho"), De = document.getElementById("ls-red");

  function draw() {
    const T = +Te.value, where = We.value, rho = +Re.value, red = De.value;
    document.getElementById("ls-Tv").textContent = T;
    document.getElementById("ls-rhov").textContent = DL.fmt(rho, 2);
    const f = DL.frame(svg, W, H, { l: 48, r: 190, t: 40, b: 34 }), g = f.g;

    const dx = 3, dh = 8, dy = 2;
    const net = RN.net("vanilla", dx, dh, dy, { seed: 15, rho: rho, uScale: 0.6 });
    const X = RN.seq(T, dx, 41), Y = RN.targets(T, dy, "sigmoid", 43, where);
    const g1 = DL.seqBPTT(net, X, Y);
    const scale = red === "mean" ? 1 / T : 1;
    /* per-step gradient at the state, and per-step contribution to ∂L/∂W */
    const st = g1.st;
    const dHtot = [];
    {
      let dn = DL.zeros(dh);
      const per = new Array(T).fill(0), perW = new Array(T).fill(0);
      for (let t = T - 1; t >= 0; t--) {
        const dh_t = g1.dH[t].map((v, j) => v + dn[j]);
        per[t] = RN.norm(dh_t) * scale;
        const dz = dh_t.map((v, j) => v * (1 - st.H[t][j] * st.H[t][j]));
        perW[t] = RN.norm(dz) * RN.norm(st.Hprev[t]) * scale;
        dn = DL.vecmat(dz, DL.transpose(net.p.W));
      }
      dHtot.push(per, perW);
    }
    const per = dHtot[0], perW = dHtot[1];

    /* target strip */
    const x = d3.scaleBand().domain(d3.range(T)).range([0, f.iw]).padding(0.15);
    RN.title(g, 0, -22, "which steps carry a target");
    for (let t = 0; t < T; t++) {
      g.append("rect").attr("x", x(t)).attr("y", -14).attr("width", x.bandwidth()).attr("height", 9).attr("rx", 2)
        .attr("fill", Y[t] ? DC.bad : DC.line).attr("fill-opacity", Y[t] ? 0.85 : 0.5);
    }

    const all = per.concat(perW).filter(v => v > 0);
    const y = RN.logY(all, f.ih);
    DL.gridY(g, y, f.iw, 5);
    DL.axisB(g, d3.scaleLinear().domain([1, T]).range([0, f.iw]), f.ih, 8, "time step");
    DL.axisL(g, y, 5, "‖∂L/∂hₜ‖", d => DL.fmtE(d, 0));
    for (let t = 0; t < T; t++) {
      if (per[t] > 0) g.append("rect").attr("x", x(t)).attr("y", y(per[t])).attr("width", x.bandwidth())
        .attr("height", Math.max(0, f.ih - y(per[t]))).attr("fill", DC.accent).attr("fill-opacity", 0.55);
    }
    DL.curve(g, perW.map((v, i) => [x(i) + x.bandwidth() / 2, y(Math.max(v, 1e-300))]), { stroke: DC.a2, w: 1.8 });
    /* the ρᵈ reference, anchored at the last target */
    const lastY = per.map((v, i) => Y[i] ? i : -1).filter(i => i >= 0).pop();
    if (lastY !== undefined && lastY > 0) {
      const ref = [];
      for (let t = 0; t <= lastY; t++) ref.push([x(t) + x.bandwidth() / 2, y(Math.max(1e-300, per[lastY] * Math.pow(rho, lastY - t)))]);
      DL.curve(g, ref, { stroke: DC.muted, w: 1.2, dash: "4 3" });
    }
    DL.legend(g, [{ label: "‖∂L/∂hₜ‖", color: DC.accent },
    { label: "step's share of ∂L/∂W", color: DC.a2 },
    { label: "ρ(W) decay reference", color: DC.muted, dash: "4 3" }],
      4, f.ih + 30, { vertical: false, step: 172, font: 9.5 });

    const pos = per.filter(v => v > 0);
    const ratio = pos.length ? Math.max(...pos) / Math.min(...pos) : 1;
    const mx = Math.max(...per);
    let cut = -1;
    for (let t = 0; t < T; t++) if (per[t] > mx * 1e-3) { cut = t; break; }
    const sx = f.iw + 16;
    const kv = DL.kv(g, sx, 10, { keyW: 118, size: 10.5, lead: 16 });
    kv("targets", String(Y.filter(v => v).length) + " / " + T);
    kv("largest ‖∂L/∂hₜ‖", RN.P(Math.max(...per), 3));
    kv("smallest", RN.P(Math.min(...pos.length ? pos : [0]), 3));
    kv("ratio", RN.P(ratio, 3), ratio > 1e4 ? DC.bad : DC.ink);
    kv("first step above 10⁻³ ×", cut >= 0 ? String(cut + 1) : "—");
    kv("reduction", red === "mean" ? "mean over steps" : "sum over steps");
    kv("total ‖∂L/∂W‖", RN.P(RN.fnorm(g1.dW) * scale, 4), DC.a2);
    RN.note(g, sx, 130, "switching sum ↔ mean rescales", DC.muted, 9.5);
    RN.note(g, sx, 142, "every bar by exactly 1/T and", DC.muted, 9.5);
    RN.note(g, sx, 154, "changes no relative size.", DC.muted, 9.5);

    document.getElementById("ls-readout").innerHTML =
      `${Y.filter(v => v).length} of ${T} steps carry a target · largest per-step gradient <b>${RN.P(Math.max(...per), 3)}</b>, ` +
      `smallest <b>${RN.P(pos.length ? Math.min(...pos) : 0, 3)}</b> — a ratio of <b>${RN.P(ratio, 3)}</b> · ` +
      (cut === 0 ? `<b>no</b> step receives less than a thousandth of the maximum — with a target everywhere the profile is nearly flat · `
        : `every step before step <b>${cut + 1}</b> receives less than a thousandth of the maximum · `) +
      `‖∂L/∂W‖ = <b>${RN.P(RN.fnorm(g1.dW) * scale, 4)}</b> under the ${red === "mean" ? "mean" : "sum"} reduction.`;
  }
  [Te, Re].forEach(e => e.addEventListener("input", draw));
  [We, De].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 8 · #dv-svg — ∂L/∂V as a sum of outer products ═══════════ */
(function () {
  const svg = d3.select("#dv-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Ae = document.getElementById("dv-t"), Te = document.getElementById("dv-T"),
    Oe = document.getElementById("dv-out"), Se = document.getElementById("dv-s");

  function draw() {
    const T = +Te.value;
    Ae.max = String(T);
    const upto = Math.min(+Ae.value, T), out = Oe.value, seed = +Se.value;
    document.getElementById("dv-Tv").textContent = T;
    document.getElementById("dv-tv").textContent = upto;
    document.getElementById("dv-sv").textContent = seed;
    const f = DL.frame(svg, W, H, { l: 14, r: 14, t: 24, b: 16 }), g = f.g;

    const dx = 3, dh = 4, dy = 3;
    const net = RN.net("vanilla", dx, dh, dy, { seed: seed, out: out, rho: 0.9 });
    const X = RN.seq(T, dx, seed + 60), Y = RN.targets(T, dy, out, seed + 70);
    const st = DL.seqForward(net, X);
    const hd = DL.seqHead(net, st, Y);

    /* running accumulator */
    const acc = DL.zeros2(dh, dy);
    for (let t = 0; t < upto; t++) for (let a = 0; a < dh; a++) for (let j = 0; j < dy; j++)
      acc[a][j] += st.H[t][a] * hd.dS[t][j];

    /* the full analytic and the finite-difference version */
    const full = DL.seqBPTT(net, X, Y, st).dV;
    const num = DL.seqNumGrad(net, X, Y, 1e-5).dV;

    /* ---- left: the terms ---- */
    RN.title(g, 0, 0, "the last few terms, hₜᵀ · (aₜ − yₜ)");
    const cw = 13, showT = Math.min(upto, 5), first = Math.max(0, upto - showT);
    for (let k = 0; k < showT; k++) {
      const t = first + k, yy = 18 + k * (dh * cw + 16);
      RN.note(g, 0, yy - 3, "t = " + (t + 1), DC.muted, 9.5);
      for (let a = 0; a < dh; a++) {
        g.append("rect").attr("x", 34).attr("y", yy + a * cw).attr("width", cw).attr("height", cw)
          .attr("fill", RN.signColor(st.H[t][a], 1)).attr("stroke", DC.line).attr("stroke-opacity", 0.4);
      }
      RN.note(g, 34, yy - 3, "hₜᵀ", DC.a2, 9);
      for (let j = 0; j < dy; j++) {
        g.append("rect").attr("x", 60 + j * cw).attr("y", yy).attr("width", cw).attr("height", cw)
          .attr("fill", RN.signColor(hd.dS[t][j], 1)).attr("stroke", DC.line).attr("stroke-opacity", 0.4);
      }
      RN.note(g, 60, yy - 3, "δₜ", DC.bad, 9);
      for (let a = 0; a < dh; a++) for (let j = 0; j < dy; j++) {
        g.append("rect").attr("x", 116 + j * cw).attr("y", yy + a * cw).attr("width", cw).attr("height", cw)
          .attr("fill", RN.signColor(st.H[t][a] * hd.dS[t][j], 0.6)).attr("stroke", DC.line).attr("stroke-opacity", 0.4);
      }
      RN.note(g, 116, yy - 3, "outer product", DC.muted, 9);
    }

    /* ---- centre: accumulator, FD, difference ---- */
    const cx = 250, big = 28;
    const grid = (M, ox, oy, lab, col) => {
      RN.title(g, ox, oy - 8, lab, col);
      const mm = Math.max(1e-12, Math.max(...RN.flat(M).map(Math.abs)));
      for (let a = 0; a < M.length; a++) for (let j = 0; j < M[0].length; j++) {
        g.append("rect").attr("x", ox + j * big).attr("y", oy + a * big).attr("width", big - 1).attr("height", big - 1)
          .attr("rx", 2).attr("fill", RN.signColor(M[a][j], mm)).attr("stroke", DC.line).attr("stroke-opacity", 0.5);
        g.append("text").attr("x", ox + j * big + big / 2 - 0.5).attr("y", oy + a * big + big / 2 + 3)
          .attr("text-anchor", "middle").attr("font-size", 8)
          .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink)
          .text(Math.abs(M[a][j]) < 1e-4 ? "0" : DL.fmt(M[a][j], 2));
      }
    };
    grid(acc, cx, 20, `accumulated over ${upto} step${upto === 1 ? "" : "s"}`, DC.a2);
    grid(num, cx, 20 + dh * big + 40, "central differences on the whole loss", DC.good);
    const diff = acc.map((r, a) => r.map((v, j) => v - num[a][j]));
    grid(diff, cx, 20 + 2 * (dh * big + 40), upto === T ? "difference (should be round-off)" : "difference (partial sum, so not yet zero)", upto === T ? DC.good : DC.muted);

    /* ---- right: numbers ---- */
    const sx = cx + dy * big + 40;
    let worst = 0;
    for (let a = 0; a < dh; a++) for (let j = 0; j < dy; j++) {
      const e = Math.abs(full[a][j] - num[a][j]) / Math.max(1e-8, Math.abs(full[a][j]), Math.abs(num[a][j]));
      worst = Math.max(worst, e);
    }
    const kv = DL.kv(g, sx, 30, { keyW: 174, size: 10.5, lead: 16 });
    kv("terms accumulated", upto + " of " + T);
    kv("‖partial sum‖_F", RN.P(RN.fnorm(acc), 4));
    kv("‖complete ∂L/∂V‖_F", RN.P(RN.fnorm(full), 4), DC.a2);
    kv("max |analytic − numeric|", DL.fmtE(Math.max(...RN.flat(full.map((r, a) => r.map((v, j) => Math.abs(v - num[a][j]))))), 2));
    kv("worst relative error", DL.fmtE(worst, 2), worst < 1e-5 ? DC.good : DC.bad);
    kv("∂L/∂c = Σₜ δₜ", "[" + DL.seqBPTT(net, X, Y, st).dc.map(v => DL.fmt(v, 3)).join(", ") + "]");
    RN.note(g, sx, 140, "V is OUTSIDE the recurrence, so its", DC.muted, 9.5);
    RN.note(g, sx, 152, "gradient is a plain sum of outer", DC.muted, 9.5);
    RN.note(g, sx, 164, "products with no travel at all.", DC.muted, 9.5);
    RN.note(g, sx, 184, "read-out: " + Oe.options[Oe.selectedIndex].text, DC.a2, 9.5);
    RN.note(g, sx, 196, "∂Lₜ/∂sₜ = aₜ − yₜ for all three.", DC.muted, 9.5);

    document.getElementById("dv-readout").innerHTML =
      `${upto} of ${T} terms accumulated · ‖partial sum‖ = <b>${RN.P(RN.fnorm(acc), 4)}</b>, ` +
      `‖complete ∂L/∂V‖ = <b>${RN.P(RN.fnorm(full), 4)}</b> · ` +
      `worst relative disagreement with central differences over all ${dh * dy} entries: <b>${DL.fmtE(worst, 2)}</b> · ` +
      `read-out <b>${Oe.options[Oe.selectedIndex].text}</b>, and the same expression covers all three.`;
  }
  [Ae, Te, Se].forEach(e => e.addEventListener("input", draw));
  Oe.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 9 · #pt-svg — every path from a loss to a use of W ═══════════ */
(function () {
  const svg = d3.select("#pt-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const Te = document.getElementById("pt-T"), Le = document.getElementById("pt-t"),
    Ie = document.getElementById("pt-i"), Re = document.getElementById("pt-rho"),
    Ae = document.getElementById("pt-all");

  /* ONE term of §09's double sum: h_{i−1}ᵀ · δₜVᵀ · (Dₜ Wᵀ … D_{i+1} Wᵀ) · Dᵢ  */
  function term(net, st, hd, t, i) {
    let u = hd.dH[t].slice();                 // = δₜ Vᵀ
    for (let j = t; j > i; j--) {
      const gz = u.map((v, k) => v * (1 - st.H[j][k] * st.H[j][k]));
      u = DL.vecmat(gz, DL.transpose(net.p.W));
    }
    const g = u.map((v, k) => v * (1 - st.H[i][k] * st.H[i][k]));
    const M = DL.zeros2(net.dh, net.dh), hp = st.Hprev[i];
    for (let a = 0; a < net.dh; a++) for (let b = 0; b < net.dh; b++) M[a][b] = hp[a] * g[b];
    return M;
  }

  function draw() {
    const T = +Te.value;
    Le.max = String(T); Ie.max = String(T);
    const t = Math.min(+Le.value, T), i = Math.min(+Ie.value, t);
    const rho = +Re.value, all = Ae.checked;
    document.getElementById("pt-Tv").textContent = T;
    document.getElementById("pt-tv").textContent = t;
    document.getElementById("pt-iv").textContent = i;
    document.getElementById("pt-rhov").textContent = DL.fmt(rho, 2);
    const f = DL.frame(svg, W, H, { l: 26, r: 16, t: 22, b: 16 }), g = f.g;

    const dx = 3, dh = 4, dy = 2;
    const net = RN.net("vanilla", dx, dh, dy, { seed: 6, rho: rho, uScale: 0.6 });
    const X = RN.seq(T, dx, 81), Y = RN.targets(T, dy, "sigmoid", 83);
    const st = DL.seqForward(net, X), hd = DL.seqHead(net, st, Y);

    /* ---- graph ---- */
    const step = Math.min(64, (f.iw - 40) / Math.max(1, T));
    const px = k => 22 + k * step;
    const yy = { x: 168, h: 116, o: 66, L: 34 };
    const r = 11;
    for (let k = 0; k < T; k++) {
      const Xp = px(k);
      DL.arrow(g, Xp, yy.x - r - 1, Xp, yy.h + r + 1, { color: DC.line, w: 1, head: 3, op: 0.6 });
      if (k > 0) DL.arrow(g, px(k - 1) + r + 1, yy.h, Xp - r - 1, yy.h, { color: DC.line, w: 1.1, head: 4, op: 0.6 });
      DL.arrow(g, Xp, yy.h - r - 1, Xp, yy.o + r + 1, { color: DC.line, w: 1, head: 3, op: 0.6 });
      DL.arrow(g, Xp, yy.o - r - 1, Xp, yy.L + 6, { color: DC.line, w: 1, head: 3, op: 0.6 });
    }
    if (all) {
      for (let tt = 0; tt < T; tt++) for (let ii = 0; ii <= tt; ii++) {
        const nrm = RN.fnorm(term(net, st, hd, tt, ii));
        const op = Math.min(0.5, 0.5 * nrm / Math.max(1e-12, RN.fnorm(term(net, st, hd, tt, tt))));
        for (let u = ii; u < tt; u++)
          g.append("line").attr("x1", px(u)).attr("x2", px(u + 1)).attr("y1", yy.h).attr("y2", yy.h)
            .attr("stroke", DC.violet).attr("stroke-width", 4).attr("stroke-opacity", op);
      }
    } else {
      g.append("line").attr("x1", px(t - 1)).attr("x2", px(t - 1)).attr("y1", yy.L + 8).attr("y2", yy.h)
        .attr("stroke", DC.violet).attr("stroke-width", 3.4).attr("stroke-opacity", 0.75);
      for (let u = i - 1; u < t - 1; u++)
        g.append("line").attr("x1", px(u)).attr("x2", px(u + 1)).attr("y1", yy.h).attr("y2", yy.h)
          .attr("stroke", DC.violet).attr("stroke-width", 3.4).attr("stroke-opacity", 0.75);
      g.append("circle").attr("cx", px(i - 1)).attr("cy", yy.h).attr("r", r + 5)
        .attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 2);
      RN.note(g, px(i - 1) - 14, yy.h + 30, "use of W", DC.a2, 9);
    }
    for (let k = 0; k < T; k++) {
      const Xp = px(k);
      [[yy.x, DC.accent, "x"], [yy.h, DC.a2, "h"], [yy.o, DC.good, "o"]].forEach(d => {
        g.append("circle").attr("cx", Xp).attr("cy", d[0]).attr("r", r)
          .attr("fill", DC.panel2).attr("stroke", d[1]).attr("stroke-width", 1.2);
        g.append("text").attr("x", Xp).attr("y", d[0] + 3.5).attr("text-anchor", "middle")
          .attr("font-size", 8).attr("fill", DC.muted).text(d[2] + (k + 1));
      });
      g.append("text").attr("x", Xp).attr("y", yy.L).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("fill", k + 1 === t && !all ? DC.violet : DC.bad)
        .attr("font-weight", k + 1 === t && !all ? 600 : 400).text("L" + (k + 1));
    }

    /* ---- the factors of the chosen term ---- */
    const travel = t - i;
    const factors = ["h" + (i - 1 > 0 ? i - 1 : "₀") + "ᵀ", "δ" + t, "Vᵀ"];
    for (let j = t; j > i; j--) { factors.push("D" + j); factors.push("Wᵀ"); }
    factors.push("D" + i);
    RN.note(g, 0, 200, "the chosen term  =  " + factors.join(" · "), DC.ink, 11);
    RN.note(g, 0, 214, `it contains Wᵀ exactly ${travel} time${travel === 1 ? "" : "s"} — one per recurrent step travelled`,
      travel > 0 ? DC.a2 : DC.muted, 10);

    /* ---- term-magnitude triangle ---- */
    const oy = 236, cw = Math.min(20, (f.iw - 250) / T);
    RN.title(g, 0, oy - 8, "‖term‖ for every (loss step, use step) pair — the lower triangle of §09's double sum");
    const norms = [], all2 = [];
    for (let tt = 1; tt <= T; tt++) {
      const row = [];
      for (let ii = 1; ii <= T; ii++) row.push(ii <= tt ? RN.fnorm(term(net, st, hd, tt - 1, ii - 1)) : NaN);
      norms.push(row); row.forEach(v => { if (isFinite(v) && v > 0) all2.push(v); });
    }
    const lo = all2.length ? Math.min(...all2) : 1e-12, hi = all2.length ? Math.max(...all2) : 1;
    const cs = d3.scaleLog().domain([Math.max(lo, hi * 1e-10), hi]).range([0, 1]).clamp(true);
    for (let tt = 0; tt < T; tt++) for (let ii = 0; ii < T; ii++) {
      const v = norms[tt][ii];
      const el = g.append("rect").attr("x", 30 + ii * cw).attr("y", oy + tt * cw)
        .attr("width", cw - 1).attr("height", cw - 1).attr("rx", 1)
        .attr("shape-rendering", "crispEdges");
      if (!isFinite(v)) el.attr("fill", DC.panel).attr("fill-opacity", 0.2);
      else el.attr("fill", RN.heat(cs(Math.max(v, 1e-300))));
      if (!all && tt + 1 === t && ii + 1 === i)
        g.append("rect").attr("x", 30 + ii * cw - 1).attr("y", oy + tt * cw - 1)
          .attr("width", cw + 1).attr("height", cw + 1).attr("fill", "none")
          .attr("stroke", DC.ink).attr("stroke-width", 2);
    }
    RN.note(g, 30, oy - 20, "use step i →", DC.muted, 9);
    g.append("text").attr("transform", `translate(20,${oy + T * cw / 2}) rotate(-90)`)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", DC.muted).text("loss step t");

    const chosen = RN.fnorm(term(net, st, hd, t - 1, i - 1));
    let tot = 0;
    const sum = DL.zeros2(dh, dh);
    for (let tt = 0; tt < T; tt++) for (let ii = 0; ii <= tt; ii++) {
      const M = term(net, st, hd, tt, ii);
      for (let a = 0; a < dh; a++) for (let b = 0; b < dh; b++) sum[a][b] += M[a][b];
      tot++;
    }
    const totalNorm = RN.fnorm(sum);
    const sx = 30 + T * cw + 26;
    const kv = DL.kv(g, sx, oy + 12, { keyW: 152, size: 10.5, lead: 16 });
    kv("terms in the double sum", String(T * (T + 1) / 2));
    kv("this term ‖·‖_F", RN.P(chosen, 4));
    kv("share of ‖∂L/∂W‖", DL.fmt(100 * chosen / Math.max(1e-300, totalNorm), 3) + "%");
    kv("recurrent factors in it", String(travel), travel > 8 ? DC.bad : DC.ink);
    kv("‖∂L/∂W‖ (all terms)", RN.P(totalNorm, 4), DC.a2);
    const diag = RN.fnorm(term(net, st, hd, T - 1, T - 1));
    kv("this term / the i = t term", DL.fmtE(chosen / Math.max(1e-300, diag), 2));

    const zeroNote = (i === 1)
      ? ` <b>This term is exactly zero</b>, and so is the whole of column 1: h₀ = 0, so the outer product h₀ᵀgₜ vanishes and the use of W at step 1 contributes nothing to its gradient, whatever the loss is.`
      : "";
    document.getElementById("pt-readout").innerHTML =
      `loss at step <b>${t}</b>, use of W at step <b>${i}</b> — the term travels <b>${travel}</b> recurrent step${travel === 1 ? "" : "s"} ` +
      `and contains <code>Wᵀ</code> <b>${travel}</b> time${travel === 1 ? "" : "s"} · ‖term‖ = <b>${RN.P(chosen, 4)}</b>, ` +
      `which is <b>${DL.fmt(100 * chosen / Math.max(1e-300, totalNorm), 3)}%</b> of ‖∂L/∂W‖ = ${RN.P(totalNorm, 4)} · ` +
      `the double sum has <b>${T * (T + 1) / 2}</b> terms at T = ${T}.` + zeroNote;
  }
  [Te, Le, Ie, Re].forEach(e => e.addEventListener("input", draw));
  Ae.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 10 · #du-svg — the three gradients and the shared factor ═══════════ */
(function () {
  const svg = d3.select("#du-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const Te = document.getElementById("du-T"), Re = document.getElementById("du-rho"),
    We = document.getElementById("du-w"), Se = document.getElementById("du-share");

  function draw() {
    const T = +Te.value, rho = +Re.value, which = We.value, share = Se.checked;
    document.getElementById("du-Tv").textContent = T;
    document.getElementById("du-rhov").textContent = DL.fmt(rho, 2);
    const f = DL.frame(svg, W, H, { l: 50, r: 196, t: 26, b: 34 }), g = f.g;

    const dx = 3, dh = 6, dy = 2;
    const net = RN.net("vanilla", dx, dh, dy, { seed: 19, rho: rho, uScale: 0.6 });
    const X = RN.seq(T, dx, 91), Y = RN.targets(T, dy, "sigmoid", 93);
    const gr = DL.seqBPTT(net, X, Y), st = gr.st;

    const perU = [], perW = [], perB = [], perZ = [];
    {
      let dn = DL.zeros(dh);
      const rows = [];
      for (let t = T - 1; t >= 0; t--) {
        const dh_t = gr.dH[t].map((v, j) => v + dn[j]);
        const dz = dh_t.map((v, j) => v * (1 - st.H[t][j] * st.H[t][j]));
        rows[t] = { z: RN.norm(dz), U: RN.norm(dz) * RN.norm(X[t]), W: RN.norm(dz) * RN.norm(st.Hprev[t]), b: RN.norm(dz) };
        dn = DL.vecmat(dz, DL.transpose(net.p.W));
      }
      rows.forEach(r => { perU.push(r.U); perW.push(r.W); perB.push(r.b); perZ.push(r.z); });
    }
    const series = [
      { k: "U", v: perU, c: DC.accent, lab: "∂L/∂U per step" },
      { k: "W", v: perW, c: DC.a2, lab: "∂L/∂W per step" },
      { k: "b", v: perB, c: DC.good, lab: "∂L/∂b per step" }
    ].filter(s => which === "all" || which === s.k);

    const allv = series.reduce((a, s) => a.concat(s.v), []).concat(share ? perZ : []).filter(v => v > 0);
    const y = RN.logY(allv, f.ih);
    const x = d3.scaleBand().domain(d3.range(T)).range([0, f.iw]).padding(0.18);
    DL.gridY(g, y, f.iw, 5);
    DL.axisB(g, d3.scaleLinear().domain([1, T]).range([0, f.iw]), f.ih, 8, "time step");
    DL.axisL(g, y, 5, "‖contribution‖", d => DL.fmtE(d, 0));
    const nb = series.length;
    series.forEach((s, si) => {
      for (let t = 0; t < T; t++) {
        if (!(s.v[t] > 0)) continue;
        g.append("rect").attr("x", x(t) + si * x.bandwidth() / nb).attr("y", y(s.v[t]))
          .attr("width", x.bandwidth() / nb).attr("height", Math.max(0, f.ih - y(s.v[t])))
          .attr("fill", s.c).attr("fill-opacity", 0.62);
      }
    });
    if (share) DL.curve(g, perZ.map((v, i) => [x(i) + x.bandwidth() / 2, y(Math.max(v, 1e-300))]),
      { stroke: DC.violet, w: 2, dash: "5 3" });
    DL.legend(g, series.map(s => ({ label: s.lab, color: s.c }))
      .concat(share ? [{ label: "the shared factor ‖∂L/∂zₜ‖", color: DC.violet, dash: "5 3" }] : []),
      4, f.ih + 30, { vertical: false, step: 148, font: 9.5 });

    const sx = f.iw + 16;
    const kv = DL.kv(g, sx, 6, { keyW: 118, size: 10.5, lead: 16 });
    kv("‖∂L/∂U‖", RN.P(RN.fnorm(gr.dU), 4), DC.accent);
    kv("‖∂L/∂W‖", RN.P(RN.fnorm(gr.dW), 4), DC.a2);
    kv("‖∂L/∂b‖", RN.P(RN.norm(gr.db), 4), DC.good);
    kv("‖∂L/∂V‖", RN.P(RN.fnorm(gr.dV), 4), DC.muted);
    const posW = perW.filter(v => v > 0);
    const spread = posW.length ? Math.max(...posW) / Math.min(...posW) : 1;
    kv("max/min per-step (W)", RN.P(spread, 3), spread > 1e4 ? DC.bad : DC.ink);
    const mx = Math.max(...perZ);
    let cut = 0; for (let t = 0; t < T; t++) if (perZ[t] > mx * 0.01) { cut = t + 1; break; }
    kv("first step above 1% of max", String(cut));
    kv("measured per-step rate", DL.fmt(RN.logSlope(d3.range(T), perZ), 4), DC.violet);
    kv("ρ(W) actually used", DL.fmt(DL.specRad(net.p.W), 4));
    RN.note(g, sx, 156, "the three bar series are the violet", DC.muted, 9.5);
    RN.note(g, sx, 168, "line scaled by ‖xₜ‖, ‖hₜ₋₁‖ and 1.", DC.muted, 9.5);

    document.getElementById("du-readout").innerHTML =
      `‖∂L/∂U‖ <b>${RN.P(RN.fnorm(gr.dU), 4)}</b> · ‖∂L/∂W‖ <b>${RN.P(RN.fnorm(gr.dW), 4)}</b> · ` +
      `‖∂L/∂b‖ <b>${RN.P(RN.norm(gr.db), 4)}</b> · within ∂L/∂W the largest per-step contribution is ` +
      `<b>${RN.P(spread, 3)}×</b> the smallest · the shared factor ‖∂L/∂zₜ‖ decays at a measured ` +
      `<b>${DL.fmt(RN.logSlope(d3.range(T), perZ), 4)}</b> per step, against ρ(W) = ${DL.fmt(DL.specRad(net.p.W), 4)} · ` +
      (cut <= 1 ? `with a target at every step <b>no</b> step contributes under 1% of the maximum — the profile is nearly flat, ` +
        `and it is the sequence-to-vector wiring of §06 that produces the geometric decay.`
        : `every step before step <b>${cut}</b> contributes under 1% of the maximum.`);
  }
  [Te, Re].forEach(e => e.addEventListener("input", draw));
  We.addEventListener("change", draw);
  Se.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 11 · #rc-svg — the double sum against the recursion ═══════════ */
(function () {
  const svg = d3.select("#rc-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Te = document.getElementById("rc-T"), He = document.getElementById("rc-dh"),
    We = document.getElementById("rc-w"), Se = document.getElementById("rc-s");

  /* both computed here with REAL operation counters — page-specific, so local */
  function doubleSum(net, st, hd, want) {
    const dh = net.dh, dx = net.dx, T = st.T;
    let ops = 0;
    const acc = want === "b" ? DL.zeros(dh) : DL.zeros2(want === "U" ? dx : dh, dh);
    for (let t = 0; t < T; t++) {
      for (let i = t; i >= 0; i--) {
        /* recompute the whole travel from scratch — that is the point */
        let u = hd.dH[t].slice();
        for (let j = t; j > i; j--) {
          const gz = u.map((v, k) => v * (1 - st.H[j][k] * st.H[j][k]));
          ops += dh;
          u = DL.vecmat(gz, DL.transpose(net.p.W));
          ops += dh * dh;
        }
        const g = u.map((v, k) => v * (1 - st.H[i][k] * st.H[i][k]));
        ops += dh;
        if (want === "b") { for (let k = 0; k < dh; k++) acc[k] += g[k]; ops += dh; }
        else {
          const src = want === "U" ? st.X[i] : st.Hprev[i];
          for (let a = 0; a < src.length; a++) for (let k = 0; k < dh; k++) acc[a][k] += src[a] * g[k];
          ops += src.length * dh;
        }
      }
    }
    return { M: acc, ops: ops };
  }
  function recursion(net, st, hd, want) {
    const dh = net.dh, dx = net.dx, T = st.T;
    let ops = 0, dn = DL.zeros(dh);
    const acc = want === "b" ? DL.zeros(dh) : DL.zeros2(want === "U" ? dx : dh, dh);
    for (let t = T - 1; t >= 0; t--) {
      const dht = hd.dH[t].map((v, j) => v + dn[j]); ops += dh;
      const gz = dht.map((v, j) => v * (1 - st.H[t][j] * st.H[t][j])); ops += dh;
      if (want === "b") { for (let k = 0; k < dh; k++) acc[k] += gz[k]; ops += dh; }
      else {
        const src = want === "U" ? st.X[t] : st.Hprev[t];
        for (let a = 0; a < src.length; a++) for (let k = 0; k < dh; k++) acc[a][k] += src[a] * gz[k];
        ops += src.length * dh;
      }
      dn = DL.vecmat(gz, DL.transpose(net.p.W)); ops += dh * dh;
    }
    return { M: acc, ops: ops };
  }

  function draw() {
    const T = +Te.value, dh = +He.value, want = We.value, seed = +Se.value;
    document.getElementById("rc-Tv").textContent = T;
    document.getElementById("rc-dhv").textContent = dh;
    document.getElementById("rc-sv").textContent = seed;
    const f = DL.frame(svg, W, H, { l: 14, r: 14, t: 24, b: 16 }), g = f.g;

    const dx = 3, dy = 2;
    const net = RN.net("vanilla", dx, dh, dy, { seed: seed, rho: 0.95, uScale: 0.6 });
    const X = RN.seq(T, dx, seed + 200), Y = RN.targets(T, dy, "sigmoid", seed + 210);
    const st = DL.seqForward(net, X), hd = DL.seqHead(net, st, Y);
    const A = doubleSum(net, st, hd, want), B = recursion(net, st, hd, want);
    const M1 = want === "b" ? [A.M] : A.M, M2 = want === "b" ? [B.M] : B.M;
    const D = M1.map((r, a) => r.map((v, j) => v - M2[a][j]));
    const maxD = Math.max(...RN.flat(D).map(Math.abs));

    const cw = Math.min(26, 190 / Math.max(M1[0].length, 1));
    const grid = (M, ox, lab, col) => {
      RN.title(g, ox, 0, lab, col);
      const mm = Math.max(1e-14, Math.max(...RN.flat(M).map(Math.abs)));
      for (let a = 0; a < M.length; a++) for (let j = 0; j < M[0].length; j++) {
        g.append("rect").attr("x", ox + j * cw).attr("y", 12 + a * cw).attr("width", cw - 1).attr("height", cw - 1)
          .attr("rx", 1).attr("fill", RN.signColor(M[a][j], mm)).attr("stroke", DC.line).attr("stroke-opacity", 0.4);
      }
    };
    grid(M1, 0, "the literal double sum", DC.bad);
    grid(M2, 226, "the O(T) recursion", DC.good);
    grid(D, 452, "difference", DC.muted);
    RN.note(g, 452, 12 + M1.length * cw + 16, "max |Δ| = " + DL.fmtE(maxD, 2), maxD < 1e-10 ? DC.good : DC.bad, 11);
    RN.note(g, 452, 12 + M1.length * cw + 30, "double-precision round-off is ~1e−16 per operation", DC.muted, 9);

    /* ---- op counts and the growth curve ---- */
    const oy = Math.max(150, 12 + M1.length * cw + 56);
    RN.title(g, 0, oy - 8, "scalar multiply-accumulates actually performed, counted as the code ran");
    const bw = 300;
    const sc = d3.scaleLog().domain([Math.min(A.ops, B.ops) / 2, Math.max(A.ops, B.ops) * 1.3]).range([4, bw]);
    [["double sum", A.ops, DC.bad], ["recursion", B.ops, DC.good]].forEach((d, i) => {
      const yy = oy + i * 30;
      g.append("rect").attr("x", 96).attr("y", yy).attr("width", sc(d[1])).attr("height", 18).attr("rx", 3)
        .attr("fill", d[2]).attr("fill-opacity", 0.55).attr("stroke", d[2]);
      RN.note(g, 0, yy + 13, d[0], DC.muted, 10.5);
      g.append("text").attr("x", 96 + sc(d[1]) + 8).attr("y", yy + 13).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(DL.commas(d[1]));
    });
    g.append("text").attr("x", 96).attr("y", oy + 76).attr("font-size", 11).attr("fill", DC.a2)
      .attr("font-weight", 600).text("ratio  " + DL.fmt(A.ops / B.ops, 2) + "×");

    /* growth curve */
    const gx = 470, gw = f.iw - gx, gh = 110;
    const Ts = d3.range(2, 33);
    const oa = [], ob = [];
    Ts.forEach(tt => {
      /* closed forms of the two counters, verified against the live counts */
      const nUse = tt * (tt + 1) / 2;
      const travel = Ts.length ? (tt * (tt * tt - 1) / 6) : 0;   // Σₜ Σᵢ (t−i)
      const srcw = want === "U" ? dx : dh;
      oa.push(travel * (dh + dh * dh) + nUse * (dh + (want === "b" ? dh : srcw * dh)));
      ob.push(tt * (dh + dh + dh * dh + (want === "b" ? dh : srcw * dh)));
    });
    const gg = g.append("g").attr("transform", `translate(${gx},${oy - 4})`);
    const xs = d3.scaleLinear().domain([2, 32]).range([0, gw - 10]);
    const ys = d3.scaleLog().domain([Math.min(...ob) / 1.5, Math.max(...oa) * 1.5]).range([gh, 0]);
    DL.gridY(gg, ys, gw - 10, 3);
    DL.axisB(gg, xs, gh, 5, "T");
    DL.axisL(gg, ys, 3, "MACs", d => DL.big(d));
    DL.curve(gg, oa.map((v, i) => [xs(Ts[i]), ys(v)]), { stroke: DC.bad, w: 1.8 });
    DL.curve(gg, ob.map((v, i) => [xs(Ts[i]), ys(v)]), { stroke: DC.good, w: 1.8 });
    gg.append("line").attr("x1", xs(Math.min(32, T))).attr("x2", xs(Math.min(32, T)))
      .attr("y1", 0).attr("y2", gh).attr("stroke", DC.a2).attr("stroke-dasharray", "3 3");

    /* where the ratio reaches 100 */
    let t100 = null;
    for (let tt = 2; tt <= 4000; tt++) {
      const travel = tt * (tt * tt - 1) / 6, nUse = tt * (tt + 1) / 2, srcw = want === "U" ? dx : dh;
      const a = travel * (dh + dh * dh) + nUse * (dh + (want === "b" ? dh : srcw * dh));
      const b = tt * (dh + dh + dh * dh + (want === "b" ? dh : srcw * dh));
      if (a / b >= 100) { t100 = tt; break; }
    }
    document.getElementById("rc-readout").innerHTML =
      `∂L/∂${want} · max |double sum − recursion| = <b>${DL.fmtE(maxD, 2)}</b> ` +
      `${maxD < 1e-10 ? "— double-precision round-off, so they are the same computation" : "— NOT round-off"} · ` +
      `operations counted: double sum <b>${DL.commas(A.ops)}</b>, recursion <b>${DL.commas(B.ops)}</b>, ` +
      `ratio <b>${DL.fmt(A.ops / B.ops, 2)}×</b> at T = ${T}` +
      (t100 ? ` · the ratio reaches 100× at T ≈ <b>${t100}</b>.` : ".");
  }
  [Te, He, Se].forEach(e => e.addEventListener("input", draw));
  We.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 12 · #wk-svg — the worked example, live ═══════════ */
(function () {
  const svg = d3.select("#wk-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const ws = ["wk-w00", "wk-w01", "wk-w10", "wk-w11"].map(i => document.getElementById(i));
  const Ve = document.getElementById("wk-w"), Re = document.getElementById("wk-reset");
  const DEF = [0.6, 0.4, -0.5, 0.7];

  function build() {
    const w = ws.map((e, i) => { const v = parseFloat(e.value); return isFinite(v) ? v : DEF[i]; });
    return {
      cell: "vanilla", dx: 2, dh: 2, dy: 1, out: "sigmoid",
      p: { kind: "vanilla", dx: 2, dh: 2, U: [[0.5, -0.3], [0.2, 0.8]], W: [[w[0], w[1]], [w[2], w[3]]], b: [0.1, -0.2] },
      V: [[0.9], [-0.6]], c: [0.05]
    };
  }
  const X = [[1, 0], [0, 1], [1, 1]], Y = [[1], [0], [1]];

  function draw() {
    const view = Ve.value;
    const net = build();
    const f = DL.frame(svg, W, H, { l: 14, r: 14, t: 24, b: 16 }), g = f.g;
    const st = DL.seqForward(net, X);
    const hd = DL.seqHead(net, st, Y);
    const gr = DL.seqBPTT(net, X, Y, st);
    const num = DL.seqNumGrad(net, X, Y, 1e-5);
    const L = DL.seqLoss(net, X, Y, st);

    const cellW = 108, rowH = 26;
    function row(y, cells, cols) {
      cells.forEach((c, i) => {
        if (c === null) return;
        g.append("rect").attr("x", 96 + i * cellW).attr("y", y - 15).attr("width", cellW - 4).attr("height", rowH - 4)
          .attr("rx", 3).attr("fill", DC.panel2).attr("stroke", DC.line);
        g.append("text").attr("x", 96 + i * cellW + (cellW - 4) / 2).attr("y", y + 2).attr("text-anchor", "middle")
          .attr("font-size", 9.5).attr("font-family", "SF Mono, Menlo, monospace")
          .attr("fill", (cols && cols[i]) || DC.ink).text(c);
      });
    }
    const lab = (y, t, col) => g.append("text").attr("x", 88).attr("y", y + 2).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", col || DC.muted).text(t);
    const vec = (v, d) => "[" + v.map(x => DL.fmt(x, d === undefined ? 4 : d)).join(", ") + "]";

    if (view === "fwd") {
      RN.title(g, 96, 2, "t = 1"); RN.title(g, 96 + cellW, 2, "t = 2"); RN.title(g, 96 + 2 * cellW, 2, "t = 3");
      let y = 34;
      lab(y, "xₜ", DC.accent); row(y, [0, 1, 2].map(t => "[" + X[t].join(", ") + "]")); y += rowH;
      lab(y, "hₜ₋₁", DC.a2); row(y, [0, 1, 2].map(t => vec(st.Hprev[t]))); y += rowH;
      lab(y, "zₜ", DC.ink); row(y, [0, 1, 2].map(t => vec(st.H[t].map(v => Math.atanh(Math.min(0.999999999, Math.max(-0.999999999, v))))))); y += rowH;
      lab(y, "hₜ = tanh(zₜ)", DC.a2); row(y, [0, 1, 2].map(t => vec(st.H[t]))); y += rowH;
      lab(y, "sₜ = hₜV + c", DC.good); row(y, [0, 1, 2].map(t => vec(st.S[t]))); y += rowH;
      lab(y, "aₜ = σ(sₜ)", DC.good); row(y, [0, 1, 2].map(t => vec(st.A[t]))); y += rowH;
      lab(y, "yₜ", DC.muted); row(y, [0, 1, 2].map(t => "[" + Y[t].join("") + "]")); y += rowH;
      lab(y, "Lₜ", DC.bad); row(y, [0, 1, 2].map(t => DL.fmt(DL.seqLossAt(net, st.A[t], st.S[t], Y[t]), 6)), [DC.bad, DC.bad, DC.bad]);
      y += rowH + 12;
      g.append("text").attr("x", 96).attr("y", y).attr("font-size", 12).attr("fill", DC.a2).attr("font-weight", 600)
        .text("L = L₁ + L₂ + L₃ = " + DL.fmt(L, 6));
      RN.note(g, 96, y + 22, "h₀ = [0, 0], so z₁ is just x₁U + b — the first row of U plus b.", DC.muted, 10);
    } else if (view === "bwd") {
      RN.title(g, 96, 2, "t = 3"); RN.title(g, 96 + cellW, 2, "t = 2"); RN.title(g, 96 + 2 * cellW, 2, "t = 1");
      const ord = [2, 1, 0];
      const dhAll = [], gz = [], back = [];
      { let dn = [0, 0];
        for (const t of ord) {
          const d = hd.dH[t].map((v, j) => v + dn[j]);
          const z = d.map((v, j) => v * (1 - st.H[t][j] * st.H[t][j]));
          dhAll[t] = d; gz[t] = z;
          dn = DL.vecmat(z, DL.transpose(net.p.W)); back[t] = dn.slice();
        } }
      let y = 34;
      lab(y, "δₜ = aₜ − yₜ", DC.bad); row(y, ord.map(t => DL.fmt(hd.dS[t][0], 6))); y += rowH;
      lab(y, "δₜVᵀ — own loss", DC.good); row(y, ord.map(t => vec(hd.dH[t]))); y += rowH;
      lab(y, "gₜ₊₁Wᵀ — the future", DC.violet); row(y, ord.map(t => t === 2 ? "[0, 0]" : vec(back[t + 1]))); y += rowH;
      lab(y, "∂L/∂hₜ  (their sum)", DC.a2); row(y, ord.map(t => vec(dhAll[t])), [DC.a2, DC.a2, DC.a2]); y += rowH;
      lab(y, "1 − hₜ⊙hₜ", DC.muted); row(y, ord.map(t => vec(st.H[t].map(v => 1 - v * v)))); y += rowH;
      lab(y, "gₜ = ∂L/∂zₜ", DC.ink); row(y, ord.map(t => vec(gz[t]))); y += rowH;
      lab(y, "hₜ₋₁ᵀgₜ → ∂L/∂W", DC.a2); row(y, ord.map(t => "‖·‖ = " + DL.fmt(RN.norm(gz[t]) * RN.norm(st.Hprev[t]), 6))); y += rowH;
      lab(y, "gₜWᵀ, passed back", DC.violet); row(y, ord.map(t => vec(back[t])));
      y += rowH + 14;
      const zeroW = net.p.W.every(r => r.every(v => v === 0));
      RN.note(g, 96, y, zeroW
        ? "W = 0: the violet row is exactly zero — no gradient reaches back at all, and the read-out gradients are untouched."
        : "The violet row is the ONLY thing that makes this different from three independent one-step problems.",
        zeroW ? DC.bad : DC.muted, 10.5);
      RN.note(g, 96, y + 16, "At t = 1 the contribution to ∂L/∂W is h₀ᵀg₁ = 0, because h₀ = 0.", DC.muted, 10);
    } else {
      const items = [
        ["∂L/∂U", gr.dU, [[num.flat[0], num.flat[1]], [num.flat[2], num.flat[3]]]],
        ["∂L/∂W", gr.dW, [[num.flat[4], num.flat[5]], [num.flat[6], num.flat[7]]]],
        ["∂L/∂V", gr.dV, num.dV],
        ["∂L/∂b", [gr.db], [[num.flat[8], num.flat[9]]]],
        ["∂L/∂c", [gr.dc], [num.dc]]
      ];
      let x = 0, worst = 0, worstName = "";
      items.forEach(it => {
        const A = it[1], B = it[2];
        RN.title(g, x, 8, it[0], DC.a2);
        const cw = 62, rh = 20;
        const put = (M, oy, col) => {
          for (let a = 0; a < M.length; a++) for (let j = 0; j < M[0].length; j++) {
            g.append("rect").attr("x", x + j * cw).attr("y", oy + a * rh).attr("width", cw - 2).attr("height", rh - 2)
              .attr("rx", 2).attr("fill", DC.panel2).attr("stroke", col).attr("stroke-opacity", 0.5);
            g.append("text").attr("x", x + j * cw + (cw - 2) / 2).attr("y", oy + a * rh + 14).attr("text-anchor", "middle")
              .attr("font-size", 8.5).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink)
              .text(DL.fmt(M[a][j], 6));
          }
        };
        RN.note(g, x, 24, "analytic", DC.good, 9);
        put(A, 28, DC.good);
        RN.note(g, x, 28 + A.length * 20 + 14, "central differences", DC.accent, 9);
        put(B, 28 + A.length * 20 + 18, DC.accent);
        let w = 0;
        for (let a = 0; a < A.length; a++) for (let j = 0; j < A[0].length; j++) {
          const e = Math.abs(A[a][j] - B[a][j]) / Math.max(1e-9, Math.abs(A[a][j]), Math.abs(B[a][j]));
          if (e > w) w = e;
        }
        if (w > worst) { worst = w; worstName = it[0]; }
        RN.note(g, x, 28 + 2 * (A.length * 20) + 40, "worst rel " + DL.fmtE(w, 2), w < 1e-5 ? DC.good : DC.bad, 9.5);
        x += Math.max(2, A[0].length) * 62 + 22;
      });
      g.append("text").attr("x", 0).attr("y", 200).attr("font-size", 12).attr("fill", worst < 1e-5 ? DC.good : DC.bad)
        .attr("font-weight", 600).text("worst relative error over all 11 parameters: " + DL.fmtE(worst, 3) + "  (at " + worstName + ")");
      RN.note(g, 0, 220, "The numeric column re-runs the forward pass twice per parameter and never touches the backward code.", DC.muted, 10);
    }

    /* how much of ∂L/∂W came from steps other than the last */
    const last = (() => {
      const dn = DL.zeros(2);
      const d = hd.dH[2].map((v, j) => v + dn[j]);
      const z = d.map((v, j) => v * (1 - st.H[2][j] * st.H[2][j]));
      return RN.norm(z) * RN.norm(st.Hprev[2]);
    })();
    let worst = 0;
    const A = [gr.dU[0], gr.dU[1], gr.dW[0], gr.dW[1], gr.dV[0], gr.dV[1], gr.db, gr.dc];
    const Bn = [[num.flat[0], num.flat[1]], [num.flat[2], num.flat[3]], [num.flat[4], num.flat[5]],
    [num.flat[6], num.flat[7]], num.dV[0], num.dV[1], [num.flat[8], num.flat[9]], num.dc];
    A.forEach((r, k) => r.forEach((v, j) => {
      const e = Math.abs(v - Bn[k][j]) / Math.max(1e-9, Math.abs(v), Math.abs(Bn[k][j]));
      if (e > worst) worst = e;
    }));
    const share = 1 - last / Math.max(1e-300, RN.fnorm(gr.dW) + last * 0);
    document.getElementById("wk-readout").innerHTML =
      `W = [[${net.p.W[0].map(v => DL.fmt(v, 2)).join(", ")}], [${net.p.W[1].map(v => DL.fmt(v, 2)).join(", ")}]] · ` +
      `L = <b>${DL.fmt(L, 6)}</b> · ‖∂L/∂W‖ = <b>${DL.fmt(RN.fnorm(gr.dW), 6)}</b>, of which the last step alone contributes ` +
      `<b>${DL.fmt(last, 6)}</b> · worst relative error against central differences over all 11 parameters: ` +
      `<b>${DL.fmtE(worst, 3)}</b>.`;
  }
  ws.forEach(e => { e.addEventListener("input", draw); e.addEventListener("change", draw); });
  Ve.addEventListener("change", draw);
  Re.addEventListener("click", () => { ws.forEach((e, i) => e.value = String(DEF[i])); draw(); });
  draw();
})();

/* ═══════════ 13 · #au-svg — every parameter, analytic against numeric ═══════════ */
(function () {
  const svg = d3.select("#au-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const Ce = document.getElementById("au-c"), Oe = document.getElementById("au-o"),
    Te = document.getElementById("au-T"), Be = document.getElementById("au-b"),
    Se = document.getElementById("au-s");

  /* a DELIBERATELY buggy vanilla backward, for the three faults on offer.
     The correct path always goes through DL.seqBPTT; only this one is local. */
  function buggy(net, st, hd, bug) {
    const dh = net.dh, dx = net.dx, T = st.T;
    const dU = DL.zeros2(dx, dh), dW = DL.zeros2(dh, dh), db = DL.zeros(dh);
    let dn = DL.zeros(dh);
    for (let t = T - 1; t >= 0; t--) {
      const d = hd.dH[t].map((v, j) => v + dn[j]);
      const gz = (bug === "notanh") ? d.slice() : d.map((v, j) => v * (1 - st.H[t][j] * st.H[t][j]));
      if (bug !== "last" || t === T - 1) {
        for (let a = 0; a < dh; a++) for (let j = 0; j < dh; j++) dW[a][j] += st.Hprev[t][a] * gz[j];
      }
      for (let a = 0; a < dx; a++) for (let j = 0; j < dh; j++) dU[a][j] += st.X[t][a] * gz[j];
      for (let j = 0; j < dh; j++) db[j] += gz[j];
      dn = DL.vecmat(gz, bug === "wt" ? net.p.W : DL.transpose(net.p.W));
    }
    return { dU, dW, db };
  }

  function draw() {
    const cell = Ce.value, out = Oe.value, T = +Te.value, bug = Be.value, seed = +Se.value;
    document.getElementById("au-Tv").textContent = T;
    document.getElementById("au-sv").textContent = seed;
    /* the bug selector only applies to the vanilla cell — say so rather than pretend */
    const bugActive = (cell === "vanilla") ? bug : "none";
    const f = DL.frame(svg, W, H, { l: 54, r: 200, t: 24, b: 40 }), g = f.g;

    const dx = 3, dh = 5, dy = 4;
    const net = RN.net(cell, dx, dh, dy, { seed: seed, out: out, rho: cell === "vanilla" ? 0.95 : null, wScale: cell === "vanilla" ? null : 0.4, uScale: 0.5, forgetBias: 1 });
    const X = RN.seq(T, dx, seed + 300), Y = RN.targets(T, dy, out, seed + 310);
    const st = DL.seqForward(net, X), hd = DL.seqHead(net, st, Y);
    const good = DL.seqBPTT(net, X, Y, st);
    const num = DL.seqNumGrad(net, X, Y, 1e-5);

    let anaFlat;
    if (bugActive === "none") anaFlat = DL.cellGradFlatten(net.p, good);
    else { const b = buggy(net, st, hd, bugActive); anaFlat = DL.cellGradFlatten(net.p, b); }

    /* group the flat vector by parameter name */
    const groups = {};
    num.names.forEach((nm, i) => { const k = nm.slice(0, nm.indexOf("[")); (groups[k] = groups[k] || []).push(i); });
    const pts = [];
    Object.keys(groups).forEach(k => groups[k].forEach(i => pts.push({ k: k, a: anaFlat[i], n: num.flat[i] })));
    RN.flat(good.dV).forEach((v, i) => pts.push({ k: "V", a: v, n: RN.flat(num.dV)[i] }));
    good.dc.forEach((v, i) => pts.push({ k: "c", a: v, n: num.dc[i] }));

    const mx = Math.max(...pts.map(p => Math.max(Math.abs(p.a), Math.abs(p.n))), 1e-12);
    const lo = mx * 1e-6;
    const s = d3.scaleSymlog().domain([-mx * 1.2, mx * 1.2]).constant(lo).range([0, f.iw]).clamp(true);
    const sy = d3.scaleSymlog().domain([-mx * 1.2, mx * 1.2]).constant(lo).range([f.ih, 0]).clamp(true);
    DL.gridY(g, sy, f.iw, 5); DL.gridX(g, s, f.ih, 5);
    DL.axisB(g, s, f.ih, 5, "analytic", d => DL.fmtE(d, 0));
    DL.axisL(g, sy, 5, "central differences", d => DL.fmtE(d, 0));
    g.append("line").attr("x1", s(-mx)).attr("x2", s(mx)).attr("y1", sy(-mx)).attr("y2", sy(mx))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "4 3");
    const gk = Object.keys(groups).concat(["V", "c"]);
    const cscale = d3.scaleOrdinal().domain(gk).range([DC.accent, DC.a2, DC.good, DC.violet, DC.teal, DC.rose, DC.lime, DC.bad, DC.muted, "#e879f9", "#38bdf8", "#facc15", "#fb923c", "#94a3b8"]);
    pts.forEach(p => {
      g.append("circle").attr("cx", s(p.a)).attr("cy", sy(p.n)).attr("r", 2.6)
        .attr("fill", cscale(p.k)).attr("fill-opacity", 0.75);
    });

    /* per-group worst relative error */
    const per = {};
    gk.forEach(k => {
      const sub = pts.filter(p => p.k === k);
      const M = Math.max(...sub.map(p => Math.abs(p.a)), 1e-30);
      per[k] = Math.max(...sub.map(p => Math.abs(p.a - p.n) / Math.max(Math.abs(p.a), Math.abs(p.n), 1e-3 * M)));
    });
    const worst = Math.max(...Object.values(per));
    const okFrac = pts.filter(p => Math.abs(p.a - p.n) <= 1e-3 * Math.max(Math.abs(p.a), Math.abs(p.n), 1e-12)).length / pts.length;

    const sx = f.iw + 16, bw = 118;
    RN.title(g, sx, 0, "worst relative error, per group");
    const bs = d3.scaleLog().domain([1e-11, Math.max(1e-2, worst * 3)]).range([0, bw]).clamp(true);
    gk.forEach((k, i) => {
      const yy = 14 + i * 17;
      g.append("rect").attr("x", sx + 26).attr("y", yy).attr("width", Math.max(1, bs(Math.max(per[k], 1e-11)))).attr("height", 11)
        .attr("fill", cscale(k)).attr("fill-opacity", 0.75);
      RN.note(g, sx, yy + 9.5, k, cscale(k), 9.5);
      g.append("text").attr("x", sx + 26 + bw + 6).attr("y", yy + 9.5).attr("font-size", 9)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", per[k] > 1e-3 ? DC.bad : DC.ink)
        .text(DL.fmtE(per[k], 1));
    });
    const yLine = 14 + gk.length * 17 + 12;
    [["achievable here", 1e-6, DC.good], ["a real bug", 1e-2, DC.bad]].forEach((d, i) => {
      g.append("line").attr("x1", sx + 26 + bs(d[1])).attr("x2", sx + 26 + bs(d[1])).attr("y1", 10).attr("y2", yLine - 16)
        .attr("stroke", d[2]).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.7);
      RN.note(g, sx + 26, yLine + i * 12, "─ " + d[0] + " ≈ " + DL.fmtE(d[1], 0), d[2], 9);
    });
    const verdict = worst < 1e-4;
    g.append("text").attr("x", sx).attr("y", yLine + 34).attr("font-size", 11.5)
      .attr("font-weight", 600).attr("fill", verdict ? DC.good : DC.bad)
      .text(verdict ? "PASS" : "FAIL");
    RN.note(g, sx + 44, yLine + 34, "worst " + DL.fmtE(worst, 2), DC.muted, 10);
    if (cell !== "vanilla" && bug !== "none")
      RN.note(g, sx, yLine + 52, "(the injected bug applies to the", DC.a2, 9.5),
        RN.note(g, sx, yLine + 63, " vanilla cell only — this is the", DC.a2, 9.5),
        RN.note(g, sx, yLine + 74, " correct " + cell.toUpperCase() + " backward pass)", DC.a2, 9.5);

    const bugText = {
      none: "none — as derived", wt: "W instead of Wᵀ when travelling back",
      notanh: "the tanh derivative dropped", last: "only the last step accumulates into ∂L/∂W"
    }[bugActive];
    document.getElementById("au-readout").innerHTML =
      `<b>${DL.CELL[cell].label}</b>, ${Oe.options[Oe.selectedIndex].text}, T = ${T}, ${pts.length} parameters · ` +
      `bug: <b>${bugText}</b>${cell !== "vanilla" && bug !== "none" ? " (only injectable into the vanilla cell)" : ""} · ` +
      `worst relative error <b>${DL.fmtE(worst, 2)}</b> · ` +
      `<b>${DL.fmt(100 * okFrac, 1)}%</b> of parameters agree to better than one part in a thousand · ` +
      `verdict: <b>${verdict ? "pass" : "fail"}</b>.`;
  }
  [Te, Se].forEach(e => e.addEventListener("input", draw));
  [Ce, Oe, Be].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 14 · #ct-svg — time, memory and latency ═══════════ */
(function () {
  const svg = d3.select("#ct-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Te = document.getElementById("ct-T"), Ne = document.getElementById("ct-N"),
    He = document.getElementById("ct-dh"), Le = document.getElementById("ct-L"),
    Se = document.getElementById("ct-st"), Be = document.getElementById("ct-b");
  const K = { vanilla: 1, gru: 5, lstm: 6 };

  function memBytes(cell, T, N, dh, L, strat) {
    const eff = strat === "ckpt" ? Math.ceil(Math.sqrt(T)) : (strat === "trunc" ? Math.min(T, 32) : T);
    return N * eff * dh * K[cell] * L * 4;
  }
  function draw() {
    const T = +Te.value, N = +Ne.value, dh = +He.value, L = +Le.value, strat = Se.value, budget = +Be.value;
    document.getElementById("ct-Tv").textContent = T;
    document.getElementById("ct-Nv").textContent = N;
    document.getElementById("ct-dhv").textContent = dh;
    document.getElementById("ct-Lv").textContent = L;
    document.getElementById("ct-bv").textContent = budget;
    const f = DL.frame(svg, W, H, { l: 12, r: 12, t: 24, b: 20 }), g = f.g;

    /* ---- memory panel ---- */
    const pw = 260, ph = 240;
    RN.title(g, 0, 0, "activation memory against sequence length");
    const gg = g.append("g").attr("transform", "translate(38,20)");
    const Ts = d3.range(0, 41).map(i => Math.round(8 * Math.pow(2048 / 8, i / 40)));
    const curves = ["vanilla", "gru", "lstm"].map(c => ({
      c: c, pts: Ts.map(t => [t, memBytes(c, t, N, dh, L, strat) / 1073741824])
    }));
    const allv = curves.reduce((a, cu) => a.concat(cu.pts.map(p => p[1])), []).concat([budget]);
    const x = d3.scaleLog().domain([8, 2048]).range([0, pw - 40]);
    const y = d3.scaleLog().domain([Math.max(1e-6, Math.min(...allv.filter(v => v > 0)) / 2), Math.max(...allv) * 2]).range([ph, 0]).clamp(true);
    DL.gridY(gg, y, pw - 40, 4);
    DL.axisB(gg, x, ph, 4, "T", d3.format("~s"));
    DL.axisL(gg, y, 4, "GB", d => DL.fmtE(d, 0));
    gg.append("line").attr("x1", 0).attr("x2", pw - 40).attr("y1", y(budget)).attr("y2", y(budget))
      .attr("stroke", DC.bad).attr("stroke-dasharray", "4 3");
    RN.note(gg, 2, y(budget) - 4, budget + " GB budget", DC.bad, 9);
    const maxT = {};
    curves.forEach(cu => {
      DL.curve(gg, cu.pts.map(p => [x(p[0]), y(Math.max(p[1], 1e-9))]), { stroke: RN.CELLC[cu.c], w: 1.9 });
      let mt = null;
      for (let t = 8; t <= 100000; t++) { if (memBytes(cu.c, t, N, dh, L, strat) / 1073741824 > budget) { mt = t - 1; break; } }
      maxT[cu.c] = mt === null ? ">100000" : mt;
      if (mt !== null && mt >= 8 && mt <= 2048)
        gg.append("circle").attr("cx", x(mt)).attr("cy", y(budget)).attr("r", 3.4).attr("fill", RN.CELLC[cu.c]);
    });
    gg.append("line").attr("x1", x(Math.min(2048, Math.max(8, T)))).attr("x2", x(Math.min(2048, Math.max(8, T))))
      .attr("y1", 0).attr("y2", ph).attr("stroke", DC.a2).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.7);
    DL.legend(gg, ["vanilla", "gru", "lstm"].map(c => ({ label: DL.CELL[c].label, color: RN.CELLC[c] })),
      4, ph + 32, { vertical: false, step: 82, font: 9.5 });

    /* ---- arithmetic panel ---- */
    const ax = 320;
    RN.title(g, ax, 0, "MACs per training step (per example), forward + backward");
    const macs = c => T * DL.CELL[c].nGate * (dh * dh + dh * dh) * L;
    const mm = Math.max(...["vanilla", "gru", "lstm"].map(macs));
    ["vanilla", "gru", "lstm"].forEach((c, i) => {
      const yy = 24 + i * 42, m = macs(c), bw = 200;
      g.append("rect").attr("x", ax + 56).attr("y", yy).attr("width", bw * m / mm / 3).attr("height", 16)
        .attr("fill", RN.CELLC[c]).attr("fill-opacity", 0.7);
      g.append("rect").attr("x", ax + 56 + bw * m / mm / 3).attr("y", yy).attr("width", bw * m / mm * 2 / 3).attr("height", 16)
        .attr("fill", RN.CELLC[c]).attr("fill-opacity", 0.32);
      RN.note(g, ax, yy + 12, DL.CELL[c].label, DC.muted, 10);
      RN.note(g, ax + 56, yy + 30, "fwd " + DL.big(m) + " · bwd ≈ " + DL.big(2 * m) + " · total " + DL.big(3 * m), DC.ink, 9.5);
    });
    RN.note(g, ax + 56, 24 + 3 * 42 + 4, "the darker segment is the forward pass; part 2's 1 : 2 : 3 rule holds exactly", DC.muted, 9);

    /* ---- serial chain ---- */
    const sy = 200;
    RN.title(g, ax, sy, "serial dependency chain, per layer");
    const chains = [["recurrent", T, DC.a2], ["attention", 1, DC.teal]];
    const cs = d3.scaleLog().domain([1, Math.max(2, T)]).range([2, 220]);
    chains.forEach((d, i) => {
      const yy = sy + 20 + i * 34;
      g.append("rect").attr("x", ax + 76).attr("y", yy).attr("width", cs(d[1])).attr("height", 16).attr("rx", 3)
        .attr("fill", d[2]).attr("fill-opacity", 0.6).attr("stroke", d[2]);
      RN.note(g, ax, yy + 12, d[0], DC.muted, 10);
      g.append("text").attr("x", ax + 76 + cs(d[1]) + 8).attr("y", yy + 12).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(DL.commas(d[1]));
    });
    RN.note(g, ax, sy + 96, "same arithmetic; " + DL.commas(T) + "× the latency.", DC.a2, 10);

    const cur = c => memBytes(c, T, N, dh, L, strat) / 1073741824;
    document.getElementById("ct-readout").innerHTML =
      `T=${T} N=${N} dₕ=${dh} ${L} layer${L === 1 ? "" : "s"}, ${Se.options[Se.selectedIndex].text} · ` +
      `activation memory: vanilla <b>${DL.fmt(cur("vanilla"), 3)} GB</b>, GRU <b>${DL.fmt(cur("gru"), 3)} GB</b>, ` +
      `LSTM <b>${DL.fmt(cur("lstm"), 3)} GB</b> · longest T that fits ${budget} GB: ` +
      `vanilla <b>${maxT.vanilla}</b>, GRU <b>${maxT.gru}</b>, LSTM <b>${maxT.lstm}</b> · ` +
      `LSTM training arithmetic <b>${DL.big(3 * macs("lstm"))}</b> MACs per example per step · ` +
      `serial chain <b>${DL.commas(T)}</b> products against <b>1</b> for attention.`;
  }
  [Te, Ne, He, Le, Be].forEach(e => e.addEventListener("input", draw));
  Se.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 15 · #tc-svg — which terms each truncation form keeps ═══════════ */
(function () {
  const svg = d3.select("#tc-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Te = document.getElementById("tc-T"), Ke = document.getElementById("tc-k"),
    Fe = document.getElementById("tc-f"), Re = document.getElementById("tc-rho");

  function draw() {
    const T = +Te.value;
    Ke.max = String(T);
    const k = Math.min(+Ke.value, T), form = Fe.value, rho = +Re.value;
    document.getElementById("tc-Tv").textContent = T;
    document.getElementById("tc-kv").textContent = k;
    document.getElementById("tc-rhov").textContent = DL.fmt(rho, 2);
    const f = DL.frame(svg, W, H, { l: 40, r: 210, t: 30, b: 24 }), g = f.g;

    const dx = 3, dh = 4, dy = 2;
    const net = RN.net("vanilla", dx, dh, dy, { seed: 17, rho: rho, uScale: 0.6 });
    const X = RN.seq(T, dx, 401), Y = RN.targets(T, dy, "sigmoid", 403);
    const st = DL.seqForward(net, X), hd = DL.seqHead(net, st, Y);

    const keep = (t, i) => {                     // t, i are 0-based
      if (i > t) return false;
      if (form === "full") return true;
      if (form === "A") return t - i < k;
      const w = Math.floor(t / k) * k;           // form B: same window
      return i >= w;
    };
    const termNorm = (t, i) => {
      let u = hd.dH[t].slice();
      for (let j = t; j > i; j--) {
        const gz = u.map((v, m) => v * (1 - st.H[j][m] * st.H[j][m]));
        u = DL.vecmat(gz, DL.transpose(net.p.W));
      }
      const gg2 = u.map((v, m) => v * (1 - st.H[i][m] * st.H[i][m]));
      return RN.norm(gg2) * RN.norm(st.Hprev[i]);
    };

    const cw = Math.min(15, Math.min((f.iw) / T, 240 / T));
    const vals = [];
    for (let t = 0; t < T; t++) for (let i = 0; i <= t; i++) vals.push(termNorm(t, i));
    const pos = vals.filter(v => v > 0);
    const cs = d3.scaleLog().domain([Math.max(Math.min(...pos), Math.max(...pos) * 1e-10), Math.max(...pos)]).range([0, 1]).clamp(true);

    let kept = 0, keptMass = 0, totMass = 0, idx = 0;
    for (let t = 0; t < T; t++) for (let i = 0; i < T; i++) {
      const inTri = i <= t;
      const v = inTri ? vals[idx++] : NaN;
      const kp = inTri && keep(t, i);
      if (inTri) { totMass += v; if (kp) { kept++; keptMass += v; } }
      const el = g.append("rect").attr("x", i * cw).attr("y", t * cw).attr("width", cw - 0.6).attr("height", cw - 0.6)
        .attr("shape-rendering", "crispEdges");
      if (!inTri) el.attr("fill", DC.panel).attr("fill-opacity", 0.12);
      else if (kp) el.attr("fill", RN.heat(cs(Math.max(v, 1e-300))));
      else el.attr("fill", DC.line).attr("fill-opacity", 0.45);
    }
    if (form === "B") for (let w = k; w < T; w += k)
      g.append("line").attr("x1", w * cw).attr("x2", w * cw).attr("y1", 0).attr("y2", T * cw)
        .attr("stroke", DC.a2).attr("stroke-width", 1.2).attr("stroke-opacity", 0.8);
    RN.note(g, 0, -16, "use step i →", DC.muted, 9.5);
    g.append("text").attr("transform", `translate(-24,${T * cw / 2}) rotate(-90)`).attr("text-anchor", "middle")
      .attr("font-size", 9.5).attr("fill", DC.muted).text("loss step t");
    RN.note(g, 0, T * cw + 16, "coloured = kept · grey = dropped", DC.muted, 9.5);

    /* history depth per loss */
    const hx = T * cw + 26, hw = Math.max(60, f.iw - hx);
    RN.title(g, hx, -16, "history each loss gets");
    const depth = [];
    for (let t = 0; t < T; t++) { let c = 0; for (let i = 0; i <= t; i++) if (keep(t, i)) c++; depth.push(c); }
    const dx2 = d3.scaleLinear().domain([0, Math.max(...depth)]).range([0, hw - 8]);
    for (let t = 0; t < T; t++)
      g.append("rect").attr("x", hx).attr("y", t * cw).attr("width", Math.max(0.6, dx2(depth[t]))).attr("height", cw - 0.6)
        .attr("fill", form === "B" ? DC.a2 : DC.good).attr("fill-opacity", 0.7);
    const meanD = d3.mean(depth);
    g.append("line").attr("x1", hx + dx2(meanD)).attr("x2", hx + dx2(meanD)).attr("y1", 0).attr("y2", T * cw)
      .attr("stroke", DC.ink).attr("stroke-dasharray", "3 3");
    RN.note(g, hx, T * cw + 16, "mean " + DL.fmt(meanD, 2) + " steps", DC.ink, 9.5);

    const sx = f.iw + 16;
    const kv = DL.kv(g, sx, 0, { keyW: 130, size: 10.5, lead: 16 });
    kv("terms in the full sum", String(T * (T + 1) / 2));
    kv("terms kept", DL.commas(kept) + "  (" + DL.fmt(100 * kept / (T * (T + 1) / 2), 1) + "%)");
    kv("gradient mass kept", DL.fmt(100 * keptMass / Math.max(1e-300, totMass), 2) + "%",
      keptMass / Math.max(1e-300, totMass) > 0.98 ? DC.good : DC.a2);
    kv("mean history per loss", DL.fmt(meanD, 2));
    kv("minimum history", String(Math.min(...depth)), Math.min(...depth) === 1 && form === "B" ? DC.bad : DC.ink);
    kv("states to hold", form === "full" ? String(T) : String(k));
    kv("memory saving", form === "full" ? "1.00×" : DL.fmt(T / k, 2) + "×", DC.a2);
    RN.note(g, sx, 128, form === "B" ? "note the sawtooth: a loss at the start of" : (form === "A" ? "every loss gets exactly k steps," : "nothing is dropped."), DC.muted, 9.5);
    RN.note(g, sx, 140, form === "B" ? "a window sees ONE step of history." : (form === "A" ? "except near the very beginning." : ""), DC.muted, 9.5);

    document.getElementById("tc-readout").innerHTML =
      `<b>${Fe.options[Fe.selectedIndex].text}</b>, k = ${k}, T = ${T} · kept <b>${DL.commas(kept)}</b> of ` +
      `<b>${T * (T + 1) / 2}</b> terms, carrying <b>${DL.fmt(100 * keptMass / Math.max(1e-300, totMass), 2)}%</b> of the total ` +
      `gradient magnitude · history per loss: mean <b>${DL.fmt(meanD, 2)}</b>, minimum <b>${Math.min(...depth)}</b> · ` +
      `states held <b>${form === "full" ? T : k}</b> instead of ${T}, a <b>${form === "full" ? "1.00" : DL.fmt(T / k, 2)}×</b> memory saving.`;
  }
  [Te, Ke, Re].forEach(e => e.addEventListener("input", draw));
  Fe.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 16 · #tb-svg — the truncation bias, measured ═══════════ */
(function () {
  const svg = d3.select("#tb-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Te = document.getElementById("tb-T"), Re = document.getElementById("tb-rho"),
    Me = document.getElementById("tb-m"), Ge = document.getElementById("tb-g"),
    Ye = document.getElementById("tb-y");

  function draw() {
    const T = +Te.value, rho = +Re.value, meas = Me.value, want = Ge.value, where = Ye.value;
    document.getElementById("tb-Tv").textContent = T;
    document.getElementById("tb-rhov").textContent = DL.fmt(rho, 2);
    const f = DL.frame(svg, W, H, { l: 54, r: 200, t: 24, b: 40 }), g = f.g;

    const dx = 3, dh = 8, dy = 3;
    const net = RN.net("vanilla", dx, dh, dy, { seed: 17, rho: rho, uScale: 0.6 });
    const X = RN.seq(T, dx, 71), Y = RN.targets(T, dy, "sigmoid", 73, where);
    const st = DL.seqForward(net, X);
    const pick = g2 => want === "b" ? g2.db.slice() : RN.flat(g2["d" + want]);
    const full = pick(DL.seqBPTT(net, X, Y, st));
    const nf = RN.norm(full);

    const ks = [];
    for (let k = 1; k <= T; k++) ks.push(k);
    const rowsA = [], rowsB = [];
    ks.forEach(k => {
      const a = pick(DL.seqBPTTtrunc(net, X, Y, k, st)), b = pick(DL.seqBPTTchunk(net, X, Y, k, st));
      const m = gg2 => {
        if (meas === "rel") return RN.norm(gg2.map((v, i) => v - full[i])) / Math.max(1e-300, nf);
        if (meas === "cos") return Math.acos(Math.min(1, Math.max(-1, RN.cosv(gg2, full)))) * 180 / Math.PI;
        return RN.norm(gg2) / Math.max(1e-300, nf);
      };
      rowsA.push(m(a)); rowsB.push(m(b));
    });

    const isLog = meas !== "mag";
    const x = d3.scaleLinear().domain([1, T]).range([0, f.iw]);
    let y;
    if (isLog) {
      const pos = rowsA.concat(rowsB).filter(v => v > 0);
      const lo = pos.length ? Math.max(Math.min(...pos), 1e-9) : 1e-9;
      y = d3.scaleLog().domain([lo * 0.6, Math.max(...rowsA.concat(rowsB), 1e-8) * 1.6]).range([f.ih, 0]).clamp(true);
    } else {
      y = d3.scaleLinear().domain([0, Math.max(1.2, ...rowsA.concat(rowsB))]).range([f.ih, 0]);
    }
    DL.gridY(g, y, f.iw, 5);
    DL.axisB(g, x, f.ih, 8, "truncation window k");
    DL.axisL(g, y, 5, meas === "rel" ? "‖gₖ − g‖ / ‖g‖" : meas === "cos" ? "direction error (degrees)" : "‖gₖ‖ / ‖g‖",
      d => isLog ? DL.fmtE(d, 0) : DL.fmt(d, 2));
    if (isLog) {
      const ref = ks.map(k => [x(k), y(Math.max(1e-12, Math.pow(rho, k) * (meas === "cos" ? 57.3 : 1)))]);
      DL.curve(g, ref, { stroke: DC.muted, w: 1.2, dash: "4 3" });
    } else {
      g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y(1)).attr("y2", y(1))
        .attr("stroke", DC.muted).attr("stroke-dasharray", "4 3");
    }
    const plot = (rows, col) => {
      const pts = rows.map((v, i) => [x(ks[i]), y(isLog ? Math.max(v, 1e-12) : v)]);
      DL.curve(g, pts, { stroke: col, w: 2 });
      pts.forEach(p => g.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", 2.2).attr("fill", col));
    };
    plot(rowsA, DC.good); plot(rowsB, DC.a2);
    DL.legend(g, [{ label: "form A — per-loss", color: DC.good }, { label: "form B — chunked", color: DC.a2 },
    { label: isLog ? "ρᵏ reference" : "exact", color: DC.muted, dash: "4 3" }],
      4, f.ih + 30, { vertical: false, step: 156, font: 9.5 });

    /* find the non-monotone dips in form B */
    let dips = 0;
    for (let i = 1; i + 1 < rowsB.length; i++) if (rowsB[i] > rowsB[i - 1] + 1e-12) dips++;

    /* history depth panel */
    const kSel = Math.max(1, Math.min(Math.round(T / 4), T));
    const sx = f.iw + 16;
    const kv = DL.kv(g, sx, 4, { keyW: 130, size: 10.5, lead: 16 });
    const iSel = kSel - 1;
    kv("at k = " + kSel + ", form A", meas === "cos" ? DL.fmt(rowsA[iSel], 2) + "°" : RN.P(rowsA[iSel], 4), DC.good);
    kv("at k = " + kSel + ", form B", meas === "cos" ? DL.fmt(rowsB[iSel], 2) + "°" : RN.P(rowsB[iSel], 4), DC.a2);
    kv("form B / form A", DL.fmt(rowsB[iSel] / Math.max(1e-12, rowsA[iSel]), 2) + "×", DC.bad);
    let k1 = null;
    for (let i = 0; i < rowsA.length; i++) if (meas === "rel" && rowsA[i] < 0.01) { k1 = ks[i]; break; }
    kv("form A below 1% at k =", k1 === null ? (meas === "rel" ? "never" : "n/a") : String(k1));
    kv("form B not monotone at", String(dips) + " of " + (rowsB.length - 2) + " windows", dips > 0 ? DC.bad : DC.good);
    kv("k = 1 direction error", DL.fmt(Math.acos(Math.min(1, Math.max(-1, RN.cosv(pick(DL.seqBPTTtrunc(net, X, Y, 1, st)), full)))) * 180 / Math.PI, 2) + "°", DC.bad);
    kv("k = T error (a check)", RN.P(rowsA[rowsA.length - 1], 3), rowsA[rowsA.length - 1] < 1e-9 ? DC.good : DC.bad);
    RN.note(g, sx, 132, "the last row must be zero: truncating", DC.muted, 9.5);
    RN.note(g, sx, 144, "at k = T is full BPTT, so it is a", DC.muted, 9.5);
    RN.note(g, sx, 156, "check on the measurement itself.", DC.muted, 9.5);

    document.getElementById("tb-readout").innerHTML =
      `∂L/∂${want}, T = ${T}, ρ(W) = ${DL.fmt(rho, 2)}, targets at ${where === "all" ? "every step" : "the last step only"} · ` +
      `at k = ${kSel}: form A <b>${meas === "cos" ? DL.fmt(rowsA[iSel], 2) + "°" : RN.P(rowsA[iSel], 4)}</b>, ` +
      `form B <b>${meas === "cos" ? DL.fmt(rowsB[iSel], 2) + "°" : RN.P(rowsB[iSel], 4)}</b> ` +
      `— chunked is <b>${DL.fmt(rowsB[iSel] / Math.max(1e-12, rowsA[iSel]), 2)}×</b> worse · ` +
      `form B increases with k at <b>${dips}</b> of ${rowsB.length - 2} window sizes, so it is <b>not monotone</b> · ` +
      `at k = T both are <b>${RN.P(rowsA[rowsA.length - 1], 3)}</b>, which is the correctness check.`;
  }
  [Te, Re].forEach(e => e.addEventListener("input", draw));
  [Me, Ge, Ye].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 17 · #pw-svg — a matrix raised to a power ═══════════ */
(function () {
  const svg = d3.select("#pw-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const He = document.getElementById("pw-dh"), Re = document.getElementById("pw-rho"),
    De = document.getElementById("pw-d"), Ne = document.getElementById("pw-nl"),
    Se = document.getElementById("pw-s");

  function draw() {
    const dh = +He.value, rho = +Re.value, d = +De.value, nl = Ne.value, seed = +Se.value;
    document.getElementById("pw-dhv").textContent = dh;
    document.getElementById("pw-rhov").textContent = DL.fmt(rho, 2);
    document.getElementById("pw-dv").textContent = d;
    document.getElementById("pw-sv").textContent = seed;
    const f = DL.frame(svg, W, H, { l: 20, r: 20, t: 24, b: 24 }), g = f.g;

    const Wm = RN.mat(dh, seed, rho);
    const e1 = DL.eigGeneral(Wm), eD = DL.eigGeneral(DL.matPow(Wm, d));
    const rhoA = e1.rho;

    /* ---- left: the complex plane ---- */
    const R = 132, cx = R + 16, cy = 16 + R;
    RN.title(g, 0, 0, "the eigenvalues, and the same eigenvalues raised to the power " + d);
    const lim = Math.max(1.35, rhoA * 1.25, Math.max(...eD.re.map((v, i) => Math.hypot(v, eD.im[i]))) * 0.0 + 1.35);
    const sc = R / lim;
    g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", sc).attr("fill", "none")
      .attr("stroke", DC.ink).attr("stroke-dasharray", "4 3").attr("stroke-opacity", 0.8);
    g.append("line").attr("x1", cx - R).attr("x2", cx + R).attr("y1", cy).attr("y2", cy).attr("stroke", DC.grid);
    g.append("line").attr("x1", cx).attr("x2", cx).attr("y1", cy - R).attr("y2", cy + R).attr("stroke", DC.grid);
    RN.note(g, cx + sc + 3, cy - 5, "|λ| = 1", DC.ink, 9);
    const dom = e1.re.map((v, i) => Math.hypot(v, e1.im[i])).indexOf(rhoA);
    for (let i = 0; i < dh; i++) {
      const m = Math.hypot(e1.re[i], e1.im[i]);
      g.append("circle").attr("cx", cx + e1.re[i] * sc).attr("cy", cy - e1.im[i] * sc).attr("r", i === dom ? 5 : 3.6)
        .attr("fill", m > 1 ? DC.bad : DC.accent).attr("fill-opacity", 0.85)
        .attr("stroke", i === dom ? DC.ink : "none").attr("stroke-width", 1.4);
      const md = Math.hypot(eD.re[i], eD.im[i]);
      const clamped = Math.min(md, lim * 0.98);
      const ang = Math.atan2(eD.im[i], eD.re[i]);
      g.append("circle").attr("cx", cx + Math.cos(ang) * clamped * sc).attr("cy", cy - Math.sin(ang) * clamped * sc)
        .attr("r", 2.6).attr("fill", DC.a2).attr("fill-opacity", 0.9);
    }
    DL.legend(g, [{ label: "λ(W)", color: DC.accent }, { label: "λ(W) inside |λ|>1", color: DC.bad },
    { label: "λ(Wᵈ), radially clamped", color: DC.a2 }], 4, 2 * R + 34, { vertical: false, step: 100, font: 9 });

    /* ---- right: measured norm against distance ---- */
    const gx = 2 * R + 60, gw = f.iw - gx, gh = 210;
    RN.title(g, gx, 0, "gradient carried back through d steps");
    const gg = g.append("g").attr("transform", `translate(${gx},18)`);
    const ds = d3.range(1, 61);
    const linear = ds.map(k => DL.specNorm(DL.matPow(Wm, k), 260, 3));
    let tanhC = null;
    if (nl === "tanh") {
      const net = { cell: "vanilla", dx: 2, dh: dh, dy: 1, out: "linear",
        p: { kind: "vanilla", dx: 2, dh: dh, U: RN.mat(dh, seed + 5).slice(0, 2).map(r => r.slice(0, dh)), W: Wm, b: DL.zeros(dh) },
        V: DL.zeros2(dh, 1), c: [0] };
      const X = RN.seq(61, 2, seed + 11);
      tanhC = ds.map(k => DL.specNorm(DL.stateJac(net, X, 61 - k, 1e-5), 200, 3));
    }
    const pred = ds.map(k => Math.pow(rhoA, k));
    const allv = linear.concat(pred).concat(tanhC || []).filter(v => v > 0 && isFinite(v));
    const x = d3.scaleLinear().domain([1, 60]).range([0, gw - 12]);
    const y = d3.scaleLog().domain([Math.max(1e-16, Math.min(...allv) / 3), Math.max(...allv) * 3]).range([gh, 0]).clamp(true);
    DL.gridY(gg, y, gw - 12, 5);
    DL.axisB(gg, x, gh, 6, "distance d");
    DL.axisL(gg, y, 5, "‖J‖₂", v => DL.fmtE(v, 0));
    [[1.18e-38, 3.4e38, DC.muted, "float32 range"]].forEach(b => {
      gg.append("rect").attr("x", 0).attr("y", y(b[1])).attr("width", gw - 12)
        .attr("height", Math.max(0, y(b[0]) - y(b[1]))).attr("fill", DC.good).attr("fill-opacity", 0.05);
    });
    DL.curve(gg, pred.map((v, i) => [x(ds[i]), y(Math.max(v, 1e-300))]), { stroke: DC.muted, w: 1.3, dash: "4 3" });
    DL.curve(gg, linear.map((v, i) => [x(ds[i]), y(Math.max(v, 1e-300))]), { stroke: DC.accent, w: 2 });
    if (tanhC) DL.curve(gg, tanhC.map((v, i) => [x(ds[i]), y(Math.max(v, 1e-300))]), { stroke: DC.a2, w: 2 });
    gg.append("line").attr("x1", x(d)).attr("x2", x(d)).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.6);
    DL.legend(gg, [{ label: "measured ‖Wᵈ‖₂", color: DC.accent }, { label: "ρ(W)ᵈ", color: DC.muted, dash: "4 3" }]
      .concat(tanhC ? [{ label: "through tanh", color: DC.a2 }] : []),
      4, gh + 30, { vertical: false, step: 128, font: 9.5 });

    const measured = linear[Math.min(d, 60) - 1];
    const predicted = Math.pow(rhoA, d);
    const nn = DL.specNorm(Wm, 300, 3);
    let dFloor = null;
    for (let k = 1; k <= 400; k++) if (Math.pow(rhoA, k) < 1.18e-38) { dFloor = k; break; }
    const kv = DL.kv(g, gx, 268, { keyW: 152, size: 10.5, lead: 15 });
    kv("ρ(W) achieved", DL.fmt(rhoA, 6), DC.accent);
    kv("‖W‖₂ (spectral norm)", DL.fmt(nn, 6));
    kv("‖W‖₂ / ρ(W) — non-normality", DL.fmt(nn / rhoA, 4), nn / rhoA > 2 ? DC.a2 : DC.ink);
    kv("measured ‖Wᵈ‖₂ at d = " + d, DL.fmtE(measured, 3));
    kv("ρ(W)ᵈ", DL.fmtE(predicted, 3));
    kv("ratio", DL.fmt(measured / Math.max(1e-300, predicted), 4), DC.a2);
    kv("ρ(Wᵈ) − ρ(W)ᵈ, relative", DL.fmtE(Math.abs(eD.rho - predicted) / Math.max(1e-300, predicted), 2), DC.good);
    kv("d at the float32 floor", dFloor === null ? "beyond 400" : String(dFloor), DC.bad);

    document.getElementById("pw-readout").innerHTML =
      `ρ(W) = <b>${DL.fmt(rhoA, 6)}</b>, ‖W‖₂ = <b>${DL.fmt(nn, 6)}</b>, so the non-normality factor is ` +
      `<b>${DL.fmt(nn / rhoA, 4)}</b> · at d = ${d} the measured ‖Wᵈ‖₂ is <b>${DL.fmtE(measured, 3)}</b> against ` +
      `ρᵈ = <b>${DL.fmtE(predicted, 3)}</b>, a ratio of <b>${DL.fmt(measured / Math.max(1e-300, predicted), 4)}</b> ` +
      `that does not grow with d · the identity ρ(Wᵈ) = ρ(W)ᵈ holds to <b>${DL.fmtE(Math.abs(eD.rho - predicted) / Math.max(1e-300, predicted), 2)}</b> relative · ` +
      (dFloor === null ? `a gradient of 1 is still representable in float32 at d = 400, the end of the extrapolation.`
        : `a gradient of 1 falls below the float32 floor at d ≈ <b>${dFloor}</b>.`);
  }
  [He, Re, De, Se].forEach(e => e.addEventListener("input", draw));
  Ne.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 18 · #sr-svg — sweeping the spectral radius ═══════════ */
(function () {
  const svg = d3.select("#sr-svg");
  if (svg.empty()) return;
  const W = 760, H = 410;
  const He = document.getElementById("sr-dh"), De = document.getElementById("sr-D"),
    Ne = document.getElementById("sr-nl"), Ve = document.getElementById("sr-v");

  /* the ACTUAL backward operator of a tanh recurrence, assembled from the
     states of a real forward pass: Dₜ Wᵀ · Dₜ₋₁ Wᵀ · … — no approximation */
  function backOp(Wm, H2, from, to) {          // from = T-1 down to `to`
    const dh = Wm.length, WT = DL.transpose(Wm);
    let M = DL.eye(dh);
    for (let j = from; j > to; j--) {
      const D = DL.zeros2(dh, dh);
      for (let i = 0; i < dh; i++) D[i][i] = 1 - H2[j][i] * H2[j][i];
      M = DL.matmul(M, DL.matmul(D, WT));
    }
    return M;
  }
  function curveFor(dh, rho, D, mode, seed) {
    const Wm = RN.mat(dh, seed || 9, rho);
    if (mode === "lin") return { vals: d3.range(1, D + 1).map(k => DL.specNorm(DL.matPow(Wm, k), 200, 3)), gate: 1 };
    const drive = mode === "tanhbig" ? 3.2 : 0.7;
    const p = { kind: "vanilla", dx: 3, dh: dh, U: RN.mat(dh, 77).slice(0, 3).map(r => r.slice(0, dh)).map(r => r.map(v => v * drive)), W: Wm, b: DL.zeros(dh) };
    const X = RN.seq(D + 1, 3, 55).map(r => r.map(v => v * drive));
    const st = DL.CELL.vanilla.forward(p, X);
    const gate = d3.mean(st.H.map(h => d3.mean(h.map(v => 1 - v * v))));
    const vals = d3.range(1, D + 1).map(k => DL.specNorm(backOp(Wm, st.H, D, D - k), 200, 3));
    return { vals: vals, gate: gate };
  }

  function draw() {
    const dh = +He.value, D = +De.value, mode = Ne.value, view = Ve.value;
    document.getElementById("sr-dhv").textContent = dh;
    document.getElementById("sr-Dv").textContent = D;
    const f = DL.frame(svg, W, H, { l: 52, r: 176, t: 26, b: 40 }), g = f.g;

    const rhos = d3.range(0.5, 1.601, 0.05).map(v => Math.round(v * 100) / 100);
    const data = rhos.map(r => curveFor(dh, r, D, mode));
    const meanGate = d3.mean(data.map(d => d.gate));

    if (view === "heat") {
      const x = d3.scaleLinear().domain([rhos[0], rhos[rhos.length - 1]]).range([0, f.iw]);
      const y = d3.scaleLinear().domain([1, D]).range([f.ih, 0]);
      const cw = f.iw / rhos.length, ch = f.ih / D;
      const cs = d3.scaleSequential(d3.interpolateRdBu).domain([8, -8]);
      data.forEach((d, i) => d.vals.forEach((v, k) => {
        g.append("rect").attr("x", i * cw).attr("y", f.ih - (k + 1) * ch).attr("width", cw + 0.5).attr("height", ch + 0.5)
          .attr("shape-rendering", "crispEdges").attr("fill", cs(Math.log10(Math.max(v, 1e-30))));
      }));
      /* the contour where the measured norm crosses 1 */
      const path = [];
      for (let k = 1; k <= D; k++) {
        let cross = null;
        for (let i = 0; i + 1 < rhos.length; i++) {
          const a = data[i].vals[k - 1], b = data[i + 1].vals[k - 1];
          if ((a - 1) * (b - 1) <= 0 && a !== b) { cross = rhos[i] + (rhos[i + 1] - rhos[i]) * (1 - a) / (b - a); break; }
        }
        if (cross !== null) path.push([x(cross), y(k)]);
      }
      if (path.length > 1) DL.curve(g, path, { stroke: "#ffffff", w: 2 });
      DL.axisB(g, x, f.ih, 6, "ρ(W)");
      DL.axisL(g, y, 5, "distance d");
      RN.note(g, 6, 14, "white contour: ‖J‖₂ = 1", "#ffffff", 9.5);
    } else if (view === "slices") {
      const show = [0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5];
      const x = d3.scaleLinear().domain([1, D]).range([0, f.iw]);
      const allv = show.map(r => data[rhos.indexOf(r)]).filter(Boolean).reduce((a, d) => a.concat(d.vals), []).filter(v => v > 0);
      const y = d3.scaleLog().domain([Math.max(1e-22, Math.min(...allv) / 3), Math.max(...allv) * 3]).range([f.ih, 0]).clamp(true);
      DL.gridY(g, y, f.iw, 5);
      DL.axisB(g, x, f.ih, 6, "distance d");
      DL.axisL(g, y, 5, "‖J‖₂", v => DL.fmtE(v, 0));
      g.append("rect").attr("x", 0).attr("y", y(3.4e38)).attr("width", f.iw)
        .attr("height", Math.max(0, y(1.18e-38) - y(3.4e38))).attr("fill", DC.good).attr("fill-opacity", 0.05);
      show.forEach((r, i) => {
        const d = data[rhos.indexOf(r)];
        if (!d) return;
        const col = d3.interpolateTurbo(i / (show.length - 1));
        DL.curve(g, d.vals.map((v, k) => [x(k + 1), y(Math.max(v, 1e-300))]), { stroke: col, w: 1.9 });
        DL.curve(g, d3.range(1, D + 1).map(k => [x(k), y(Math.max(1e-300, Math.pow(r, k)))]),
          { stroke: col, w: 1, dash: "3 3", op: 0.6 });
      });
      DL.legend(g, show.map((r, i) => ({ label: "ρ = " + DL.fmt(r, 2), color: d3.interpolateTurbo(i / (show.length - 1)) })),
        f.iw - 66, 6, { vertical: true, gap: 14, font: 9.5 });
    } else {
      const rates = data.map(d => RN.logSlope(d3.range(1, D + 1), d.vals));
      const x = d3.scaleLinear().domain([rhos[0], rhos[rhos.length - 1]]).range([0, f.iw]);
      const y = d3.scaleLinear().domain([0, Math.max(1.7, ...rates.filter(isFinite))]).range([f.ih, 0]);
      DL.gridY(g, y, f.iw, 5); DL.gridX(g, x, f.ih, 5);
      DL.axisB(g, x, f.ih, 6, "ρ(W)");
      DL.axisL(g, y, 5, "measured per-step factor");
      DL.curve(g, rhos.map((r, i) => [x(r), y(r)]), { stroke: DC.muted, w: 1.2, dash: "4 3" });
      DL.curve(g, rhos.map((r, i) => [x(r), y(Math.min(y.domain()[1], rates[i] || 0))]), { stroke: DC.accent, w: 2.2 });
      g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y(1)).attr("y2", y(1))
        .attr("stroke", DC.a2).attr("stroke-dasharray", "3 3");
      DL.legend(g, [{ label: "measured", color: DC.accent }, { label: "identity — the linear prediction", color: DC.muted, dash: "4 3" }],
        6, f.ih + 30, { vertical: false, step: 200, font: 9.5 });
    }

    /* where the measured rate crosses 1 */
    const rates = data.map(d => RN.logSlope(d3.range(1, D + 1), d.vals));
    let cross = null;
    for (let i = 0; i + 1 < rhos.length; i++) {
      if (isFinite(rates[i]) && isFinite(rates[i + 1]) && (rates[i] - 1) * (rates[i + 1] - 1) <= 0 && rates[i] !== rates[i + 1])
        { cross = rhos[i] + (rhos[i + 1] - rhos[i]) * (1 - rates[i]) / (rates[i + 1] - rates[i]); break; }
    }
    const iOne = rhos.indexOf(1);
    const sx = f.iw + 16;
    const kv = DL.kv(g, sx, 6, { keyW: 122, size: 10.5, lead: 16 });
    kv("recurrence", mode === "lin" ? "linear" : mode === "tanh" ? "tanh, mild drive" : "tanh, saturating");
    kv("mean tanh gate", mode === "lin" ? "1 (none)" : DL.fmt(meanGate, 4), DC.a2);
    kv("rate at ρ = 1", DL.fmt(rates[iOne], 4), Math.abs(rates[iOne] - 1) < 0.02 ? DC.good : DC.a2);
    kv("ρ where rate = 1", cross === null ? "outside the sweep" : DL.fmt(cross, 3), DC.ink);
    kv("predicted ρ·gate = 1 at", mode === "lin" ? "ρ = 1" : DL.fmt(1 / Math.max(1e-9, meanGate), 3), DC.muted);
    let dF = null;
    for (let k = 1; k <= 500; k++) if (Math.pow(Math.max(1e-9, rates[iOne]), k) < 1.18e-38) { dF = k; break; }
    kv("d to the float32 floor at ρ=1", dF === null ? "> 500" : String(dF));
    RN.note(g, sx, 118, "the threshold is ρ = 1 for a linear", DC.muted, 9.5);
    RN.note(g, sx, 130, "recurrence and moves RIGHT once the", DC.muted, 9.5);
    RN.note(g, sx, 142, "tanh gate takes its own factor out.", DC.muted, 9.5);

    document.getElementById("sr-readout").innerHTML =
      `${mode === "lin" ? "linear recurrence" : mode === "tanh" ? "tanh, mild drive" : "tanh, driven into saturation"} · ` +
      `measured per-step growth factor at ρ(W) = 1 is <b>${DL.fmt(rates[iOne], 4)}</b> ` +
      `${mode === "lin" ? "— exactly the linear prediction" : `, because the tanh contributes a mean gate of <b>${DL.fmt(meanGate, 4)}</b>`} · ` +
      `the measured rate reaches 1 at ρ(W) = <b>${cross === null ? "outside the sweep" : DL.fmt(cross, 3)}</b>` +
      `${mode === "lin" ? "" : `, against the prediction 1/gate = <b>${DL.fmt(1 / Math.max(1e-9, meanGate), 3)}</b>`} · ` +
      `at ρ = 1 a gradient of 1 reaches the float32 floor after <b>${dF === null ? "> 500" : dF}</b> steps.`;
  }
  [He, De].forEach(e => e.addEventListener("input", draw));
  [Ne, Ve].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 19 · #cr-svg — three candidate criteria ═══════════ */
(function () {
  const svg = d3.select("#cr-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Me = document.getElementById("cr-m"), De = document.getElementById("cr-d"),
    Se = document.getElementById("cr-s");

  function build(key, s) {
    const c = Math.cos(Math.PI / 3), sn = Math.sin(Math.PI / 3);
    const base = {
      all5: [[.5, .5, .5, .5], [.5, .5, .5, .5], [.5, .5, .5, .5], [.5, .5, .5, .5]],
      all3: DL.zeros2(4, 4).map(r => r.map(() => 0.3)),
      all2: DL.zeros2(8, 8).map(r => r.map(() => 0.2)),
      rot: [[0.9 * c, -0.9 * sn], [0.9 * sn, 0.9 * c]],
      shear: [[0.9, 9], [0, 0.9]],
      rand: RN.mat(5, 13),
      orth: RN.ortho(5, 13),
      diag: (() => { const A = DL.zeros2(5, 5); const r = DL.rng(3); for (let i = 0; i < 5; i++) A[i][i] = 0.4 + 0.6 * r(); return A; })()
    }[key];
    return base.map(r => r.map(v => v * s));
  }

  function draw() {
    const key = Me.value, d = +De.value, s = +Se.value;
    document.getElementById("cr-dv").textContent = d;
    document.getElementById("cr-sv").textContent = DL.fmt(s, 2);
    const f = DL.frame(svg, W, H, { l: 16, r: 16, t: 24, b: 24 }), g = f.g;

    const A = build(key, s), n = A.length;
    const maxEnt = Math.max(...RN.flat(A).map(Math.abs));
    const nrm2 = DL.specNorm(A, 400, 3);
    const rho = DL.specRad(A);
    const powers = d3.range(1, 121).map(k => DL.specNorm(DL.matPow(A, k), 240, 3));
    const at = powers[Math.min(d, 120) - 1];
    const peak = Math.max(...powers), peakAt = powers.indexOf(peak) + 1;
    const interiorPeak = peakAt > 1 && peakAt < powers.length;   /* a real hump, not the end of a rising curve */
    let firstBelow = null;
    for (let k = 0; k < powers.length; k++) if (powers[k] < 1) { firstBelow = k + 1; break; }
    const verdictTruth = rho < 1 ? "vanishes eventually" : "explodes";

    /* ---- left: the matrix ---- */
    RN.title(g, 0, 0, "the matrix");
    const cw = Math.min(38, 160 / n);
    const mm = Math.max(...RN.flat(A).map(Math.abs), 1e-12);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      g.append("rect").attr("x", j * cw).attr("y", 14 + i * cw).attr("width", cw - 1).attr("height", cw - 1)
        .attr("rx", 2).attr("fill", RN.signColor(A[i][j], mm)).attr("stroke", DC.line).attr("stroke-opacity", 0.5);
      if (n <= 5) g.append("text").attr("x", j * cw + cw / 2 - 0.5).attr("y", 14 + i * cw + cw / 2 + 3)
        .attr("text-anchor", "middle").attr("font-size", 8.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink)
        .text(Math.abs(A[i][j]) < 1e-9 ? "0" : DL.fmt(A[i][j], 2));
    }

    /* ---- the three candidate criteria ---- */
    const ty = 14 + n * cw + 26;
    RN.title(g, 0, ty - 10, "three candidate criteria");
    const rows = [
      ["max |entry|", maxEnt, maxEnt < 1 ? "vanishes" : "explodes"],
      ["‖A‖₂  (spectral norm)", nrm2, nrm2 < 1 ? "vanishes" : "explodes"],
      ["ρ(A)  (spectral radius)", rho, rho < 1 ? "vanishes" : "explodes"]
    ];
    rows.forEach((r, i) => {
      const yy = ty + i * 24;
      RN.note(g, 0, yy + 10, r[0], DC.muted, 10);
      g.append("text").attr("x", 168).attr("y", yy + 10).attr("text-anchor", "end").attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(DL.fmt(r[1], 4));
      const ok = (r[2] === "vanishes") === (rho < 1);
      RN.chip(g, 180, yy + 10, r[2], ok ? DC.good : DC.bad);
    });
    RN.note(g, 0, ty + 3 * 24 + 14, "green = agrees with the measurement · red = wrong", DC.muted, 9);
    RN.note(g, 0, ty + 3 * 24 + 28, "measured verdict: " + verdictTruth, DC.a2, 10.5);

    /* ---- right: ‖Aᵈ‖ against d ---- */
    const gx = 300, gw = f.iw - gx, gh = 250;
    RN.title(g, gx, 0, "‖Aᵈ‖₂ against d — the measurement");
    const gg = g.append("g").attr("transform", `translate(${gx},16)`);
    const x = d3.scaleLinear().domain([1, 120]).range([0, gw - 12]);
    const refL = d3.range(1, 121).map(k => Math.pow(rho, k));
    const refU = d3.range(1, 121).map(k => Math.pow(nrm2, k));
    const allv = powers.concat(refL).concat(refU).filter(v => v > 0 && isFinite(v));
    const y = d3.scaleLog().domain([Math.max(1e-20, Math.min(...allv) / 3), Math.max(...allv) * 3]).range([gh, 0]).clamp(true);
    DL.gridY(gg, y, gw - 12, 5);
    DL.axisB(gg, x, gh, 6, "d");
    DL.axisL(gg, y, 5, "‖Aᵈ‖₂", v => DL.fmtE(v, 0));
    gg.append("line").attr("x1", 0).attr("x2", gw - 12).attr("y1", y(1)).attr("y2", y(1))
      .attr("stroke", DC.ink).attr("stroke-dasharray", "4 3").attr("stroke-opacity", 0.8);
    DL.curve(gg, refU.map((v, i) => [x(i + 1), y(Math.max(v, 1e-300))]), { stroke: DC.bad, w: 1.1, dash: "3 3" });
    DL.curve(gg, refL.map((v, i) => [x(i + 1), y(Math.max(v, 1e-300))]), { stroke: DC.good, w: 1.1, dash: "3 3" });
    DL.curve(gg, powers.map((v, i) => [x(i + 1), y(Math.max(v, 1e-300))]), { stroke: DC.accent, w: 2.2 });
    if (interiorPeak) {
      gg.append("circle").attr("cx", x(peakAt)).attr("cy", y(peak)).attr("r", 3.4).attr("fill", DC.a2);
      RN.note(gg, x(peakAt) + 6, y(peak) - 4, "transient peak " + DL.fmt(peak, 2) + " at d = " + peakAt, DC.a2, 9);
    }
    if (firstBelow) {
      gg.append("line").attr("x1", x(firstBelow)).attr("x2", x(firstBelow)).attr("y1", 0).attr("y2", gh)
        .attr("stroke", DC.good).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.7);
      RN.note(gg, x(firstBelow) + 4, 12, "first d with ‖Aᵈ‖ < 1: " + firstBelow, DC.good, 9);
    }
    gg.append("line").attr("x1", x(Math.min(d, 120))).attr("x2", x(Math.min(d, 120))).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.a2).attr("stroke-dasharray", "3 3");
    DL.legend(gg, [{ label: "measured ‖Aᵈ‖₂", color: DC.accent }, { label: "ρᵈ (the rate)", color: DC.good, dash: "3 3" },
    { label: "‖A‖₂ᵈ (the bound)", color: DC.bad, dash: "3 3" }], 4, gh + 30, { vertical: false, step: 140, font: 9.5 });

    const wrong = rows.filter(r => (r[2] === "vanishes") !== (rho < 1)).map(r => r[0]);
    document.getElementById("cr-readout").innerHTML =
      `<b>${Me.options[Me.selectedIndex].text}</b> × ${DL.fmt(s, 2)} · max|entry| <b>${DL.fmt(maxEnt, 4)}</b>, ` +
      `‖A‖₂ <b>${DL.fmt(nrm2, 4)}</b>, ρ(A) <b>${DL.fmt(rho, 6)}</b> · measured ‖Aᵈ‖₂ at d = ${d} is <b>${DL.fmtE(at, 3)}</b>` +
      (interiorPeak ? `, rising to a transient peak of <b>${DL.fmt(peak, 3)}</b> at d = ${peakAt} before turning over`
        : `, still ${powers[powers.length - 1] > powers[0] ? "rising" : "falling"} monotonically at the end of the plotted range`) +
      (firstBelow ? `, and first falling below 1 at d = <b>${firstBelow}</b>` : ", and never falling below 1 within 120 steps") +
      ` · ${wrong.length ? `the criterion "<b>${wrong.join('" and "')}</b>" gets this matrix <b>wrong</b>.` : "all three criteria agree here."}`;
  }
  [De, Se].forEach(e => e.addEventListener("input", draw));
  Me.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 20 · #td-svg — tied against fresh weights ═══════════ */
(function () {
  const svg = d3.select("#td-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Ne = document.getElementById("td-n"), Le = document.getElementById("td-L"),
    Ge = document.getElementById("td-g"), Ke = document.getElementById("td-k"),
    Re = document.getElementById("td-res");

  function draw() {
    const n = +Ne.value, L = +Le.value, gain = +Ge.value, K = +Ke.value, res = Re.checked;
    document.getElementById("td-nv").textContent = n;
    document.getElementById("td-Lv").textContent = L;
    document.getElementById("td-gv").textContent = DL.fmt(gain, 2);
    document.getElementById("td-kv").textContent = K;
    const f = DL.frame(svg, W, H, { l: 54, r: 190, t: 24, b: 40 }), g = f.g;

    const sd = gain / Math.sqrt(n);
    const tiedRuns = [], freshRuns = [];
    /* ONE matrix, drawn once and shared by every tied trajectory, so that the
       spread between tied trajectories is due ONLY to the starting direction. */
    const rW = DL.rng(31), Wt = DL.zeros2(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Wt[i][j] = DL.randn(rW) * sd;
    for (let k = 0; k < K; k++) {
      const rt = DL.rng(1000 + k), rf = DL.rng(5000 + k);
      let v = []; for (let i = 0; i < n; i++) v.push(DL.randn(rt));
      let nv = RN.norm(v); v = v.map(x => x / nv);
      const tr = [0];
      let cur = v.slice();
      for (let l = 1; l <= L; l++) {
        let nx = DL.vecmat(cur, Wt);
        if (res) for (let i = 0; i < n; i++) nx[i] += cur[i];
        cur = nx; tr.push(Math.log(Math.max(RN.norm(cur), 1e-300)));
      }
      tiedRuns.push(tr);
      /* fresh: a NEW matrix at every step */
      let v2 = []; for (let i = 0; i < n; i++) v2.push(DL.randn(rf));
      let nv2 = RN.norm(v2); v2 = v2.map(x => x / nv2);
      const fr = [0];
      let cur2 = v2.slice();
      for (let l = 1; l <= L; l++) {
        const Wf = DL.zeros2(n, n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Wf[i][j] = DL.randn(rf) * sd;
        let nx = DL.vecmat(cur2, Wf);
        if (res) for (let i = 0; i < n; i++) nx[i] += cur2[i];
        cur2 = nx; fr.push(Math.log(Math.max(RN.norm(cur2), 1e-300)));
      }
      freshRuns.push(fr);
    }
    const allv = tiedRuns.concat(freshRuns).reduce((a, r) => a.concat(r), []);
    const x = d3.scaleLinear().domain([0, L]).range([0, f.iw]);
    const y = d3.scaleLinear().domain([Math.min(...allv) - 1, Math.max(...allv) + 1]).range([f.ih, 0]);
    DL.gridY(g, y, f.iw, 5);
    DL.axisB(g, x, f.ih, 6, "depth / steps");
    DL.axisL(g, y, 5, "log ‖signal‖");
    const band = (runs, col) => {
      const mean = [], sdv = [];
      for (let l = 0; l <= L; l++) {
        const vs = runs.map(r => r[l]);
        mean.push(d3.mean(vs)); sdv.push(runs.length > 1 ? d3.deviation(vs) || 0 : 0);
      }
      const up = mean.map((m, l) => [x(l), y(m + sdv[l])]);
      const dn = mean.map((m, l) => [x(l), y(m - sdv[l])]).reverse();
      g.append("path").attr("d", up.concat(dn).map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ") + " Z")
        .attr("fill", col).attr("fill-opacity", 0.13).attr("stroke", "none");
      runs.forEach(r => DL.curve(g, r.map((v, l) => [x(l), y(v)]), { stroke: col, w: 0.9, op: 0.45 }));
      DL.curve(g, mean.map((m, l) => [x(l), y(m)]), { stroke: col, w: 2.2 });
      return { mean: mean[L], sd: sdv[L] };
    };
    const T2 = band(tiedRuns, DC.a2), F2 = band(freshRuns, DC.accent);
    const rwPred = Math.sqrt(L / (2 * n));      /* what a log-magnitude random walk predicts */
    DL.legend(g, [{ label: "tied — the SAME matrix every step", color: DC.a2 },
    { label: "fresh matrix every step", color: DC.accent }],
      4, f.ih + 30, { vertical: false, step: 232, font: 9.5 });

    const sx = f.iw + 16;
    const kv = DL.kv(g, sx, 6, { keyW: 118, size: 10.5, lead: 16 });
    kv("tied: mean log ‖·‖", DL.fmt(T2.mean, 3), DC.a2);
    kv("tied: sd", DL.fmt(T2.sd, 4), DC.a2);
    kv("fresh: mean log ‖·‖", DL.fmt(F2.mean, 3), DC.accent);
    kv("fresh: sd", DL.fmt(F2.sd, 4), DC.accent);
    kv("sd ratio, fresh / tied", DL.fmt(F2.sd / Math.max(1e-9, T2.sd), 2) + "×", DC.ink);
    kv("random-walk prediction √(L/2n)", DL.fmt(rwPred, 3), DC.muted);
    kv("ρ(W) of the shared matrix", DL.fmt(DL.specRad(Wt), 4), DC.a2);
    const out = tiedRuns.concat(freshRuns).filter(r => Math.abs(r[L]) > Math.log(3.4e38)).length;
    kv("runs outside float32", String(out) + " of " + (2 * K), out ? DC.bad : DC.good);
    kv("residual path", res ? "on" : "off", res ? DC.good : DC.muted);
    RN.note(g, sx, 148, "the tied family is a set of straight", DC.muted, 9.5);
    RN.note(g, sx, 160, "lines whose spread does NOT widen;", DC.muted, 9.5);
    RN.note(g, sx, 172, "the fresh family is a random walk", DC.muted, 9.5);
    RN.note(g, sx, 184, "whose spread grows like √L.", DC.muted, 9.5);

    document.getElementById("td-readout").innerHTML =
      `gain ${DL.fmt(gain, 2)}, width ${n}, depth ${L}, ${K} trajector${K === 1 ? "y" : "ies"} each` +
      `${res ? ", residual path ON" : ""} · at the final layer, tied weights give mean log-norm ` +
      `<b>${DL.fmt(T2.mean, 3)}</b> ± <b>${DL.fmt(T2.sd, 4)}</b> and fresh weights <b>${DL.fmt(F2.mean, 3)}</b> ± ` +
      `<b>${DL.fmt(F2.sd, 4)}</b> — the fresh spread is <b>${DL.fmt(F2.sd / Math.max(1e-9, T2.sd), 2)}×</b> the tied one · ` +
      `all tied trajectories share ONE matrix, ρ(W) = ${DL.fmt(DL.specRad(Wt), 4)}, so their spread comes only from the starting ` +
      `direction and does not widen with depth; the fresh spread does, and the random-walk prediction √(L/2n) = ` +
      `<b>${DL.fmt(rwPred, 3)}</b> against a measured <b>${DL.fmt(F2.sd, 3)}</b> · <b>${out}</b> of ${2 * K} runs left the float32 range.`;
  }
  [Ne, Le, Ge, Ke].forEach(e => e.addEventListener("input", draw));
  Re.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 21 · #cl-svg — the cliff, and an honest negative result ═══════════ */
(function () {
  const svg = d3.select("#cl-svg");
  if (svg.empty()) return;
  const W = 760, H = 410;
  const Te = document.getElementById("cl-T"), Ve = document.getElementById("cl-v"),
    Le = document.getElementById("cl-lr"), Ke = document.getElementById("cl-k"),
    Ne = document.getElementById("cl-n");

  function draw() {
    const T = +Te.value, thr = +Ve.value, lr = +Le.value, kind = Ke.value, steps = +Ne.value;
    document.getElementById("cl-Tv").textContent = T;
    document.getElementById("cl-vv").textContent = DL.fmt(thr, 2);
    document.getElementById("cl-lrv").textContent = DL.fmt(lr, 3);
    document.getElementById("cl-nv").textContent = steps;
    const f = DL.frame(svg, W, H, { l: 46, r: 250, t: 24, b: 40 }), g = f.g;

    const dx = 2, dh = 2, dy = 1;
    const base = RN.net("vanilla", dx, dh, dy, { seed: 23, out: "sigmoid", rho: 0.6, uScale: 0.5 });
    const X = RN.seq(T, dx, 231), Y = RN.targets(T, dy, "sigmoid", 233, "last");
    const at = (a2, b2) => { base.p.W[0][0] = a2; base.p.W[1][1] = b2; return base; };
    const lossAt = (a2, b2) => DL.seqLoss(at(a2, b2), X, Y);
    const gradAt = (a2, b2) => { const gr = DL.seqBPTT(at(a2, b2), X, Y); return [gr.dW[0][0], gr.dW[1][1]]; };

    const lo = -1.6, hi = 1.6, N = 48;
    const gs = d3.range(N).map(i => lo + (hi - lo) * i / (N - 1));
    const Z = [], allZ = [], GN = [], allG = [];
    let steepest = { n: -1, a: 0, b: 0 }, flattest = { n: Infinity };
    for (let i = 0; i < N; i++) {
      const rowZ = [], rowG = [];
      for (let j = 0; j < N; j++) {
        const v = lossAt(gs[j], gs[i]); rowZ.push(v); allZ.push(v);
        const gn = Math.hypot(...gradAt(gs[j], gs[i])); rowG.push(gn); allG.push(gn);
        if (gn > steepest.n) steepest = { n: gn, a: gs[j], b: gs[i] };
        if (gn < flattest.n) flattest = { n: gn };
      }
      Z.push(rowZ); GN.push(rowG);
    }
    const gw = f.iw, gh = 300;
    const x = d3.scaleLinear().domain([lo, hi]).range([0, gw]);
    const y = d3.scaleLinear().domain([lo, hi]).range([gh, 0]);
    const cs = d3.scaleSequentialLog(d3.interpolateMagma).domain([Math.max(1e-6, d3.min(allG)), d3.max(allG)]);
    const cwx = gw / N, cwy = gh / N;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
      g.append("rect").attr("x", j * cwx).attr("y", gh - (i + 1) * cwy).attr("width", cwx + 0.6).attr("height", cwy + 0.6)
        .attr("shape-rendering", "crispEdges").attr("fill", cs(Math.max(1e-9, GN[i][j])));
    /* loss iso-lines, drawn by marking the cells a level crosses. No dependency
       on d3-contour, which this vendored build does not necessarily carry. */
    {
      const zmin = d3.min(allZ), zmax = d3.max(allZ);
      const levels = d3.range(1, 7).map(k => zmin + (zmax - zmin) * k / 7);
      levels.forEach(lv => {
        for (let i = 0; i + 1 < N; i++) for (let j = 0; j + 1 < N; j++) {
          const a1 = Z[i][j], a2 = Z[i][j + 1], a3 = Z[i + 1][j];
          if ((a1 - lv) * (a2 - lv) < 0 || (a1 - lv) * (a3 - lv) < 0)
            g.append("rect").attr("x", j * cwx).attr("y", gh - (i + 1) * cwy)
              .attr("width", cwx).attr("height", cwy)
              .attr("fill", "#ffffff").attr("fill-opacity", 0.14).attr("shape-rendering", "crispEdges");
        }
      });
    }
    /* the exact ρ(W) = 1 locus */
    const locus = [];
    for (let i = 0; i < 200; i++) {
      const a2 = lo + (hi - lo) * i / 199;
      let l = lo, r = hi, found = null;
      const rf = b2 => DL.specRad([[a2, base.p.W[0][1]], [base.p.W[1][0], b2]]) - 1;
      if (rf(l) * rf(r) < 0) { for (let it = 0; it < 40; it++) { const m = (l + r) / 2; if (rf(l) * rf(m) <= 0) r = m; else l = m; } found = (l + r) / 2; }
      if (found !== null) locus.push([x(a2), y(found)]);
    }
    if (locus.length > 1) DL.curve(g, locus, { stroke: DC.teal, w: 1.8, dash: "5 3" });
    DL.axisB(g, x, gh, 5, "W₀₀");
    DL.axisL(g, y, 5, "W₁₁");
    RN.note(g, 4, 12, "colour: ‖∂L/∂W‖ · white contours: the loss · dashed: ρ(W) = 1", "#ffffff", 9.5);

    /* the two single steps FROM the steepest point */
    const [sga, sgb] = gradAt(steepest.a, steepest.b);
    const raw = [-lr * sga, -lr * sgb];
    let clipped;
    if (kind === "norm") { const n2 = Math.hypot(...raw), fct = n2 > lr * thr ? (lr * thr) / n2 : 1; clipped = [raw[0] * fct, raw[1] * fct]; }
    else if (kind === "value") clipped = [-lr * DL.clamp(sga, -thr, thr), -lr * DL.clamp(sgb, -thr, thr)];
    else clipped = raw.slice();
    const sx0 = x(steepest.a), sy0 = y(steepest.b);
    g.append("circle").attr("cx", sx0).attr("cy", sy0).attr("r", 5).attr("fill", DC.good).attr("stroke", "#fff").attr("stroke-width", 1.2);
    const cl6 = v => DL.clamp(v, lo - 0.05, hi + 0.05);
    DL.arrow(g, sx0, sy0, x(cl6(steepest.a + raw[0])), y(cl6(steepest.b + raw[1])), { color: DC.bad, w: 2.4, head: 8 });
    if (kind !== "none")
      DL.arrow(g, sx0, sy0, x(cl6(steepest.a + clipped[0])), y(cl6(steepest.b + clipped[1])), { color: DC.accent, w: 2.4, head: 8 });
    RN.note(g, sx0 + 8, sy0 - 8, "steepest point", DC.good, 9.5);

    /* the descent trajectory from that same point */
    let a = steepest.a, b = steepest.b, fired = 0, big = 0, left = false;
    const traj = [[a, b]];
    for (let s2 = 0; s2 < steps; s2++) {
      const [ga, gb] = gradAt(a, b), n2 = Math.hypot(ga, gb);
      big = Math.max(big, n2);
      let da = -lr * ga, db = -lr * gb;
      if (kind === "norm") { const fct = n2 > thr ? thr / n2 : 1; if (fct < 1) fired++; da *= fct; db *= fct; }
      else if (kind === "value") { if (Math.abs(ga) > thr || Math.abs(gb) > thr) fired++; da = -lr * DL.clamp(ga, -thr, thr); db = -lr * DL.clamp(gb, -thr, thr); }
      a += da; b += db;
      if (!isFinite(a) || Math.abs(a) > 6 || Math.abs(b) > 6) { left = true; break; }
      traj.push([a, b]);
    }
    DL.curve(g, traj.map(p2 => [x(cl6(p2[0])), y(cl6(p2[1]))]), { stroke: DC.a2, w: 1.6 });
    traj.forEach(p2 => g.append("circle").attr("cx", x(cl6(p2[0]))).attr("cy", y(cl6(p2[1]))).attr("r", 1.6).attr("fill", DC.a2).attr("fill-opacity", 0.85));

    /* right: the gradient-norm histogram over the whole grid */
    const sx = f.iw + 16, sw = 220, sh = 140;
    RN.title(g, sx, 0, "‖∂L/∂W‖ over the whole grid");
    const gg = g.append("g").attr("transform", `translate(${sx},16)`);
    const lg = allG.filter(v => v > 0).map(Math.log10);
    const bins = d3.bin().domain([d3.min(lg), d3.max(lg)]).thresholds(22)(lg);
    const hx = d3.scaleLinear().domain([d3.min(lg), d3.max(lg)]).range([0, sw - 40]);
    const hy = d3.scaleLinear().domain([0, d3.max(bins, bb => bb.length) || 1]).range([sh, 0]);
    bins.forEach(bb => gg.append("rect").attr("x", hx(bb.x0)).attr("y", hy(bb.length))
      .attr("width", Math.max(1, hx(bb.x1) - hx(bb.x0) - 1)).attr("height", sh - hy(bb.length))
      .attr("fill", DC.a2).attr("fill-opacity", 0.7));
    DL.axisB(gg, hx, sh, 4, "log₁₀ ‖g‖");
    if (kind !== "none" && Math.log10(thr) >= hx.domain()[0] && Math.log10(thr) <= hx.domain()[1]) {
      gg.append("line").attr("x1", hx(Math.log10(thr))).attr("x2", hx(Math.log10(thr))).attr("y1", 0).attr("y2", sh)
        .attr("stroke", DC.bad).attr("stroke-dasharray", "4 3");
      RN.note(gg, hx(Math.log10(thr)) + 3, 10, "clip", DC.bad, 9);
    } else if (kind !== "none") {
      RN.note(gg, 2, 10, "the threshold is off this scale entirely", DC.bad, 9);
    }
    const kv = DL.kv(g, sx, 194, { keyW: 172, size: 10.5, lead: 15 });
    kv("steepest ‖g‖ on the grid", RN.P(steepest.n, 4), DC.bad);
    kv("flattest ‖g‖", RN.P(flattest.n, 3));
    kv("dynamic range", RN.P(steepest.n / Math.max(1e-300, flattest.n), 3) + "×", DC.a2);
    kv("one step, unclipped", RN.P(Math.hypot(...raw), 4), DC.bad);
    kv("one step, clipped", kind === "none" ? "— (clipping off)" : RN.P(Math.hypot(...clipped), 4), DC.accent);
    kv("clip fired on the trajectory", fired + " of " + traj.length, fired ? DC.a2 : DC.muted);
    kv("largest ‖g‖ the path met", RN.P(big, 4));
    RN.note(g, sx, 310, "NEGATIVE RESULT: the descent path slides", DC.bad, 9.5);
    RN.note(g, sx, 322, "AWAY from the ridge, so on this two-", DC.muted, 9.5);
    RN.note(g, sx, 334, "parameter slice it never meets a gradient", DC.muted, 9.5);
    RN.note(g, sx, 346, "large enough to matter. The catastrophic", DC.muted, 9.5);
    RN.note(g, sx, 358, "step is a high-dimensional, minibatch,", DC.muted, 9.5);
    RN.note(g, sx, 370, "late-training event this slice cannot show.", DC.muted, 9.5);

    document.getElementById("cl-readout").innerHTML =
      `T = ${T}, ${Ke.options[Ke.selectedIndex].text}${kind === "none" ? "" : " at " + DL.fmt(thr, 2)}, learning rate ${DL.fmt(lr, 3)} · ` +
      `over the grid the gradient norm ranges from <b>${RN.P(flattest.n, 3)}</b> to <b>${RN.P(steepest.n, 4)}</b>, ` +
      `a dynamic range of <b>${RN.P(steepest.n / Math.max(1e-300, flattest.n), 3)}×</b> — that range IS the cliff · ` +
      `one step from the steepest point has length <b>${RN.P(Math.hypot(...raw), 4)}</b> unclipped and ` +
      `<b>${kind === "none" ? "—" : RN.P(Math.hypot(...clipped), 4)}</b> clipped · ` +
      `<b>honest negative result:</b> the descent trajectory started at that same steepest point moves away from the ridge, ` +
      `meets a largest gradient of only <b>${RN.P(big, 4)}</b>, and triggers the clip on <b>${fired}</b> of ${traj.length} steps` +
      `${left ? " before leaving the plotted region" : ""}. A full-batch path on a two-parameter slice does not reproduce the ` +
      `catastrophic single step of §21 — that is a high-dimensional, minibatch phenomenon, and this figure shows the surface it lives on rather than pretending to reproduce it.`;
  }
  [Te, Ve, Le, Ne].forEach(e => e.addEventListener("input", draw));
  Ke.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 22 · #rm-svg — four remedies, measured ═══════════ */
(function () {
  const svg = d3.select("#rm-svg");
  if (svg.empty()) return;
  const W = 760, H = 380;
  const De = document.getElementById("rm-D"), Re = document.getElementById("rm-rho"),
    Ae = document.getElementById("rm-a"), Me = document.getElementById("rm-m");

  function draw() {
    const D = +De.value, rho = +Re.value, alpha = +Ae.value, m = +Me.value;
    document.getElementById("rm-Dv").textContent = D;
    document.getElementById("rm-rhov").textContent = DL.fmt(rho, 2);
    document.getElementById("rm-av").textContent = DL.fmt(alpha, 2);
    document.getElementById("rm-mv").textContent = m;
    const f = DL.frame(svg, W, H, { l: 54, r: 196, t: 24, b: 40 }), g = f.g;

    const dx = 3, dh = 8, T = D + 1;
    const X = RN.seq(T, dx, 501);
    const Wm = RN.mat(dh, 9, rho);
    const U = RN.mat(dh, 77).slice(0, dx).map(r => r.slice(0, dh)).map(r => r.map(v => v * 0.6));
    const WT = DL.transpose(Wm);

    /* plain: the true backward operator of the tanh recurrence */
    const p = { kind: "vanilla", dx: dx, dh: dh, U: U, W: Wm, b: DL.zeros(dh) };
    const st = DL.CELL.vanilla.forward(p, X);
    const back = (from, to, mul) => {
      let M = DL.eye(dh);
      for (let j = from; j > to; j--) M = DL.matmul(M, mul(j));
      return M;
    };
    const Dj = j => { const A = DL.zeros2(dh, dh); for (let i = 0; i < dh; i++) A[i][i] = 1 - st.H[j][i] * st.H[j][i]; return A; };
    const plain = d3.range(1, D + 1).map(k => DL.specNorm(back(D, D - k, j => DL.matmul(Dj(j), WT)), 200, 3));
    /* leaky: αI + (1−α) D Wᵀ */
    const leaky = d3.range(1, D + 1).map(k => DL.specNorm(back(D, D - k, j => {
      const A = DL.matmul(Dj(j), WT);
      const R = DL.zeros2(dh, dh);
      for (let a = 0; a < dh; a++) for (let b = 0; b < dh; b++) R[a][b] = (a === b ? alpha : 0) + (1 - alpha) * A[a][b];
      return R;
    }), 200, 3));
    /* skip at delay m: the Jacobian gains a direct identity every m steps */
    const skip = d3.range(1, D + 1).map(k => {
      let M = DL.eye(dh);
      for (let j = D; j > D - k; j--) {
        const A = DL.matmul(Dj(j), WT);
        const R = DL.zeros2(dh, dh);
        const addI = ((D - j) % m === 0);
        for (let a = 0; a < dh; a++) for (let b = 0; b < dh; b++) R[a][b] = A[a][b] * 0.7 + (addI && a === b ? 0.7 : 0);
        M = DL.matmul(M, R);
      }
      return DL.specNorm(M, 200, 3);
    });
    /* gated: a real LSTM, measured by perturbation */
    const lnet = RN.net("lstm", dx, dh, 1, { seed: 31, wScale: 0.3, uScale: 0.3, forgetBias: 2 });
    const gated = d3.range(1, D + 1).map(k => DL.specNorm(DL.stateJac(lnet, X, T - k, 1e-5), 160, 5));

    const series = [
      { lab: "plain tanh, ρ(W) = " + DL.fmt(rho, 2), v: plain, c: DC.bad, np: dx * dh + dh * dh + dh },
      { lab: "leaky, α = " + DL.fmt(alpha, 2), v: leaky, c: DC.a2, np: dx * dh + dh * dh + dh },
      { lab: "skip through time, m = " + m, v: skip, c: DC.violet, np: dx * dh + 2 * dh * dh + dh },
      { lab: "gated (LSTM, b_f = 2)", v: gated, c: DC.good, np: DL.cellParams("lstm", dx, dh, 1).total }
    ];
    const allv = series.reduce((a, s) => a.concat(s.v), []).filter(v => v > 0 && isFinite(v));
    const x = d3.scaleLinear().domain([1, D]).range([0, f.iw]);
    const y = d3.scaleLog().domain([Math.max(1e-22, Math.min(...allv) / 4), Math.max(...allv) * 4]).range([f.ih, 0]).clamp(true);
    DL.gridY(g, y, f.iw, 5);
    DL.axisB(g, x, f.ih, 6, "distance d");
    DL.axisL(g, y, 5, "‖∂hₜ/∂hₜ₋ᵈ‖₂", v => DL.fmtE(v, 0));
    g.append("rect").attr("x", 0).attr("y", y(3.4e38)).attr("width", f.iw)
      .attr("height", Math.max(0, y(1.18e-38) - y(3.4e38))).attr("fill", DC.good).attr("fill-opacity", 0.05);
    /* the geometric predictions, each INCLUDING the measured tanh gate where one
       applies — a reference at ρ alone would sit far above the measurement. */
    const gate = d3.mean(st.H.map(h => d3.mean(h.map(v => 1 - v * v))));
    const preds = [rho * gate, alpha + (1 - alpha) * rho * gate, null];
    [[plain, preds[0], DC.bad], [leaky, preds[1], DC.a2]].forEach(d => {
      DL.curve(g, d3.range(1, D + 1).map(k => [x(k), y(Math.max(1e-300, d[0][0] * Math.pow(d[1], k - 1)))]),
        { stroke: d[2], w: 1, dash: "3 3", op: 0.55 });
    });
    /* the skip construction mixes a damped branch with a periodic identity and
       has no clean closed-form per-step factor, so NO reference is drawn for it
       and the readout says why rather than printing a number that misses. */
    series.forEach(s => DL.curve(g, s.v.map((v, i) => [x(i + 1), y(Math.max(v, 1e-300))]), { stroke: s.c, w: 2 }));
    DL.legend(g, series.map(s => ({ label: s.lab, color: s.c })), 4, f.ih + 30,
      { vertical: false, step: 178, font: 9 });

    const sx = f.iw + 16;
    RN.title(g, sx, 0, "per-step factor · reach · size");
    series.forEach((s, i) => {
      const yy = 18 + i * 44;
      const rate = RN.logSlope(d3.range(1, D + 1), s.v);
      let reach = null;
      for (let k = 0; k < s.v.length; k++) if (s.v[k] < s.v[0] * 1e-6) { reach = k + 1; break; }
      RN.note(g, sx, yy, s.lab, s.c, 9.5);
      RN.note(g, sx, yy + 12, "measured " + DL.fmt(rate, 4) +
        ((i < 3 && preds[i] !== null) ? "  ·  predicted " + DL.fmt(preds[i], 4) : "  ·  no closed form") +
        "  ·  reach " + (reach === null ? "> " + D : reach), DC.ink, 9.5);
      RN.note(g, sx, yy + 24, "cell parameters " + DL.commas(s.np), DC.muted, 9);
    });
    RN.note(g, sx, 18 + 4 * 44, "mean tanh gate on this run: " + DL.fmt(gate, 4), DC.a2, 9.5);
    RN.note(g, sx, 18 + 4 * 44 + 12, "the plain reference is ρ(W) × that gate, not", DC.muted, 9);
    RN.note(g, sx, 18 + 4 * 44 + 22, "ρ(W) alone; it is approximate, because the", DC.muted, 9);
    RN.note(g, sx, 18 + 4 * 44 + 32, "gate varies over the run.", DC.muted, 9);

    const rates = series.map(s => RN.logSlope(d3.range(1, D + 1), s.v));
    document.getElementById("rm-readout").innerHTML =
      `mean tanh gate on this run <b>${DL.fmt(gate, 4)}</b>, so the plain cell's prediction is ρ(W)·gate = ` +
      `<b>${DL.fmt(preds[0], 4)}</b> · measured per-step decay factors at d up to ${D}: plain <b>${DL.fmt(rates[0], 4)}</b> ` +
      `(predicted ${DL.fmt(preds[0], 4)}), leaky <b>${DL.fmt(rates[1], 4)}</b> (predicted ${DL.fmt(preds[1], 4)}), ` +
      `skip <b>${DL.fmt(rates[2], 4)}</b> (no closed form: the skip branch mixes a damped path with a periodic identity, ` +
      `so no reference line is drawn for it — but it is <b>${DL.fmt(rates[2] / Math.max(1e-9, rates[0]), 3)}×</b> the plain cell's ` +
      `per-step factor, which is the comparison that matters), gated <b>${DL.fmt(rates[3], 4)}</b> · ` +
      `the gated cell costs <b>${DL.commas(series[3].np)}</b> parameters against <b>${DL.commas(series[0].np)}</b> for the plain one, ` +
      `a factor of <b>${DL.fmt(series[3].np / series[0].np, 2)}×</b>.`;
  }
  [De, Re, Ae, Me].forEach(e => e.addEventListener("input", draw));
  draw();
})();

/* ═══════════ 23 · #lk-svg — leaky units, delays and reservoirs ═══════════ */
(function () {
  const svg = d3.select("#lk-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Ae = document.getElementById("lk-a"), Al = document.getElementById("lk-al"),
    Me = document.getElementById("lk-m"), Re = document.getElementById("lk-rho"),
    Te = document.getElementById("lk-T");

  /* one forward pass of the chosen architecture; returns the H trace */
  function run(kind, X, dh, rho, alpha, m, seed) {
    const T = X.length, dx = X[0].length;
    const Wm = RN.mat(dh, seed || 9, kind === "esn" ? 3 : rho);
    const U = RN.mat(dh, 77).slice(0, dx).map(r => r.slice(0, dh)).map(r => r.map(v => v * 0.7));
    const H = [], hist = [];
    let h = DL.zeros(dh);
    const alphas = [];
    for (let i = 0; i < dh; i++) alphas.push(kind === "mixed" ? [0.5, 0.8, 0.95, 0.99][i % 4] : alpha);
    for (let t = 0; t < T; t++) {
      hist.push(h.slice());
      let src = h;
      if (kind === "skip" && t >= m) {
        src = h.map((v, i) => 0.7 * v + 0.7 * hist[t - m][i]);
      }
      const z = DL.zeros(dh);
      for (let i = 0; i < dh; i++) { const hi = src[i]; if (hi) for (let j = 0; j < dh; j++) z[j] += hi * Wm[i][j]; }
      for (let i = 0; i < dx; i++) { const xi = X[t][i]; if (xi) for (let j = 0; j < dh; j++) z[j] += xi * U[i][j]; }
      const cand = z.map(Math.tanh);
      if (kind === "leaky" || kind === "mixed") h = h.map((v, i) => alphas[i] * v + (1 - alphas[i]) * cand[i]);
      else h = cand;
      H.push(h.slice());
    }
    return { H, Wm, U, alphas };
  }
  /* memory capacity: Σₖ R²( x_{t−k} recovered linearly from h_t ) */
  function capacity(kind, dh, rho, alpha, m, maxLag) {
    const T = 260, n = 200, r = DL.rng(88);
    const Xs = [], Hs = [];
    for (let s = 0; s < n; s++) {
      const X = DL.zeros2(T, 1).map(() => [DL.randn(r)]);
      const R = run(kind, X, dh, rho, alpha, m, 9);
      Xs.push(X); Hs.push(R.H[T - 1].concat([1]));
    }
    let mc = 0;
    const per = [];
    for (let k = 0; k < maxLag; k++) {
      const Y = Xs.map(X => [X[T - 1 - k][0]]);
      const B = DL.ridge(Hs, Y, 1e-5);
      const P = DL.matmul(Hs, B);
      let sse = 0, sst = 0;
      for (let i = 0; i < n; i++) { sse += (P[i][0] - Y[i][0]) ** 2; sst += Y[i][0] ** 2; }
      const r2 = Math.max(0, 1 - sse / Math.max(1e-12, sst));
      per.push(r2); mc += r2;
    }
    return { mc, per };
  }

  function draw() {
    const kind = Ae.value, alpha = +Al.value, m = +Me.value, rho = +Re.value, T = +Te.value;
    document.getElementById("lk-alv").textContent = DL.fmt(alpha, 3);
    document.getElementById("lk-mv").textContent = m;
    document.getElementById("lk-rhov").textContent = DL.fmt(rho, 2);
    document.getElementById("lk-Tv").textContent = T;
    const f = DL.frame(svg, W, H, { l: 52, r: 190, t: 24, b: 34 }), g = f.g;

    const dh = 10, dx = 2;
    const X = RN.seq(T, dx, 601, "pulse");
    const R = run(kind, X, dh, rho, alpha, m, 9);
    const norms = R.H.map(h => RN.norm(h));

    const gh = 130;
    RN.title(g, 0, -8, "impulse response — one pulse at step 1, then silence");
    const x = d3.scaleLinear().domain([1, T]).range([0, f.iw]);
    const pos = norms.filter(v => v > 0);
    const y = d3.scaleLog().domain([Math.max(1e-14, Math.min(...pos) / 3), Math.max(...pos) * 3]).range([gh, 0]).clamp(true);
    DL.gridY(g, y, f.iw, 4);
    DL.axisB(g, x, gh, 6, "");
    DL.axisL(g, y, 4, "‖hₜ‖", v => DL.fmtE(v, 0));
    DL.curve(g, norms.map((v, i) => [x(i + 1), y(Math.max(v, 1e-300))]), { stroke: DC.accent, w: 2 });
    const gateMean = d3.mean(R.H.map(h => d3.mean(h.map(v => 1 - v * v))));
    const predRate = kind === "leaky" ? alpha : kind === "mixed" ? 0.99
      : kind === "skip" ? Math.pow(rho * gateMean, 1 / m)
        : (kind === "esn" ? 1 : rho * gateMean);
    DL.curve(g, norms.map((v, i) => [x(i + 1), y(Math.max(1e-300, norms[0] * Math.pow(predRate, i)))]),
      { stroke: DC.a2, w: 1.2, dash: "4 3" });
    if (kind === "mixed") {
      [0.5, 0.8, 0.95, 0.99].forEach((a, gi) => {
        const sub = R.H.map(h => Math.abs(h[gi]));
        DL.curve(g, sub.map((v, i) => [x(i + 1), y(Math.max(v, 1e-300))]),
          { stroke: d3.interpolateTurbo(gi / 3), w: 1.1, op: 0.85 });
      });
    }
    DL.legend(g, [{ label: "measured ‖hₜ‖", color: DC.accent }, { label: "the architecture's predicted rate", color: DC.a2, dash: "4 3" }],
      4, gh + 26, { vertical: false, step: 178, font: 9.5 });

    /* backward-pass counterpart */
    const by = gh + 76, bh = 120;
    RN.title(g, 0, by - 10, "the backward counterpart — ‖∂hₜ/∂hₜ₋ᵈ‖₂ against d");
    const gg = g.append("g").attr("transform", `translate(0,${by})`);
    const Dmax = Math.min(T - 1, 70);
    const WT = DL.transpose(R.Wm);
    const Dj = j => { const A = DL.zeros2(dh, dh); for (let i = 0; i < dh; i++) A[i][i] = 1 - R.H[j][i] * R.H[j][i]; return A; };
    const step = j => {
      const A = DL.matmul(Dj(j), WT);
      const out = DL.zeros2(dh, dh);
      for (let a = 0; a < dh; a++) for (let b = 0; b < dh; b++) {
        if (kind === "leaky" || kind === "mixed") out[a][b] = (a === b ? R.alphas[a] : 0) + (1 - R.alphas[a]) * A[a][b];
        else if (kind === "skip") out[a][b] = 0.7 * A[a][b] + (((T - 1 - j) % m === 0) && a === b ? 0.7 : 0);
        else out[a][b] = A[a][b];
      }
      return out;
    };
    const jac = [];
    { let M = DL.eye(dh);
      for (let k = 1; k <= Dmax; k++) { M = DL.matmul(M, step(T - k)); jac.push(DL.specNorm(M, 180, 3)); } }
    const xb = d3.scaleLinear().domain([1, Dmax]).range([0, f.iw]);
    const posb = jac.filter(v => v > 0);
    const yb = d3.scaleLog().domain([Math.max(1e-18, Math.min(...posb) / 3), Math.max(...posb) * 3]).range([bh, 0]).clamp(true);
    DL.gridY(gg, yb, f.iw, 4);
    DL.axisB(gg, xb, bh, 6, "distance d");
    DL.axisL(gg, yb, 4, "‖J‖₂", v => DL.fmtE(v, 0));
    DL.curve(gg, jac.map((v, i) => [xb(i + 1), yb(Math.max(v, 1e-300))]), { stroke: DC.violet, w: 2 });
    let reach = null;
    for (let k = 0; k < jac.length; k++) if (jac[k] < jac[0] * 1e-6) { reach = k + 1; break; }
    if (reach) {
      gg.append("line").attr("x1", xb(reach)).attr("x2", xb(reach)).attr("y1", 0).attr("y2", bh)
        .attr("stroke", DC.bad).attr("stroke-dasharray", "3 3");
      RN.note(gg, xb(reach) + 4, 12, "falls to 10⁻⁶ at d = " + reach, DC.bad, 9);
    }

    const cap = capacity(kind === "esn" ? "plain" : kind, dh, kind === "esn" ? 3 : rho, alpha, m, 30);
    const trained = kind === "esn" ? dh + 1 : (dx * dh + dh * dh + dh);
    const sx = f.iw + 16;
    const kv = DL.kv(g, sx, 4, { keyW: 122, size: 10.5, lead: 16 });
    kv("measured decay rate", DL.fmt(RN.logSlope(d3.range(1, Dmax + 1), jac), 4), DC.violet);
    kv("architecture predicts", DL.fmt(predRate, 4), DC.a2);
    kv("mean tanh gate", DL.fmt(gateMean, 4), DC.muted);
    kv("memory horizon", reach === null ? "> " + Dmax : String(reach) + " steps");
    kv("memory capacity", DL.fmt(cap.mc, 3) + " / " + dh, DC.good);
    kv("trained parameters", DL.commas(trained), kind === "esn" ? DC.good : DC.ink);
    RN.note(g, sx, 108, "memory capacity is Σₖ R² of xₜ₋ₖ", DC.muted, 9);
    RN.note(g, sx, 119, "read linearly out of hₜ; it is", DC.muted, 9);
    RN.note(g, sx, 130, "bounded above by dₕ = " + dh + ".", DC.muted, 9);
    /* per-lag capacity profile */
    const cy = 160, cw = (f.iw + 174 - f.iw - 16) > 0 ? 5 : 5;
    RN.title(g, sx, cy - 8, "R² against lag");
    const cxs = d3.scaleLinear().domain([0, cap.per.length - 1]).range([0, 160]);
    const cys = d3.scaleLinear().domain([0, 1]).range([70, 0]);
    const cg = g.append("g").attr("transform", `translate(${sx},${cy})`);
    DL.axisB(cg, cxs, 70, 4, "lag");
    DL.axisL(cg, cys, 2, "");
    DL.curve(cg, cap.per.map((v, i) => [cxs(i), cys(v)]), { stroke: DC.good, w: 1.8, fill: DC.good, fillOp: 0.12 });

    document.getElementById("lk-readout").innerHTML =
      `<b>${Ae.options[Ae.selectedIndex].text}</b> · measured per-step decay factor <b>${DL.fmt(RN.logSlope(d3.range(1, Dmax + 1), jac), 4)}</b> ` +
      `against the architecture's prediction of <b>${DL.fmt(predRate, 4)}</b> ` +
      `(which includes the measured mean tanh gate of ${DL.fmt(gateMean, 3)}, not ρ(W) alone) · gradient reach ` +
      `<b>${reach === null ? "> " + Dmax : reach}</b> steps · memory capacity <b>${DL.fmt(cap.mc, 3)}</b> of a maximum ${dh} · ` +
      `parameters that have to be trained: <b>${DL.commas(trained)}</b>` +
      `${kind === "esn" ? " — the recurrence is fixed, so this is a convex least-squares fit." : "."}`;
  }
  [Al, Me, Re, Te].forEach(e => e.addEventListener("input", draw));
  Ae.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 24 · #lm-svg — the LSTM cell, with real values ═══════════ */
(function () {
  const svg = d3.select("#lm-svg");
  if (svg.empty()) return;
  const W = 760, H = 430;
  const Te = document.getElementById("lm-t"), Be = document.getElementById("lm-bf"),
    Ie = document.getElementById("lm-in"), Ve = document.getElementById("lm-v");

  function draw() {
    const bf = +Be.value, inp = Ie.value, view = Ve.value;
    const T = 40;
    Te.max = String(T);
    const t = Math.min(+Te.value, T);
    document.getElementById("lm-tv").textContent = t;
    document.getElementById("lm-bfv").textContent = DL.fmt(bf, 1);
    const f = DL.frame(svg, W, H, { l: 16, r: 16, t: 24, b: 16 }), g = f.g;

    const dx = 3, dh = 8;
    const p = DL.CELL.lstm.init(dx, dh, { seed: 31, wScale: 0.35, uScale: 0.5, forgetBias: bf });
    const X = RN.seq(T, dx, 701, inp);
    const st = DL.CELL.lstm.forward(p, X);
    const mean = a => d3.mean(a);

    if (view === "flow") {
      const yC = 74, yH = 230;
      /* the cell-state line, thickness = mean forget gate */
      const fm = mean(st.F[t - 1]);
      g.append("line").attr("x1", 30).attr("x2", 640).attr("y1", yC).attr("y2", yC)
        .attr("stroke", DC.teal).attr("stroke-width", 2 + 12 * fm).attr("stroke-opacity", 0.55);
      RN.note(g, 30, yC - 14 - 6 * fm, "cₜ₋₁", DC.teal, 11);
      RN.note(g, 600, yC - 14 - 6 * fm, "cₜ", DC.teal, 11);
      RN.note(g, 300, yC - 16 - 6 * fm, "the cell state — no matrix, no squashing, thickness ∝ mean fₜ = " + DL.fmt(fm, 3), DC.teal, 10);
      const node = (x, y, sym, col) => {
        g.append("circle").attr("cx", x).attr("cy", y).attr("r", 13).attr("fill", DC.panel2).attr("stroke", col).attr("stroke-width", 1.6);
        g.append("text").attr("x", x).attr("y", y + 5).attr("text-anchor", "middle").attr("font-size", 14).attr("fill", col).text(sym);
      };
      node(190, yC, "×", DC.a2); node(330, yC, "+", DC.good);
      const gate = (x, key, lab, col, sym) => {
        const v = mean(st[key][t - 1]);
        g.append("rect").attr("x", x - 40).attr("y", yH - 20).attr("width", 80).attr("height", 40).attr("rx", 6)
          .attr("fill", col).attr("fill-opacity", 0.10 + 0.5 * (key === "G" ? Math.abs(v) : v))
          .attr("stroke", col).attr("stroke-width", 1.5);
        g.append("text").attr("x", x).attr("y", yH - 3).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", DC.ink).text(lab);
        g.append("text").attr("x", x).attr("y", yH + 12).attr("text-anchor", "middle").attr("font-size", 11)
          .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col).text(DL.fmt(v, 3));
        g.append("text").attr("x", x).attr("y", yH - 28).attr("text-anchor", "middle").attr("font-size", 9.5)
          .attr("fill", DC.muted).text(sym);
        return v;
      };
      const vF = gate(120, "F", "forget  σ", DC.a2, "fₜ");
      const vI = gate(250, "I", "input  σ", DC.accent, "iₜ");
      const vG = gate(380, "G", "candidate  tanh", DC.violet, "gₜ");
      const vO = gate(510, "O", "output  σ", DC.good, "oₜ");
      DL.arrow(g, 120, yH - 22, 190, yC + 14, { color: DC.a2, w: 1.4, head: 5 });
      node(300, 150, "×", DC.accent);
      DL.arrow(g, 250, yH - 22, 296, 162, { color: DC.accent, w: 1.4, head: 5 });
      DL.arrow(g, 380, yH - 22, 306, 162, { color: DC.violet, w: 1.4, head: 5 });
      DL.arrow(g, 300, 138, 330, yC + 14, { color: DC.good, w: 1.4, head: 5 });
      /* the branch to h */
      g.append("line").attr("x1", 470).attr("x2", 470).attr("y1", yC).attr("y2", 150).attr("stroke", DC.teal).attr("stroke-width", 1.6);
      node(470, 150, "t", DC.teal);
      g.append("text").attr("x", 470).attr("y", 172).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", DC.muted).text("tanh");
      node(560, 150, "×", DC.good);
      DL.arrow(g, 483, 150, 546, 150, { color: DC.teal, w: 1.4, head: 5 });
      DL.arrow(g, 510, yH - 22, 566, 164, { color: DC.good, w: 1.4, head: 5 });
      DL.arrow(g, 573, 150, 640, 150, { color: DC.ink, w: 1.6, head: 6 });
      RN.note(g, 646, 154, "hₜ", DC.ink, 11);
      /* inputs to the gates */
      g.append("line").attr("x1", 60).attr("x2", 570).attr("y1", 320).attr("y2", 320).attr("stroke", DC.line);
      [120, 250, 380, 510].forEach(x => DL.arrow(g, x, 318, x, yH + 22, { color: DC.line, w: 1, head: 4 }));
      RN.note(g, 20, 324, "[hₜ₋₁, xₜ]", DC.muted, 10.5);
      const kv = DL.kv(g, 20, 356, { keyW: 176, size: 10.5, lead: 15 });
      kv("‖cₜ‖", DL.fmt(RN.norm(st.C[t - 1]), 4), DC.teal);
      kv("‖hₜ‖", DL.fmt(RN.norm(st.H[t - 1]), 4));
      kv("units with fₜ > 0.9", DL.fmt(100 * st.F[t - 1].filter(v => v > 0.9).length / dh, 0) + "%", DC.a2);
      const life = fm >= 0.999999 ? Infinity : Math.log(1e-3) / Math.log(Math.max(1e-9, fm));
      kv("steps to lose 99.9% of cₜ", isFinite(life) ? DL.fmt(life, 1) : "∞", DC.good);
      document.getElementById("lm-readout").innerHTML =
        `step ${t}, b_f = ${DL.fmt(bf, 1)} · mean gates: forget <b>${DL.fmt(vF, 3)}</b>, input <b>${DL.fmt(vI, 3)}</b>, ` +
        `candidate <b>${DL.fmt(vG, 3)}</b>, output <b>${DL.fmt(vO, 3)}</b> · ‖cₜ‖ = <b>${DL.fmt(RN.norm(st.C[t - 1]), 4)}</b>, ` +
        `‖hₜ‖ = <b>${DL.fmt(RN.norm(st.H[t - 1]), 4)}</b> · <b>${DL.fmt(100 * st.F[t - 1].filter(v => v > 0.9).length / dh, 0)}%</b> of units ` +
        `have a forget gate above 0.9 · at this mean forget gate the cell state loses 99.9% of its magnitude in ` +
        `<b>${isFinite(life) ? DL.fmt(life, 1) : "∞"}</b> steps.`;
    } else {
      const rows = [["F", "forget gate fₜ", DC.a2], ["I", "input gate iₜ", DC.accent],
      ["G", "candidate gₜ", DC.violet], ["O", "output gate oₜ", DC.good], ["C", "cell state cₜ", DC.teal]];
      const cw = Math.min(16, f.iw / T), ch = 12;
      rows.forEach((r, ri) => {
        const oy = 8 + ri * (dh * ch + 22);
        RN.title(g, 0, oy - 4, r[1], r[2]);
        const M = st[r[0]];
        const mm = Math.max(...M.map(v => Math.max(...v.map(Math.abs))), 1e-9);
        for (let tt = 0; tt < T; tt++) for (let i = 0; i < dh; i++) {
          g.append("rect").attr("x", 120 + tt * cw).attr("y", oy + i * ch).attr("width", cw + 0.4).attr("height", ch + 0.4)
            .attr("shape-rendering", "crispEdges")
            .attr("fill", r[0] === "G" || r[0] === "C" ? RN.signColor(M[tt][i], mm) : RN.heat(M[tt][i]));
        }
        g.append("rect").attr("x", 120).attr("y", oy).attr("width", T * cw).attr("height", dh * ch)
          .attr("fill", "none").attr("stroke", DC.line);
        g.append("line").attr("x1", 120 + (t - 1) * cw + cw / 2).attr("x2", 120 + (t - 1) * cw + cw / 2)
          .attr("y1", oy).attr("y2", oy + dh * ch).attr("stroke", DC.ink).attr("stroke-opacity", 0.7);
      });
      const latch = st.F[T - 1].map((_, i) => d3.mean(st.F.map(fv => fv[i]))).filter(v => v > 0.9).length;
      document.getElementById("lm-readout").innerHTML =
        `the whole ${T}-step trajectory, b_f = ${DL.fmt(bf, 1)} · <b>${latch}</b> of ${dh} units have a mean forget gate ` +
        `above 0.9 across the run, and those are the ones that appear as horizontal stripes in the cell-state map · ` +
        `mean forget gate over everything: <b>${DL.fmt(d3.mean(st.F.map(v => d3.mean(v))), 4)}</b>, ` +
        `mean input gate <b>${DL.fmt(d3.mean(st.I.map(v => d3.mean(v))), 4)}</b>, ` +
        `largest |cₜ| reached <b>${DL.fmt(Math.max(...st.C.map(c => Math.max(...c.map(Math.abs)))), 3)}</b>.`;
    }
  }
  [Te, Be].forEach(e => e.addEventListener("input", draw));
  [Ie, Ve].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 25 · #gt-svg — ablate one gate and see what breaks ═══════════ */
(function () {
  const svg = d3.select("#gt-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Ae = document.getElementById("gt-a"), Te = document.getElementById("gt-t"),
    Le = document.getElementById("gt-T"), Ne = document.getElementById("gt-n");

  /* an ablation is imposed by PINNING parameters, not by forking the cell:
     zero the gate's matrices and set its bias, then re-impose that after
     every optimiser step. σ(9) = 0.99988, σ(0) = 0.5.                      */
  const PIN = {
    none: null,
    f: { gate: "f", bias: 9 }, f0: { gate: "f", bias: 0 },
    i: { gate: "i", bias: 9 }, o: { gate: "o", bias: 9 },
    nogr: { recurrentOnly: true }
  };
  const NAMES = { none: "full cell", f: "forget pinned 1", f0: "forget pinned ½",
    i: "input pinned 1", o: "output pinned 1", nogr: "gates blind to h" };
  const TN = { hold: "hold", reset: "hold + clear", ignore: "ignore distractors" };
  function impose(p, ab) {
    const pin = PIN[ab];
    if (!pin) return;
    if (pin.recurrentOnly) {
      DL.CELL.lstm.gates.forEach(k => { for (let a = 0; a < p.dh; a++) for (let b = 0; b < p.dh; b++) p.W[k][a][b] = 0; });
      return;
    }
    const k = pin.gate;
    for (let a = 0; a < p.dx; a++) for (let b = 0; b < p.dh; b++) p.U[k][a][b] = 0;
    for (let a = 0; a < p.dh; a++) for (let b = 0; b < p.dh; b++) p.W[k][a][b] = 0;
    for (let b = 0; b < p.dh; b++) p.b[k][b] = pin.bias;
  }
  const CACHE = {}, SEEDS = [4242, 909, 1717];
  function fitOne(ab, task, T, steps, seed) {
    const K = 3, dx = K + 1, dh = 10;
    const net = RN.net("lstm", dx, dh, K, { seed: seed, out: "softmax", wScale: 0.3, uScale: 0.3, forgetBias: 1 });
    impose(net.p, ab);
    const tr = RN.trainer(net, 0.03, 1.0), r = DL.rng(seed + 11), hist = [];
    for (let s = 0; s < steps; s++) {
      const out = tr.step(RN.memBatch(task, T, K, 6, r));
      impose(net.p, ab);
      if (s % 10 === 0) hist.push([s, out.loss]);
    }
    const B = RN.memBatch(task, T, K, 90, DL.rng(seed + 999));
    let acc = 0;
    for (const b of B) {
      const a = DL.seqForward(net, b.X).A[T - 1];
      let am = 0; for (let j = 1; j < K; j++) if (a[j] > a[am]) am = j;
      if (am === b.k) acc++;
    }
    const st = DL.CELL.lstm.forward(net.p, RN.memBatch(task, T, K, 1, DL.rng(7))[0].X);
    return { err: 1 - acc / B.length, hist: hist,
      gates: ["F", "I", "G", "O"].map(k2 => d3.mean(st[k2].map(v => d3.mean(v)))) };
  }
  /* a single short fit is far too noisy to compare cells with, so average three */
  function fit(ab, task, T, steps) {
    const key = ab + "|" + task + "|" + T + "|" + steps;
    if (CACHE[key]) return CACHE[key];
    const runs = SEEDS.map(sd => fitOne(ab, task, T, steps, sd));
    const res = { err: d3.mean(runs.map(r => r.err)), hist: runs[0].hist,
      gates: [0, 1, 2, 3].map(i => d3.mean(runs.map(r => r.gates[i]))),
      spread: d3.deviation(runs.map(r => r.err)) || 0, n: SEEDS.length };
    CACHE[key] = res;
    return res;
  }
  /* the ablation × task grid is computed ONCE at a fixed reference setting */
  const REF_T = 16, REF_S = 260;
  let GRID = null;
  function grid() {
    if (GRID) return GRID;
    GRID = {};
    ["none", "f", "f0", "i", "o", "nogr"].forEach(a =>
      ["hold", "reset", "ignore"].forEach(t => { GRID[a + "|" + t] = fit(a, t, REF_T, REF_S).err; }));
    return GRID;
  }

  function draw() {
    const ab = Ae.value, task = Te.value, T = +Le.value, steps = +Ne.value;
    document.getElementById("gt-Tv").textContent = T;
    document.getElementById("gt-nv").textContent = steps;
    const f = DL.frame(svg, W, H, { l: 50, r: 330, t: 24, b: 40 }), g = f.g;

    const base = fit("none", task, T, steps), abl = fit(ab, task, T, steps);
    const all = base.hist.concat(abl.hist).map(p => p[1]).filter(v => v > 0);
    const x = d3.scaleLinear().domain([0, steps]).range([0, f.iw]);
    const y = d3.scaleLog().domain([Math.max(1e-4, Math.min(...all) / 1.6), Math.max(...all) * 1.4]).range([f.ih, 0]).clamp(true);
    DL.gridY(g, y, f.iw, 4);
    DL.axisB(g, x, f.ih, 5, "optimisation step");
    DL.axisL(g, y, 4, "loss", v => DL.fmtE(v, 0));
    DL.curve(g, base.hist.map(p => [x(p[0]), y(Math.max(p[1], 1e-6))]), { stroke: DC.good, w: 2 });
    if (ab !== "none") DL.curve(g, abl.hist.map(p => [x(p[0]), y(Math.max(p[1], 1e-6))]), { stroke: DC.bad, w: 2 });
    DL.legend(g, [{ label: "the full LSTM", color: DC.good }].concat(ab !== "none" ? [{ label: "with the ablation", color: DC.bad }] : []),
      4, f.ih + 30, { vertical: false, step: 130, font: 9.5 });

    const G = grid();
    const sx = f.iw + 18;
    RN.title(g, sx, -4, "held-out error, every ablation × every task");
    RN.note(g, sx, 8, "fixed reference: T = " + REF_T + ", " + REF_S + " steps, mean of " + SEEDS.length + " seeds", DC.muted, 9);
    const abls = ["none", "f", "f0", "i", "o", "nogr"], tasks = ["hold", "reset", "ignore"];
    /* NOTE the tasks are genuinely different — see RN.memBatch */
    const cw = 66, ch = 25, top = 34;
    tasks.forEach((tk, j) => RN.note(g, sx + 130 + j * cw + 6, top - 4, TN[tk].split(" ")[0], DC.muted, 9));
    abls.forEach((a, i) => {
      RN.note(g, sx, top + i * ch + 14, NAMES[a], a === ab ? DC.ink : DC.muted, 9.5);
      tasks.forEach((tk, j) => {
        const e = G[a + "|" + tk];
        g.append("rect").attr("x", sx + 130 + j * cw).attr("y", top + i * ch + 2).attr("width", cw - 3).attr("height", ch - 5)
          .attr("rx", 2).attr("fill", d3.interpolateRdYlGn(1 - Math.min(1, e / 0.7))).attr("fill-opacity", 0.8)
          .attr("stroke", (a === ab && tk === task) ? DC.ink : DC.line).attr("stroke-width", (a === ab && tk === task) ? 2 : 0.6);
        g.append("text").attr("x", sx + 130 + j * cw + (cw - 3) / 2).attr("y", top + i * ch + 16).attr("text-anchor", "middle")
          .attr("font-size", 9).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", "#08111f")
          .text(DL.fmt(100 * e, 0) + "%");
      });
    });
    const gy = top + abls.length * ch + 26;
    RN.title(g, sx, gy - 6, "mean gate values the FULL cell chose here");
    ["forget", "input", "candidate", "output"].forEach((nm, i) => {
      const v = base.gates[i];
      g.append("rect").attr("x", sx + 88).attr("y", gy + i * 16).attr("width", 130 * Math.abs(v)).attr("height", 10)
        .attr("fill", [DC.a2, DC.accent, DC.violet, DC.good][i]).attr("fill-opacity", 0.7);
      RN.note(g, sx, gy + i * 16 + 9, nm, DC.muted, 9.5);
      g.append("text").attr("x", sx + 226).attr("y", gy + i * 16 + 9).attr("font-size", 9)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(DL.fmt(v, 3));
    });

    const what = {
      none: "nothing removed",
      f: "the cell can no longer CLEAR its memory",
      f0: "the memory decays at a fixed rate the model cannot change",
      i: "every step's candidate is written, so distractors overwrite the memory",
      o: "the cell cannot hold something without also exposing it",
      nogr: "the gates decide without knowing what the cell is currently doing"
    }[ab];
    /* which ablation is worst on each task, read off the grid rather than asserted */
    const worstOn = tk => abls.filter(a => a !== "none").reduce((x, a) => (G[a + "|" + tk] > G[x + "|" + tk] ? a : x), "f");
    const fOK = G["f|hold"] <= G["none|hold"] + 0.05;
    document.getElementById("gt-readout").innerHTML =
      `task <b>${TN[task]}</b>, T = ${T}, ${steps} steps, mean of ${base.n} seeds · full cell error <b>${DL.fmt(100 * base.err, 1)}%</b>` +
      ` (seed spread ±${DL.fmt(100 * base.spread, 1)} points)` +
      (ab === "none" ? " — no ablation selected"
        : `, with <b>${NAMES[ab]}</b> <b>${DL.fmt(100 * abl.err, 1)}%</b> — a ratio of <b>${DL.fmt(abl.err / Math.max(0.005, base.err), 2)}×</b>`) +
      ` · what the ablation removes: ${what} · chance is 66.7% · on the reference grid the worst ablation is ` +
      `<b>${NAMES[worstOn("hold")]}</b> for hold, <b>${NAMES[worstOn("reset")]}</b> for hold-and-clear and ` +
      `<b>${NAMES[worstOn("ignore")]}</b> for ignore` +
      (fOK ? ` · <b>honest negative:</b> pinning the forget gate fully OPEN does not hurt at T = ${REF_T}, because none of these tasks requires clearing within the window.` : ".");
  }
  [Le, Ne].forEach(e => e.addEventListener("input", draw));
  [Ae, Te].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 26 · #cs-svg — gradient at distance ═══════════ */
(function () {
  const svg = d3.select("#cs-svg");
  if (svg.empty()) return;
  const W = 760, H = 410;
  const De = document.getElementById("cs-D"), He = document.getElementById("cs-dh"),
    Be = document.getElementById("cs-bf"), Ze = document.getElementById("cs-bz"),
    Re = document.getElementById("cs-rho"), Pe = document.getElementById("cs-p");

  function draw() {
    const D = +De.value, dh = +He.value, bf = +Be.value, bz = +Ze.value, rho = +Re.value, path = Pe.value;
    document.getElementById("cs-Dv").textContent = D;
    document.getElementById("cs-dhv").textContent = dh;
    document.getElementById("cs-bfv").textContent = DL.fmt(bf, 1);
    document.getElementById("cs-bzv").textContent = DL.fmt(bz, 1);
    document.getElementById("cs-rhov").textContent = DL.fmt(rho, 2);
    const f = DL.frame(svg, W, H, { l: 56, r: 208, t: 24, b: 40 }), g = f.g;

    const dx = 3, T = D + 1;
    const X = RN.seq(T, dx, 91);
    const van = RN.net("vanilla", dx, dh, 2, { seed: 31, rho: rho, wScale: 0.3, uScale: 0.3 });
    const gru = RN.net("gru", dx, dh, 2, { seed: 31, wScale: 0.3, uScale: 0.3 });
    gru.p.b.z = gru.p.b.z.map(() => bz);
    const lstm = RN.net("lstm", dx, dh, 2, { seed: 31, wScale: 0.3, uScale: 0.3, forgetBias: bf });

    const ds = d3.range(1, D + 1).filter(k => k <= 6 || k % Math.max(1, Math.round(D / 24)) === 0 || k === D);
    const meas = (net, k) => DL.specNorm(DL.stateJac(net, X, T - k, 1e-5), 150, 5);
    const series = [
      { lab: "vanilla, ρ(W) = " + DL.fmt(rho, 2), v: ds.map(k => meas(van, k)), c: RN.CELLC.vanilla },
      { lab: "GRU, b_z = " + DL.fmt(bz, 1), v: ds.map(k => meas(gru, k)), c: RN.CELLC.gru },
      { lab: "LSTM, b_f = " + DL.fmt(bf, 1), v: ds.map(k => meas(lstm, k)), c: RN.CELLC.lstm }
    ];
    /* the cell-state path, when asked for */
    let cellPath = null, prodF = null;
    if (path === "c") {
      const st = DL.CELL.lstm.forward(lstm.p, X);
      cellPath = ds.map(k => {
        const kk = T - k;
        const hk = kk === 0 ? DL.zeros(dh) : st.H[kk - 1], ck = kk === 0 ? DL.zeros(dh) : st.C[kk - 1];
        const tail = X.slice(kk);
        const J = DL.jacFD(cv => { const s = DL.CELL.lstm.forward(lstm.p, tail, hk, cv); return s.C[s.T - 1]; }, ck, 1e-5);
        return DL.specNorm(J, 150, 5);
      });
      prodF = ds.map(k => {
        const kk = T - k;
        let mx = 0;
        for (let i = 0; i < dh; i++) { let pr = 1; for (let tt = kk; tt < T; tt++) pr *= st.F[tt][i]; mx = Math.max(mx, pr); }
        return mx;
      });
    }

    const allv = series.reduce((a, s) => a.concat(s.v), []).concat(cellPath || []).concat(prodF || [])
      .filter(v => v > 0 && isFinite(v));
    const x = d3.scaleLinear().domain([1, D]).range([0, f.iw]);
    const y = d3.scaleLog().domain([Math.max(1e-24, Math.min(...allv) / 4), Math.max(...allv) * 4]).range([f.ih, 0]).clamp(true);
    DL.gridY(g, y, f.iw, 5);
    DL.axisB(g, x, f.ih, 6, "distance d = T − k");
    DL.axisL(g, y, 5, path === "c" ? "‖∂c_T/∂c_k‖₂" : "‖∂h_T/∂h_k‖₂", v => DL.fmtE(v, 0));
    g.append("rect").attr("x", 0).attr("y", y(3.4e38)).attr("width", f.iw)
      .attr("height", Math.max(0, y(1.18e-38) - y(3.4e38))).attr("fill", DC.good).attr("fill-opacity", 0.05);
    if (path === "c") {
      DL.curve(g, cellPath.map((v, i) => [x(ds[i]), y(Math.max(v, 1e-300))]), { stroke: RN.CELLC.lstm, w: 2.4 });
      DL.curve(g, prodF.map((v, i) => [x(ds[i]), y(Math.max(v, 1e-300))]), { stroke: RN.CELLC.lstm, w: 1.4, dash: "4 3" });
    } else {
      series.forEach(s => DL.curve(g, s.v.map((v, i) => [x(ds[i]), y(Math.max(v, 1e-300))]), { stroke: s.c, w: 2 }));
    }
    DL.legend(g, path === "c"
      ? [{ label: "measured ‖∂c_T/∂c_k‖₂", color: RN.CELLC.lstm }, { label: "the pure product of forget gates", color: RN.CELLC.lstm, dash: "4 3" }]
      : series.map(s => ({ label: s.lab, color: s.c })),
      4, f.ih + 30, { vertical: false, step: 200, font: 9.5 });

    const rates = series.map(s => RN.logSlope(ds, s.v));
    const sx = f.iw + 16, bw = 110;
    RN.title(g, sx, -4, "measured per-step factor");
    const bs = d3.scaleLinear().domain([0, Math.max(1.1, ...rates.filter(isFinite))]).range([0, bw]);
    series.forEach((s, i) => {
      const yy = 12 + i * 34;
      g.append("rect").attr("x", sx + 4).attr("y", yy).attr("width", Math.max(1, bs(rates[i] || 0))).attr("height", 12)
        .attr("fill", s.c).attr("fill-opacity", 0.7);
      g.append("text").attr("x", sx + 4 + bw + 8).attr("y", yy + 10).attr("font-size", 9.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(DL.fmt(rates[i], 4));
      RN.note(g, sx + 4, yy + 24, s.lab, s.c, 9);
    });
    g.append("line").attr("x1", sx + 4 + bs(1)).attr("x2", sx + 4 + bs(1)).attr("y1", 8).attr("y2", 12 + 3 * 34)
      .attr("stroke", DC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.7);
    const stL = DL.CELL.lstm.forward(lstm.p, X), stG = DL.CELL.gru.forward(gru.p, X);
    const kv = DL.kv(g, sx, 132, { keyW: 148, size: 10.5, lead: 15 });
    kv("mean forget gate fₜ", DL.fmt(d3.mean(stL.F.map(v => d3.mean(v))), 4), RN.CELLC.lstm);
    kv("mean GRU carry 1 − zₜ", DL.fmt(1 - d3.mean(stG.Z.map(v => d3.mean(v))), 4), RN.CELLC.gru);
    series.forEach((s, i) => {
      let reach = null;
      for (let k = 0; k < s.v.length; k++) if (s.v[k] < s.v[0] * 1e-6) { reach = ds[k]; break; }
      kv("reach, " + ["vanilla", "GRU", "LSTM"][i], reach === null ? "> " + D : String(reach), s.c);
    });
    const ratio = series[2].v[series[2].v.length - 1] / Math.max(1e-300, series[0].v[series[0].v.length - 1]);
    kv("LSTM / vanilla at d = " + D, DL.fmtE(ratio, 2), ratio > 1 ? DC.good : DC.bad);
    if (path === "c") {
      const gap = cellPath[cellPath.length - 1] / Math.max(1e-300, prodF[prodF.length - 1]);
      kv("measured / ∏f at d = " + D, DL.fmtE(gap, 2), DC.a2);
    }

    document.getElementById("cs-readout").innerHTML =
      `measured per-step factors: vanilla <b>${DL.fmt(rates[0], 4)}</b>, GRU <b>${DL.fmt(rates[1], 4)}</b>, ` +
      `LSTM <b>${DL.fmt(rates[2], 4)}</b> · at d = ${D} the LSTM passes <b>${DL.fmtE(series[2].v[series[2].v.length - 1], 2)}</b> ` +
      `against the vanilla cell's <b>${DL.fmtE(series[0].v[series[0].v.length - 1], 2)}</b>, a ratio of <b>${DL.fmtE(ratio, 2)}</b>` +
      `${ratio < 1 ? " — the LSTM is WORSE here, because its forget gate is not open" : ""} · ` +
      `mean forget gate <b>${DL.fmt(d3.mean(stL.F.map(v => d3.mean(v))), 4)}</b>` +
      (path === "c" ? ` · the pure product of forget gates understates the measured cell-state Jacobian by <b>${DL.fmtE(cellPath[cellPath.length - 1] / Math.max(1e-300, prodF[prodF.length - 1]), 2)}×</b>.` : ".");
  }
  [De, He, Be, Ze, Re].forEach(e => e.addEventListener("input", draw));
  Pe.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 27 · #fb-svg — sweeping the forget-gate bias ═══════════ */
(function () {
  const svg = d3.select("#fb-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Re = document.getElementById("fb-r"), De = document.getElementById("fb-d"),
    He = document.getElementById("fb-dh"), Fe = document.getElementById("fb-fit");

  const FIT = {};
  function fitAt(bf, T, dh) {
    const key = DL.fmt(bf, 2) + "|" + T + "|" + dh;
    if (FIT[key] !== undefined) return FIT[key];
    const K = 3, dx = K + 2;
    const net = RN.net("lstm", dx, dh, K, { seed: 909, out: "softmax", wScale: 0.35, uScale: 0.5, forgetBias: bf });
    const tr = RN.trainer(net, 0.04, 1.0), r = DL.rng(77);
    for (let s = 0; s < 120; s++) tr.step(RN.memBatch("reset", T, K, 6, r));
    const B = RN.memBatch("reset", T, K, 90, DL.rng(313));
    let acc = 0;
    for (const b of B) {
      const a = DL.seqForward(net, b.X).A[T - 1];
      let am = 0; for (let j = 1; j < K; j++) if (a[j] > a[am]) am = j;
      if (am === b.k) acc++;
    }
    FIT[key] = 1 - acc / B.length;
    return FIT[key];
  }

  function draw() {
    const wide = Re.value === "wide", d = +De.value, dh = +He.value, doFit = Fe.value === "on";
    document.getElementById("fb-dv").textContent = d;
    document.getElementById("fb-dhv").textContent = dh;
    const f = DL.frame(svg, W, H, { l: 52, r: 250, t: 24, b: 40 }), g = f.g;

    const bs = wide ? d3.range(-3, 6.01, 0.5) : d3.range(0, 3.01, 0.25);
    const dx = 3, T = d + 1, X = RN.seq(T, dx, 91);
    const rows = bs.map(bf => {
      const net = RN.net("lstm", dx, dh, 2, { seed: 31, wScale: 0.3, uScale: 0.3, forgetBias: bf });
      const st = DL.CELL.lstm.forward(net.p, X);
      return { bf: bf, jac: DL.specNorm(DL.stateJac(net, X, T - d, 1e-5), 150, 5),
        gate: d3.mean(st.F.map(v => d3.mean(v))),
        sat: d3.mean(st.F.map(v => v.filter(x => x > 0.9).length / dh)),
        F: st.F };
    });

    const x = d3.scaleLinear().domain([bs[0], bs[bs.length - 1]]).range([0, f.iw]);
    const jv = rows.map(r => r.jac).filter(v => v > 0);
    const y = d3.scaleLog().domain([Math.max(1e-22, Math.min(...jv) / 4), Math.max(...jv) * 4]).range([f.ih, 0]).clamp(true);
    const y2 = d3.scaleLinear().domain([0, 1]).range([f.ih, 0]);
    DL.gridY(g, y, f.iw, 5);
    DL.axisB(g, x, f.ih, 6, "forget-gate bias b_f");
    DL.axisL(g, y, 5, "‖∂h_T/∂h_k‖₂ at d = " + d, v => DL.fmtE(v, 0));
    g.append("g").attr("class", "axis").attr("transform", `translate(${f.iw},0)`)
      .call(d3.axisRight(y2).ticks(4));
    g.append("text").attr("x", f.iw + 4).attr("y", -6).attr("font-size", 10).attr("fill", DC.a2).text("mean fₜ");
    DL.curve(g, rows.map(r => [x(r.bf), y(Math.max(r.jac, 1e-300))]), { stroke: RN.CELLC.lstm, w: 2.4 });
    DL.curve(g, rows.map(r => [x(r.bf), y2(r.gate)]), { stroke: DC.a2, w: 1.8, dash: "4 3" });
    if (bs[0] <= 1 && bs[bs.length - 1] >= 1) {
      g.append("line").attr("x1", x(1)).attr("x2", x(1)).attr("y1", 0).attr("y2", f.ih)
        .attr("stroke", DC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.7);
      RN.note(g, x(1) + 4, 12, "the conventional b_f = 1", DC.ink, 9);
    }
    let fitRows = null, best = null;
    if (doFit) {
      const sub = bs.filter((b, i) => i % Math.max(1, Math.round(bs.length / 8)) === 0 || i === bs.length - 1);
      fitRows = sub.map(bf => ({ bf: bf, err: fitAt(bf, 24, 6) }));
      const y3 = d3.scaleLinear().domain([0, Math.max(0.7, ...fitRows.map(r => r.err))]).range([f.ih, 0]);
      DL.curve(g, fitRows.map(r => [x(r.bf), y3(r.err)]), { stroke: DC.bad, w: 1.8 });
      fitRows.forEach(r => g.append("circle").attr("cx", x(r.bf)).attr("cy", y3(r.err)).attr("r", 2.6).attr("fill", DC.bad));
      best = fitRows.reduce((a, b) => (b.err < a.err ? b : a));
      g.append("circle").attr("cx", x(best.bf)).attr("cy", y3(best.err)).attr("r", 5)
        .attr("fill", "none").attr("stroke", DC.bad).attr("stroke-width", 2);
      RN.note(g, x(best.bf) + 7, y3(best.err) - 5, "best b_f = " + DL.fmt(best.bf, 2), DC.bad, 9.5);
    }
    DL.legend(g, [{ label: "gradient at distance " + d, color: RN.CELLC.lstm },
    { label: "mean forget gate (right axis)", color: DC.a2, dash: "4 3" }]
      .concat(doFit ? [{ label: "held-out error, hold-and-clear task", color: DC.bad }] : []),
      4, f.ih + 30, { vertical: false, step: 168, font: 9 });

    /* the gate histogram at the selected bias */
    const sel = rows.reduce((a, r) => (Math.abs(r.bf - 1) < Math.abs(a.bf - 1) ? r : a));
    const sx = f.iw + 44;
    RN.title(g, sx, -4, "distribution of fₜ at b_f = " + DL.fmt(sel.bf, 1));
    const vals = [];
    sel.F.forEach(v => v.forEach(x2 => vals.push(x2)));
    const bins = d3.bin().domain([0, 1]).thresholds(20)(vals);
    const hx = d3.scaleLinear().domain([0, 1]).range([0, 190]);
    const hy = d3.scaleLinear().domain([0, d3.max(bins, b => b.length) || 1]).range([90, 0]);
    const hg = g.append("g").attr("transform", `translate(${sx},10)`);
    bins.forEach(b => hg.append("rect").attr("x", hx(b.x0)).attr("y", hy(b.length))
      .attr("width", Math.max(1, hx(b.x1) - hx(b.x0) - 1)).attr("height", 90 - hy(b.length))
      .attr("fill", DC.a2).attr("fill-opacity", 0.7));
    DL.axisB(hg, hx, 90, 3, "fₜ");
    const kv = DL.kv(g, sx, 140, { keyW: 156, size: 10.5, lead: 15 });
    kv("mean fₜ", DL.fmt(sel.gate, 4), DC.a2);
    kv("units with fₜ > 0.9", DL.fmt(100 * sel.sat, 1) + "%");
    kv("gradient at d = " + d, DL.fmtE(sel.jac, 3), RN.CELLC.lstm);
    kv("mean fₜ raised to d", DL.fmtE(Math.pow(sel.gate, d), 3), DC.muted);
    const r0 = rows.reduce((a, r) => (Math.abs(r.bf) < Math.abs(a.bf) ? r : a));
    kv("the same at b_f = 0", DL.fmtE(r0.jac, 3), DC.bad);
    kv("ratio", DL.fmtE(sel.jac / Math.max(1e-300, r0.jac), 2), DC.good);
    if (best) kv("bias minimising the task error", DL.fmt(best.bf, 2) + "  (" + DL.fmt(100 * best.err, 1) + "%)", DC.bad);

    document.getElementById("fb-readout").innerHTML =
      `at b_f = ${DL.fmt(sel.bf, 1)} the mean forget gate is <b>${DL.fmt(sel.gate, 4)}</b> and the gradient at distance ${d} is ` +
      `<b>${DL.fmtE(sel.jac, 3)}</b>; at b_f = ${DL.fmt(r0.bf, 1)} it is <b>${DL.fmtE(r0.jac, 3)}</b> — a factor of ` +
      `<b>${DL.fmtE(sel.jac / Math.max(1e-300, r0.jac), 2)}</b> · <b>${DL.fmt(100 * sel.sat, 1)}%</b> of gate values exceed 0.9` +
      (best ? ` · the bias that minimises the hold-and-clear error is <b>${DL.fmt(best.bf, 2)}</b>, not the largest one — ` +
        `a gate pinned open cannot clear.` : ` · turn the fitting option on to see the task error, which has an interior minimum.`);
  }
  [De, He].forEach(e => e.addEventListener("input", draw));
  [Re, Fe].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 28 · #pm-svg — parameter counts and matched widths ═══════════ */
(function () {
  const svg = d3.select("#pm-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Xe = document.getElementById("pm-dx"), He = document.getElementById("pm-dh"),
    Be = document.getElementById("pm-b"), Ce = document.getElementById("pm-c");

  function draw() {
    const dx = +Xe.value, dh = +He.value, bs = +Be.value, cmp = Ce.value;
    document.getElementById("pm-dxv").textContent = dx;
    document.getElementById("pm-dhv").textContent = dh;
    const f = DL.frame(svg, W, H, { l: 14, r: 14, t: 24, b: 20 }), g = f.g;

    const kinds = ["vanilla", "gru", "lstm"];
    const counts = kinds.map(k => DL.cellParams(k, dx, dh, bs));
    /* independent check: count the arrays the library actually allocated */
    const counted = kinds.map(k => DL.cellFlatten(DL.CELL[k].init(dx, dh, { seed: 1 })).v.length);
    const totalMax = Math.max(...counts.map(c => c.total));

    RN.title(g, 0, 0, "parameter count, split by role");
    const bw = 250, bh = 26;
    kinds.forEach((k, i) => {
      const c = counts[i], yy = 24 + i * 62;
      RN.note(g, 0, yy + 17, DL.CELL[k].label, RN.CELLC[k], 11);
      let acc = 0;
      [["U", c.U, DC.accent], ["W", c.W, DC.a2], ["b", c.b, DC.muted]].forEach(pt => {
        const w = bw * pt[1] / totalMax;
        g.append("rect").attr("x", 76 + acc).attr("y", yy).attr("width", Math.max(0.5, w)).attr("height", bh)
          .attr("fill", pt[2]).attr("fill-opacity", 0.6).attr("stroke", pt[2]).attr("stroke-opacity", 0.9);
        if (w > 34) g.append("text").attr("x", 76 + acc + w / 2).attr("y", yy + 17).attr("text-anchor", "middle")
          .attr("font-size", 9).attr("fill", DC.ink).text(DL.big(pt[1]));
        acc += w;
      });
      g.append("text").attr("x", 76 + bw + 14).attr("y", yy + 17).attr("font-size", 11)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(DL.commas(c.total));
      const ok = counted[i] === DL.cellParams(k, dx, dh, 1).total;
      g.append("text").attr("x", 76 + bw + 108).attr("y", yy + 17).attr("font-size", 11)
        .attr("fill", ok ? DC.good : DC.bad).text(ok ? "✓ arrays agree" : "✗ MISMATCH");
      RN.note(g, 76, yy + 40, DL.CELL[k].nGate + " × ( " + DL.commas(dx * dh) + " + " + DL.commas(dh * dh) + " + " +
        (bs === 2 ? "2×" : "") + DL.commas(dh) + " )", DC.muted, 9.5);
    });
    DL.legend(g, [{ label: "input matrices U", color: DC.accent }, { label: "recurrent matrices W", color: DC.a2 },
    { label: "biases", color: DC.muted }], 76, 216, { vertical: false, step: 150, font: 9.5 });

    /* growth curve */
    const gx = 0, gy = 244, gw = 330, gh = 110;
    RN.title(g, gx, gy - 8, "count against hidden width (dₓ fixed)");
    const gg = g.append("g").attr("transform", `translate(${gx + 44},${gy})`);
    const ws = d3.range(0, 41).map(i => Math.round(8 * Math.pow(1024 / 8, i / 40)));
    const x = d3.scaleLog().domain([8, 1024]).range([0, gw - 54]);
    const yv = d3.scaleLog().domain([DL.cellParams("vanilla", dx, 8, bs).total / 2, DL.cellParams("lstm", dx, 1024, bs).total * 2]).range([gh, 0]);
    DL.gridY(gg, yv, gw - 54, 3);
    DL.axisB(gg, x, gh, 4, "dₕ", d3.format("~s"));
    DL.axisL(gg, yv, 3, "params", d => DL.big(d));
    kinds.forEach(k => DL.curve(gg, ws.map(w => [x(w), yv(DL.cellParams(k, dx, w, bs).total)]), { stroke: RN.CELLC[k], w: 1.8 }));
    if (dx >= 8 && dx <= 1024) {
      gg.append("line").attr("x1", x(dx)).attr("x2", x(dx)).attr("y1", 0).attr("y2", gh)
        .attr("stroke", DC.violet).attr("stroke-dasharray", "3 3");
      RN.note(gg, x(dx) + 4, 12, "dₕ = dₓ: the crossover", DC.violet, 9);
    }
    gg.append("line").attr("x1", x(Math.min(1024, Math.max(8, dh)))).attr("x2", x(Math.min(1024, Math.max(8, dh))))
      .attr("y1", 0).attr("y2", gh).attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.6);

    /* matched widths */
    const mx = 400;
    const budget = DL.cellParams("lstm", dx, dh, bs).total;
    const widthFor = k => { let best = 1; for (let m = 1; m <= 4096; m++) { if (DL.cellParams(k, dx, m, bs).total <= budget) best = m; else break; } return best; };
    const wm = { lstm: dh, gru: widthFor("gru"), vanilla: widthFor("vanilla") };
    RN.title(g, mx, gy - 8, cmp === "params" ? "widths that equalise the parameter budget" : "widths in use");
    const wmax = Math.max(...kinds.map(k => cmp === "params" ? wm[k] : dh));
    kinds.forEach((k, i) => {
      const yy = gy + 6 + i * 34, w = cmp === "params" ? wm[k] : dh;
      g.append("rect").attr("x", mx + 76).attr("y", yy).attr("width", Math.max(2, 200 * w / wmax)).attr("height", 20)
        .attr("rx", 3).attr("fill", RN.CELLC[k]).attr("fill-opacity", 0.6).attr("stroke", RN.CELLC[k]);
      RN.note(g, mx, yy + 14, DL.CELL[k].label, RN.CELLC[k], 10);
      g.append("text").attr("x", mx + 76 + Math.max(2, 200 * w / wmax) + 8).attr("y", yy + 14).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink)
        .text("dₕ = " + w + "  ·  " + DL.commas(DL.cellParams(k, dx, w, bs).total));
    });
    RN.note(g, mx, gy + 6 + 3 * 34 + 6, cmp === "params"
      ? "budget " + DL.commas(budget) + " parameters — the largest width each cell can afford"
      : "at equal width the LSTM has exactly 4× the vanilla cell's parameters", DC.muted, 9.5);

    const extra = DL.cellParams("lstm", dx, dh, 2).total - DL.cellParams("lstm", dx, dh, 1).total;
    document.getElementById("pm-readout").innerHTML =
      `dₓ = ${dx}, dₕ = ${dh}, ${bs} bias set${bs === 1 ? "" : "s"} per gate · vanilla <b>${DL.commas(counts[0].total)}</b>, ` +
      `GRU <b>${DL.commas(counts[1].total)}</b>, LSTM <b>${DL.commas(counts[2].total)}</b> — a ratio of exactly ` +
      `<b>1 : ${DL.fmt(counts[1].total / counts[0].total, 2)} : ${DL.fmt(counts[2].total / counts[0].total, 2)}</b> · ` +
      `the formula and a direct count of the allocated arrays agree for all three · ` +
      `the second bias set costs <b>${DL.commas(extra)}</b> extra parameters (= 4dₕ) · ` +
      `at this budget the widths that equalise the count are vanilla <b>${wm.vanilla}</b>, GRU <b>${wm.gru}</b>, LSTM <b>${wm.lstm}</b>.`;
  }
  [Xe, He].forEach(e => e.addEventListener("input", draw));
  [Be, Ce].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 29 · #vr-svg — the variants as edits to one diagram ═══════════ */
(function () {
  const svg = d3.select("#vr-svg");
  if (svg.empty()) return;
  const W = 760, H = 410;
  const Ve = document.getElementById("vr-v"), Xe = document.getElementById("vr-dx"),
    He = document.getElementById("vr-dh"), De = document.getElementById("vr-d");

  const INFO = {
    lstm: { n: (n, m) => 4 * (n * m + m * m + m), why: "the standard cell: three gates and one candidate.", add: [], rm: [] },
    peep: { n: (n, m) => 4 * (n * m + m * m + m) + 3 * m, why: "the three sigmoid gates get to see the cell state, which is what precise timing needs.", add: ["peep"], rm: [] },
    nofg: { n: (n, m) => 3 * (n * m + m * m + m), why: "the original 1997 cell: a pure integrator that can never clear.", add: [], rm: ["f"] },
    noog: { n: (n, m) => 3 * (n * m + m * m + m), why: "the state is always exposed; holding and acting can no longer be separated.", add: [], rm: ["o"] },
    noig: { n: (n, m) => 3 * (n * m + m * m + m), why: "every candidate is written, so distractors overwrite the memory.", add: [], rm: ["i"] },
    cifg: { n: (n, m) => 3 * (n * m + m * m + m), why: "one gate does both jobs: iₜ = 1 − fₜ. This is the GRU's coupling, inside the LSTM.", add: [], rm: ["i"] },
    nogr: { n: (n, m) => 4 * (n * m + m) + 0, why: "the gates read only the input, never the previous state — much cheaper and much worse.", add: [], rm: ["rec"] },
    gru: { n: (n, m) => 3 * (n * m + m * m + m), why: "one state, two gates, and a reset gate that acts before the matrix.", add: ["reset"], rm: ["c", "o"] }
  };

  function draw() {
    const v = Ve.value, dx = +Xe.value, dh = +He.value, d = +De.value;
    document.getElementById("vr-dxv").textContent = dx;
    document.getElementById("vr-dhv").textContent = dh;
    document.getElementById("vr-dv").textContent = d;
    const f = DL.frame(svg, W, H, { l: 14, r: 14, t: 24, b: 18 }), g = f.g;
    const inf = INFO[v];

    /* ---- diagram ---- */
    RN.title(g, 0, 0, "the cell, with this variant's edits marked");
    const yC = 54, yH = 190, gx = [40, 130, 220, 310];
    const gone = k => inf.rm.indexOf(k) >= 0;
    g.append("line").attr("x1", 20).attr("x2", 380).attr("y1", yC).attr("y2", yC)
      .attr("stroke", v === "gru" ? DC.line : DC.teal).attr("stroke-width", v === "gru" ? 1.2 : 6)
      .attr("stroke-opacity", v === "gru" ? 0.35 : 0.5);
    RN.note(g, 20, yC - 14, v === "gru" ? "no separate cell state" : "cell state cₜ", v === "gru" ? DC.bad : DC.teal, 10);
    const boxes = [["f", "forget", DC.a2], ["i", "input", DC.accent], ["c", "candidate", DC.violet], ["o", "output", DC.good]];
    const gruBoxes = [["z", "update", DC.a2], ["r", "reset", DC.accent], ["c", "candidate", DC.violet]];
    const list = v === "gru" ? gruBoxes : boxes;
    list.forEach((b, i) => {
      const x = gx[i], off = gone(b[0]);
      g.append("rect").attr("x", x - 34).attr("y", yH - 18).attr("width", 68).attr("height", 36).attr("rx", 6)
        .attr("fill", DC.panel2).attr("stroke", off ? DC.bad : b[2]).attr("stroke-width", off ? 1.2 : 1.6)
        .attr("stroke-dasharray", off ? "3 3" : null).attr("opacity", off ? 0.5 : 1);
      g.append("text").attr("x", x).attr("y", yH + 4).attr("text-anchor", "middle").attr("font-size", 10)
        .attr("fill", off ? DC.bad : DC.ink).text(b[1]);
      if (off) g.append("line").attr("x1", x - 34).attr("x2", x + 34).attr("y1", yH).attr("y2", yH)
        .attr("stroke", DC.bad).attr("stroke-width", 2);
      DL.arrow(g, x, yH - 20, x, yC + 8, { color: off ? DC.bad : b[2], w: 1.2, head: 4, op: off ? 0.4 : 0.9 });
    });
    if (inf.add.indexOf("peep") >= 0) {
      [0, 1, 3].forEach(i => {
        g.append("path").attr("d", `M${gx[i] - 40},${yC + 6} C${gx[i] - 70},${yC + 60} ${gx[i] - 70},${yH - 40} ${gx[i] - 36},${yH - 6}`)
          .attr("fill", "none").attr("stroke", DC.lime).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
      });
      RN.note(g, 20, yH + 44, "green dashed: the three peepholes (the output gate peeps at the FRESH cₜ)", DC.lime, 9.5);
    }
    if (inf.add.indexOf("reset") >= 0)
      RN.note(g, 20, yH + 44, "the reset gate multiplies hₜ₋₁ BEFORE the candidate's matrix", DC.accent, 9.5);
    if (gone("rec")) RN.note(g, 20, yH + 44, "the recurrent gate matrices are pinned to zero", DC.bad, 9.5);
    if (v === "cifg") RN.note(g, 20, yH + 44, "the input gate is replaced by 1 − fₜ, saving one matrix pair", DC.a2, 9.5);
    RN.note(g, 20, yH + 62, inf.why, DC.muted, 10);

    /* ---- parameter bars ---- */
    const px = 420;
    RN.title(g, px, 0, "parameters at dₓ = " + dx + ", dₕ = " + dh);
    const keys = Object.keys(INFO);
    const nmax = Math.max(...keys.map(k => INFO[k].n(dx, dh)));
    keys.forEach((k, i) => {
      const yy = 16 + i * 22, n = INFO[k].n(dx, dh);
      g.append("rect").attr("x", px + 78).attr("y", yy).attr("width", Math.max(1, 150 * n / nmax)).attr("height", 13)
        .attr("fill", k === v ? DC.a2 : DC.line).attr("fill-opacity", k === v ? 0.85 : 0.6);
      RN.note(g, px, yy + 11, k === "lstm" ? "standard" : k, k === v ? DC.ink : DC.muted, 9);
      g.append("text").attr("x", px + 78 + 156).attr("y", yy + 11).attr("font-size", 9)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", k === v ? DC.ink : DC.muted).text(DL.big(n));
    });

    /* ---- measured gradient at distance ---- */
    const my = 16 + keys.length * 22 + 30;
    RN.title(g, px, my - 8, "measured ‖∂h_T/∂h_k‖₂ at d = " + d);
    const dxs = 3, dhs = 8, T = d + 1, X = RN.seq(T, dxs, 91);
    const meas = {};
    const lnet = () => RN.net("lstm", dxs, dhs, 2, { seed: 31, wScale: 0.3, uScale: 0.3, forgetBias: 1 });
    meas.lstm = DL.specNorm(DL.stateJac(lnet(), X, T - d, 1e-5), 130, 5);
    { const n2 = lnet(); n2.p.b.f = n2.p.b.f.map(() => 14);
      for (let a = 0; a < dhs; a++) for (let b = 0; b < dhs; b++) n2.p.W.f[a][b] = 0;
      for (let a = 0; a < dxs; a++) for (let b = 0; b < dhs; b++) n2.p.U.f[a][b] = 0;
      meas.nofg = DL.specNorm(DL.stateJac(n2, X, T - d, 1e-5), 130, 5); }
    { const n2 = lnet(); n2.p.b.o = n2.p.b.o.map(() => 14);
      for (let a = 0; a < dhs; a++) for (let b = 0; b < dhs; b++) n2.p.W.o[a][b] = 0;
      meas.noog = DL.specNorm(DL.stateJac(n2, X, T - d, 1e-5), 130, 5); }
    { const n2 = lnet(); n2.p.b.i = n2.p.b.i.map(() => 14);
      for (let a = 0; a < dhs; a++) for (let b = 0; b < dhs; b++) n2.p.W.i[a][b] = 0;
      meas.noig = DL.specNorm(DL.stateJac(n2, X, T - d, 1e-5), 130, 5); }
    { const n2 = lnet();
      DL.CELL.lstm.gates.forEach(k => { for (let a = 0; a < dhs; a++) for (let b = 0; b < dhs; b++) n2.p.W[k][a][b] = 0; });
      meas.nogr = DL.specNorm(DL.stateJac(n2, X, T - d, 1e-5), 130, 5); }
    meas.peep = meas.lstm; meas.cifg = meas.lstm;
    meas.gru = DL.specNorm(DL.stateJac(RN.net("gru", dxs, dhs, 2, { seed: 31, wScale: 0.3, uScale: 0.3 }), X, T - d, 1e-5), 130, 5);
    const mv = Object.values(meas).filter(x => x > 0);
    const ms = d3.scaleLog().domain([Math.min(...mv) / 2, Math.max(...mv) * 2]).range([0, 150]).clamp(true);
    ["lstm", "nofg", "noog", "noig", "nogr", "gru"].forEach((k, i) => {
      const yy = my + i * 18;
      g.append("rect").attr("x", px + 78).attr("y", yy).attr("width", Math.max(1, ms(Math.max(meas[k], 1e-300)))).attr("height", 11)
        .attr("fill", k === v ? DC.good : DC.line).attr("fill-opacity", k === v ? 0.85 : 0.55);
      RN.note(g, px, yy + 9, k === "lstm" ? "standard" : k, k === v ? DC.ink : DC.muted, 9);
      g.append("text").attr("x", px + 78 + 156).attr("y", yy + 9).attr("font-size", 8.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.muted).text(DL.fmtE(meas[k], 1));
    });
    RN.note(g, px, my + 6 * 18 + 12, "peephole and coupled gates keep the standard cell's", DC.muted, 8.5);
    RN.note(g, px, my + 6 * 18 + 22, "gradient path unchanged, so they share its bar.", DC.muted, 8.5);

    const base = INFO.lstm.n(dx, dh), mine = inf.n(dx, dh);
    document.getElementById("vr-readout").innerHTML =
      `<b>${Ve.options[Ve.selectedIndex].text}</b> · <b>${DL.commas(mine)}</b> parameters against the standard cell's ` +
      `<b>${DL.commas(base)}</b> — a difference of <b>${mine >= base ? "+" : "−"}${DL.commas(Math.abs(mine - base))}</b> ` +
      `(<b>${DL.fmt(100 * (mine - base) / base, 1)}%</b>) · measured gradient at distance ${d}: ` +
      `<b>${DL.fmtE(meas[v] === undefined ? meas.lstm : meas[v], 3)}</b> against the standard cell's <b>${DL.fmtE(meas.lstm, 3)}</b> · ` +
      `${inf.why}`;
  }
  [Xe, He, De].forEach(e => e.addEventListener("input", draw));
  Ve.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 30 · #gr-svg — GRU and LSTM side by side ═══════════ */
(function () {
  const svg = d3.select("#gr-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const Ve = document.getElementById("gr-v"), Pe = document.getElementById("gr-p"),
    Te = document.getElementById("gr-t"), Be = document.getElementById("gr-bz");

  function draw() {
    const view = Ve.value, pol = Pe.value, bz = +Be.value;
    const T = 40;
    Te.max = String(T);
    const t = Math.min(+Te.value, T);
    document.getElementById("gr-tv").textContent = t;
    document.getElementById("gr-bzv").textContent = DL.fmt(bz, 1);
    const f = DL.frame(svg, W, H, { l: 16, r: 16, t: 24, b: 16 }), g = f.g;

    const dx = 3, dh = 8;
    const lp = DL.CELL.lstm.init(dx, dh, { seed: 31, wScale: 0.35, uScale: 0.5, forgetBias: 1 });
    const gp = DL.CELL.gru.init(dx, dh, { seed: 31, wScale: 0.35, uScale: 0.5 });
    gp.b.z = gp.b.z.map(() => bz);
    const X = RN.seq(T, dx, 811);
    const ls = DL.CELL.lstm.forward(lp, X), gs = DL.CELL.gru.forward(gp, X);
    const zRaw = d3.mean(gs.Z[t - 1]);                 // this page's polarity: how much NEW
    const zShown = pol === "new" ? zRaw : 1 - zRaw;
    const zLabel = pol === "new" ? "z = how much NEW" : "z = how much OLD";
    const carry = 1 - zRaw;                            // always the fraction of hₜ₋₁ kept

    if (view === "reset") {
      RN.title(g, 0, 0, "what the reset gate actually does, at step " + t);
      const hp = gs.Hprev[t - 1], r = gs.R[t - 1], x = X[t - 1];
      const q = hp.map((v, i) => r[i] * v);
      const aff = (src) => {
        const z = gp.b.c.slice();
        for (let i = 0; i < dh; i++) { const hi = src[i]; if (hi) for (let j = 0; j < dh; j++) z[j] += hi * gp.W.c[i][j]; }
        for (let i = 0; i < dx; i++) { const xi = x[i]; if (xi) for (let j = 0; j < dh; j++) z[j] += xi * gp.U.c[i][j]; }
        return z.map(Math.tanh);
      };
      const correct = aff(q);                                    // (r ⊙ h)·W
      const wrongAfter = aff(hp).map((v, i) => r[i] * v);        // r ⊙ (h·W)
      const closed = aff(DL.zeros(dh));                          // r = 0
      const strip = (v, yy, lab, col) => {
        RN.note(g, 0, yy + 12, lab, col, 10);
        const cw = 44;
        v.forEach((val, i) => {
          g.append("rect").attr("x", 200 + i * cw).attr("y", yy).attr("width", cw - 2).attr("height", 20)
            .attr("rx", 2).attr("fill", RN.signColor(val, 1)).attr("stroke", DC.line).attr("stroke-opacity", 0.5);
          g.append("text").attr("x", 200 + i * cw + (cw - 2) / 2).attr("y", yy + 14).attr("text-anchor", "middle")
            .attr("font-size", 8.5).attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink)
            .text(DL.fmt(val, 2));
        });
      };
      strip(hp, 26, "hₜ₋₁", DC.a2);
      strip(r, 56, "reset gate rₜ", DC.accent);
      strip(q, 86, "rₜ ⊙ hₜ₋₁  (what the GRU feeds W_c)", DC.violet);
      strip(correct, 126, "c̃ₜ = tanh( (rₜ⊙hₜ₋₁)W_c + xₜU_c )   ← the GRU", DC.good);
      strip(wrongAfter, 156, "rₜ ⊙ tanh( hₜ₋₁W_c + xₜU_c )   ← the common misreading", DC.bad);
      strip(correct.map((v, i) => v - wrongAfter[i]), 186, "difference", DC.muted);
      strip(closed, 226, "with rₜ = 0: the input read as if the sequence had just begun", DC.teal);
      const md = Math.max(...correct.map((v, i) => Math.abs(v - wrongAfter[i])));
      RN.note(g, 0, 266, "largest difference between the two readings: " + DL.fmt(md, 4), md > 0.02 ? DC.bad : DC.muted, 11);
      RN.note(g, 0, 284, "the reset gate acts INSIDE the affine map, so it decides which parts of the past may", DC.muted, 10);
      RN.note(g, 0, 298, "PARTICIPATE in computing the candidate — not how much of the result to keep.", DC.muted, 10);
      document.getElementById("gr-readout").innerHTML =
        `step ${t} · mean reset gate <b>${DL.fmt(d3.mean(r), 4)}</b> · applying it before the matrix and applying it after ` +
        `differ by up to <b>${DL.fmt(md, 4)}</b> per unit · with the reset gate closed the candidate becomes ` +
        `tanh(xₜU_c + b_c), which is the current input read with no history at all.`;
      return;
    }

    const cellDiag = (ox, title2, boxes, stateLine, col) => {
      RN.title(g, ox, 0, title2, col);
      g.append("line").attr("x1", ox + 10).attr("x2", ox + 300).attr("y1", 46).attr("y2", 46)
        .attr("stroke", col).attr("stroke-width", stateLine).attr("stroke-opacity", 0.5);
      boxes.forEach((b, i) => {
        const x = ox + 30 + i * 74;
        if (b === null) {
          g.append("rect").attr("x", x - 30).attr("y", 130).attr("width", 62).attr("height", 34).attr("rx", 6)
            .attr("fill", "none").attr("stroke", DC.line).attr("stroke-dasharray", "3 3");
          g.append("text").attr("x", x).attr("y", 151).attr("text-anchor", "middle").attr("font-size", 9)
            .attr("fill", DC.muted).text("(none)");
          return;
        }
        g.append("rect").attr("x", x - 30).attr("y", 130).attr("width", 62).attr("height", 34).attr("rx", 6)
          .attr("fill", b[2]).attr("fill-opacity", 0.10 + 0.5 * Math.abs(b[1])).attr("stroke", b[2]).attr("stroke-width", 1.5);
        g.append("text").attr("x", x).attr("y", 145).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", DC.ink).text(b[0]);
        g.append("text").attr("x", x).attr("y", 158).attr("text-anchor", "middle").attr("font-size", 10)
          .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", b[2]).text(DL.fmt(b[1], 3));
        DL.arrow(g, x, 128, x, 54, { color: b[2], w: 1.2, head: 4 });
      });
    };
    const fm = d3.mean(ls.F[t - 1]);
    cellDiag(0, "LSTM — four gate-like computations, a private cell state", [
      ["forget σ", fm, DC.a2], ["input σ", d3.mean(ls.I[t - 1]), DC.accent],
      ["candidate", d3.mean(ls.G[t - 1]), DC.violet], ["output σ", d3.mean(ls.O[t - 1]), DC.good]
    ], 2 + 12 * fm, DC.teal);
    RN.note(g, 10, 32, "cₜ — unbounded, never leaves the cell", DC.teal, 9.5);

    if (view !== "gru") {
      cellDiag(400, "GRU — three, and one state that is also the output", [
        ["update σ", zShown, DC.a2], ["reset σ", d3.mean(gs.R[t - 1]), DC.accent],
        ["candidate", d3.mean(gs.Ct[t - 1]), DC.violet], null
      ], 2 + 12 * carry, DC.a2);
      RN.note(g, 410, 32, "hₜ — bounded, and it IS the output", DC.a2, 9.5);
      RN.note(g, 400, 182, zLabel + " — the polarity this figure prints", DC.ink, 10);
    }

    const lp2 = DL.cellParams("lstm", dx, dh, 1).total, gp2 = DL.cellParams("gru", dx, dh, 1).total;
    const kv = DL.kv(g, 0, 226, { keyW: 210, size: 10.5, lead: 16 });
    kv("LSTM mean forget gate fₜ", DL.fmt(fm, 4), DC.a2);
    kv("GRU update gate as printed", DL.fmt(zShown, 4) + "   (" + zLabel + ")", DC.a2);
    kv("GRU carry fraction 1 − zₜ", DL.fmt(carry, 4), DC.good);
    kv("LSTM parameters (dₓ=" + dx + ", dₕ=" + dh + ")", DL.commas(lp2));
    kv("GRU parameters", DL.commas(gp2) + "   (" + DL.fmt(100 * gp2 / lp2, 0) + "% of the LSTM)", DC.good);
    kv("‖cₜ‖ (LSTM) against ‖hₜ‖ (GRU)", DL.fmt(RN.norm(ls.C[t - 1]), 3) + "  /  " + DL.fmt(RN.norm(gs.H[t - 1]), 3));
    RN.note(g, 0, 340, "switching the polarity changes every printed gate value to one minus itself and leaves", DC.muted, 10);
    RN.note(g, 0, 354, "the state trajectory bit for bit identical — it is the same model under two conventions.", DC.muted, 10);
    /* the identical trajectory, drawn to prove it */
    const nrm = gs.H.map(h => RN.norm(h));
    const xs = d3.scaleLinear().domain([1, T]).range([440, 740]);
    const ys = d3.scaleLinear().domain([0, Math.max(...nrm) * 1.1]).range([368, 300]);
    DL.curve(g, nrm.map((v, i) => [xs(i + 1), ys(v)]), { stroke: DC.a2, w: 1.6 });
    RN.note(g, 440, 296, "‖hₜ‖ of the GRU — unchanged by the polarity switch", DC.muted, 9);

    document.getElementById("gr-readout").innerHTML =
      `step ${t} · LSTM mean forget gate <b>${DL.fmt(fm, 4)}</b> · GRU update gate printed as <b>${DL.fmt(zShown, 4)}</b> ` +
      `under the convention "${zLabel}" — the fraction of hₜ₋₁ actually carried is <b>${DL.fmt(carry, 4)}</b> either way · ` +
      `parameters at dₓ=${dx}, dₕ=${dh}: LSTM <b>${DL.commas(lp2)}</b>, GRU <b>${DL.commas(gp2)}</b> ` +
      `(<b>${DL.fmt(100 * gp2 / lp2, 0)}%</b>, exactly 3/4).`;
  }
  [Te, Be].forEach(e => e.addEventListener("input", draw));
  [Ve, Pe].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 31 · #cm-svg — the adding problem, fitted live ═══════════ */
(function () {
  const svg = d3.select("#cm-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const Te = document.getElementById("cm-T"), Me = document.getElementById("cm-m"),
    He = document.getElementById("cm-dh"), Le = document.getElementById("cm-lr"),
    Go = document.getElementById("cm-go"), Rs = document.getElementById("cm-rs");
  const BASE = 1 / 6, KINDS = ["vanilla", "gru", "lstm"];
  let S = null;

  function widthFor(kind, budget, dx) { let best = 2; for (let m = 2; m <= 300; m++) { if (DL.cellParams(kind, dx, m, 1).total <= budget) best = m; else break; } return best; }
  function reset() {
    const T = +Te.value, match = Me.value, dhL = +He.value, lr = +Le.value;
    const dx = 2, budget = DL.cellParams("lstm", dx, dhL, 1).total;
    const w = { lstm: dhL, gru: match === "params" ? widthFor("gru", budget, dx) : dhL, vanilla: match === "params" ? widthFor("vanilla", budget, dx) : dhL };
    S = { T, lr, w, dx, steps: 0, r: DL.rng(771), nets: {}, tr: {}, hist: {} };
    KINDS.forEach(k => {
      const net = RN.net(k, dx, w[k], 1, { seed: 771, out: "linear", uScale: 0.5, wScale: 1 / Math.sqrt(w[k]), forgetBias: 1, vScale: 0.1 });
      net.c[0] = 1;
      S.nets[k] = net; S.tr[k] = RN.trainer(net, lr, 1.0); S.hist[k] = [];
    });
  }
  function train(n) {
    for (let s = 0; s < n; s++) {
      const B = RN.addBatch(S.T, 8, S.r);
      KINDS.forEach(k => { const o = S.tr[k].step(B); if (s % 10 === 0) S.hist[k].push([S.steps + s, 2 * o.loss]); });
    }
    S.steps += n;
  }
  function evaluate(k) {
    const B = RN.addBatch(S.T, 150, DL.rng(4321));
    let mse = 0;
    for (const b of B) { const e = DL.seqForward(S.nets[k], b.X).A[S.T - 1][0] - b.s; mse += e * e / B.length; }
    return mse;
  }
  function firstMarkerGrad(k) {
    const B = RN.addBatch(S.T, 6, DL.rng(99));
    let tot = 0;
    for (const b of B) {
      const g2 = DL.seqBPTT(S.nets[k], b.X, b.Y);
      /* the gradient at the state of the FIRST marked step is what has to survive */
      const st = g2.st, T = S.T;
      let dn = DL.zeros(S.w[k]);
      const cellB = DL.CELL[k];
      const sub = DL.sliceState(st, b.i1, T);
      const dH = [];
      for (let u = b.i1; u < T; u++) dH.push(g2.dH[u].slice());
      const gg = cellB.backward(S.nets[k].p, sub, dH);
      tot += RN.norm(gg.dh0) / B.length;
    }
    return tot;
  }

  function draw() {
    document.getElementById("cm-Tv").textContent = Te.value;
    document.getElementById("cm-dhv").textContent = He.value;
    document.getElementById("cm-lrv").textContent = DL.fmt(+Le.value, 3);
    const f = DL.frame(svg, W, H, { l: 54, r: 250, t: 24, b: 40 }), g = f.g;

    const all = KINDS.reduce((a, k) => a.concat(S.hist[k].map(p => p[1])), []).filter(v => v > 0);
    const x = d3.scaleLinear().domain([0, Math.max(10, S.steps)]).range([0, f.iw]);
    const y = d3.scaleLog().domain([Math.max(1e-5, Math.min(...all.concat([BASE])) / 2), Math.max(...all.concat([BASE])) * 2]).range([f.ih, 0]).clamp(true);
    DL.gridY(g, y, f.iw, 5);
    DL.axisB(g, x, f.ih, 5, "optimisation step");
    DL.axisL(g, y, 5, "mean squared error", v => DL.fmtE(v, 0));
    g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y(BASE)).attr("y2", y(BASE))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "4 3");
    RN.note(g, 4, y(BASE) - 4, "the trivial predictor, MSE = 0.1667", DC.muted, 9);
    KINDS.forEach(k => DL.curve(g, S.hist[k].map(p => [x(p[0]), y(Math.max(p[1], 1e-6))]), { stroke: RN.CELLC[k], w: 2 }));
    DL.legend(g, KINDS.map(k => ({ label: DL.CELL[k].label, color: RN.CELLC[k] })), 4, f.ih + 30,
      { vertical: false, step: 110, font: 9.5 });

    const sx = f.iw + 16, bw = 130;
    RN.title(g, sx, -4, "held-out MSE on fresh sequences");
    const errs = {}; KINDS.forEach(k => errs[k] = evaluate(k));
    const emax = Math.max(BASE, ...Object.values(errs));
    KINDS.forEach((k, i) => {
      const yy = 12 + i * 34;
      g.append("rect").attr("x", sx).attr("y", yy).attr("width", Math.max(1, bw * errs[k] / emax)).attr("height", 13)
        .attr("fill", RN.CELLC[k]).attr("fill-opacity", 0.75);
      g.append("text").attr("x", sx + bw + 8).attr("y", yy + 11).attr("font-size", 9.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", errs[k] < 0.02 ? DC.good : (errs[k] > 0.12 ? DC.bad : DC.ink))
        .text(DL.fmt(errs[k], 5));
      RN.note(g, sx, yy + 24, DL.CELL[k].label + "  dₕ=" + S.w[k] + "  " +
        DL.commas(DL.cellParams(k, S.dx, S.w[k], 1).total) + " params", RN.CELLC[k], 9);
    });
    g.append("line").attr("x1", sx + bw * BASE / emax).attr("x2", sx + bw * BASE / emax).attr("y1", 8).attr("y2", 12 + 3 * 34)
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");

    const gy = 12 + 3 * 34 + 26;
    RN.title(g, sx, gy - 6, "gradient reaching the FIRST marked step");
    const gr2 = {}; KINDS.forEach(k => gr2[k] = firstMarkerGrad(k));
    const gmax = Math.max(...Object.values(gr2).filter(v => v > 0), 1e-12);
    const gs2 = d3.scaleLog().domain([Math.max(1e-14, Math.min(...Object.values(gr2).filter(v => v > 0)) / 3), gmax * 2]).range([0, bw]).clamp(true);
    KINDS.forEach((k, i) => {
      const yy = gy + 6 + i * 20;
      g.append("rect").attr("x", sx).attr("y", yy).attr("width", Math.max(1, gs2(Math.max(gr2[k], 1e-300)))).attr("height", 12)
        .attr("fill", RN.CELLC[k]).attr("fill-opacity", 0.6);
      g.append("text").attr("x", sx + bw + 8).attr("y", yy + 10).attr("font-size", 9)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.muted).text(DL.fmtE(gr2[k], 2));
    });
    RN.note(g, sx, gy + 6 + 3 * 20 + 14, "this has to be non-zero for the task", DC.muted, 9);
    RN.note(g, sx, gy + 6 + 3 * 20 + 25, "to be learnable at all.", DC.muted, 9);

    const beat = KINDS.filter(k => errs[k] < BASE * 0.9);
    document.getElementById("cm-readout").innerHTML =
      `T = ${S.T}, ${Me.value === "params" ? "equal parameter budget" : "equal hidden width"}, ${S.steps} steps trained · ` +
      `held-out MSE: vanilla <b>${DL.fmt(errs.vanilla, 5)}</b> (dₕ=${S.w.vanilla}), GRU <b>${DL.fmt(errs.gru, 5)}</b> (dₕ=${S.w.gru}), ` +
      `LSTM <b>${DL.fmt(errs.lstm, 5)}</b> (dₕ=${S.w.lstm}) against the trivial predictor's 0.16667 · ` +
      `beating the trivial predictor: <b>${beat.length ? beat.map(k => DL.CELL[k].label).join(", ") : "none yet"}</b> · ` +
      `gradient reaching the first marker: vanilla <b>${DL.fmtE(gr2.vanilla, 2)}</b>, GRU <b>${DL.fmtE(gr2.gru, 2)}</b>, ` +
      `LSTM <b>${DL.fmtE(gr2.lstm, 2)}</b>.`;
  }
  function hard() { reset(); train(120); draw(); }
  [Te, He, Le].forEach(e => e.addEventListener("input", hard));
  Me.addEventListener("change", hard);
  Go.addEventListener("click", () => { train(150); draw(); });
  Rs.addEventListener("click", hard);
  hard();
})();

/* ═══════════ 32 · #bi-svg — one direction against two ═══════════ */
(function () {
  const svg = d3.select("#bi-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Te = document.getElementById("bi-T"), Qe = document.getElementById("bi-q"),
    Ee = document.getElementById("bi-e"), Me = document.getElementById("bi-m");

  function draw() {
    const T = +Te.value;
    Qe.max = String(T); Ee.max = String(T);
    const q = Math.min(+Qe.value, T), ev = Math.min(+Ee.value, T), mode = Me.value;
    document.getElementById("bi-Tv").textContent = T;
    document.getElementById("bi-qv").textContent = q;
    document.getElementById("bi-ev").textContent = ev;
    const f = DL.frame(svg, W, H, { l: 30, r: 190, t: 24, b: 34 }), g = f.g;

    const dx = 3, dh = 8;
    const pf = DL.CELL.vanilla.init(dx, dh, { seed: 41, rho: 0.9, uScale: 0.7 });
    const pb = DL.CELL.vanilla.init(dx, dh, { seed: 42, rho: 0.9, uScale: 0.7 });
    const X = RN.seq(T, dx, 841);

    /* measured influence: perturb input at position p, see the change at q */
    function influence(p) {
      const eps = 1e-4;
      const runF = XX => DL.CELL.vanilla.forward(pf, XX).H[q - 1];
      const runB = XX => { const R = DL.CELL.vanilla.forward(pb, XX.slice().reverse()).H; return R[T - q]; };
      const Xp = X.map(r => r.slice()); Xp[p - 1][0] += eps;
      const Xm = X.map(r => r.slice()); Xm[p - 1][0] -= eps;
      const dF = RN.norm(runF(Xp).map((v, i) => (v - runF(Xm)[i]) / (2 * eps)));
      const dB = RN.norm(runB(Xp).map((v, i) => (v - runB(Xm)[i]) / (2 * eps)));
      return { uni: dF, bi: Math.hypot(dF, dB) };
    }
    const inf = d3.range(1, T + 1).map(p => influence(p));

    /* ---- graph ---- */
    const step = Math.min(46, f.iw / T);
    const px = i => 12 + (i - 1) * step;
    const yF = 44, yX = 116, yB = 84;
    for (let t = 1; t <= T; t++) {
      if (t > 1) DL.arrow(g, px(t - 1) + 9, yF, px(t) - 9, yF, { color: DC.accent, w: 1.2, head: 4, op: 0.7 });
      if (t < T) DL.arrow(g, px(t + 1) - 9, yB, px(t) + 9, yB, { color: DC.violet, w: 1.2, head: 4, op: mode === "uni" ? 0.15 : 0.7 });
      DL.arrow(g, px(t), yX - 9, px(t), yF + 9, { color: DC.line, w: 1, head: 3, op: 0.5 });
      [[yF, DC.accent], [yB, mode === "uni" ? DC.line : DC.violet]].forEach(d => {
        g.append("circle").attr("cx", px(t)).attr("cy", d[0]).attr("r", 8)
          .attr("fill", DC.panel2).attr("stroke", d[1]).attr("stroke-width", 1.2);
      });
      g.append("circle").attr("cx", px(t)).attr("cy", yX).attr("r", 8)
        .attr("fill", t === ev ? DC.a2 : DC.panel2).attr("fill-opacity", t === ev ? 0.6 : 1)
        .attr("stroke", t === ev ? DC.a2 : DC.line).attr("stroke-width", t === ev ? 2 : 1);
      g.append("text").attr("x", px(t)).attr("y", yX + 22).attr("text-anchor", "middle")
        .attr("font-size", 8).attr("fill", DC.muted).text(t);
    }
    RN.note(g, 12, yF - 16, "forward chain h", DC.accent, 9.5);
    RN.note(g, 12, yB + 20, "backward chain g", mode === "uni" ? DC.line : DC.violet, 9.5);
    g.append("circle").attr("cx", px(q)).attr("cy", yF).attr("r", 12).attr("fill", "none")
      .attr("stroke", DC.good).attr("stroke-width", 2);
    RN.note(g, px(q) - 12, yF - 16, "query", DC.good, 9);
    /* highlight the accessible route */
    if (ev <= q) for (let t = ev; t < q; t++)
      g.append("line").attr("x1", px(t)).attr("x2", px(t + 1)).attr("y1", yF).attr("y2", yF)
        .attr("stroke", DC.accent).attr("stroke-width", 3.4).attr("stroke-opacity", 0.45);
    if (ev >= q && mode !== "uni") for (let t = q; t < ev; t++)
      g.append("line").attr("x1", px(t)).attr("x2", px(t + 1)).attr("y1", yB).attr("y2", yB)
        .attr("stroke", DC.violet).attr("stroke-width", 3.4).attr("stroke-opacity", 0.45);

    /* ---- influence profile ---- */
    const oy = 168, gh = f.ih - oy - 4;
    RN.title(g, 0, oy - 8, "measured influence of each input position on the output at step " + q);
    const gg = g.append("g").attr("transform", `translate(0,${oy})`);
    const x = d3.scaleLinear().domain([1, T]).range([0, f.iw]);
    const vals = inf.reduce((a, d) => a.concat([d.uni, d.bi]), []).filter(v => v > 0);
    const y = d3.scaleLog().domain([Math.max(1e-12, Math.min(...vals) / 3), Math.max(...vals) * 3]).range([gh, 0]).clamp(true);
    DL.gridY(gg, y, f.iw, 4);
    DL.axisB(gg, x, gh, 6, "perturbed position");
    DL.axisL(gg, y, 4, "|∂output / ∂input|", v => DL.fmtE(v, 0));
    gg.append("rect").attr("x", x(q)).attr("y", 0).attr("width", Math.max(0, f.iw - x(q))).attr("height", gh)
      .attr("fill", DC.bad).attr("fill-opacity", 0.06);
    RN.note(gg, x(q) + 4, 12, "the future — invisible to a causal model", DC.bad, 9);
    if (mode !== "bi") DL.curve(gg, inf.map((d, i) => [x(i + 1), y(Math.max(d.uni, 1e-300))]), { stroke: DC.accent, w: 2 });
    if (mode !== "uni") DL.curve(gg, inf.map((d, i) => [x(i + 1), y(Math.max(d.bi, 1e-300))]), { stroke: DC.violet, w: 2 });
    gg.append("line").attr("x1", x(ev)).attr("x2", x(ev)).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.a2).attr("stroke-dasharray", "3 3");

    const rightUni = d3.sum(inf.slice(q).map(d => d.uni));
    const rightBi = d3.sum(inf.slice(q).map(d => d.bi));
    const sx = f.iw + 16;
    const kv = DL.kv(g, sx, 6, { keyW: 138, size: 10.5, lead: 16 });
    kv("evidence at step", String(ev) + (ev > q ? "  (in the future)" : ev < q ? "  (in the past)" : "  (here)"), ev > q ? DC.bad : DC.good);
    kv("causal influence", DL.fmtE(inf[ev - 1].uni, 3), ev > q ? DC.bad : DC.accent);
    kv("bidirectional influence", DL.fmtE(inf[ev - 1].bi, 3), DC.violet);
    kv("ratio", ev > q ? "∞ — one is exactly 0" : DL.fmt(inf[ev - 1].bi / Math.max(1e-300, inf[ev - 1].uni), 3));
    kv("total influence to the right", DL.fmtE(rightUni, 2) + " / " + DL.fmtE(rightBi, 2));
    kv("can a causal model solve it?", ev > q ? "NO" : "yes", ev > q ? DC.bad : DC.good);
    RN.note(g, sx, 116, ev > q ? "the causal curve is IDENTICALLY ZERO" : "", DC.bad, 9.5);
    RN.note(g, sx, 128, ev > q ? "to the right of the query — not small," : "", DC.bad, 9.5);
    RN.note(g, sx, 140, ev > q ? "zero, as a matter of graph structure." : "", DC.bad, 9.5);
    RN.note(g, sx, 164, "and that is exactly why using a", DC.muted, 9.5);
    RN.note(g, sx, 176, "bidirectional layer where the future", DC.muted, 9.5);
    RN.note(g, sx, 188, "is unavailable is leakage, not a model.", DC.muted, 9.5);

    document.getElementById("bi-readout").innerHTML =
      `query at step ${q}, evidence at step ${ev} · measured influence of the evidence on the query output: ` +
      `causal <b>${DL.fmtE(inf[ev - 1].uni, 3)}</b>, bidirectional <b>${DL.fmtE(inf[ev - 1].bi, 3)}</b> · ` +
      `summed over every position to the RIGHT of the query the causal model has <b>${DL.fmtE(rightUni, 2)}</b> ` +
      `and the bidirectional model <b>${DL.fmtE(rightBi, 2)}</b> · a causal model ` +
      `<b>${ev > q ? "cannot solve this task at all" : "can solve this task"}</b>.`;
  }
  [Te, Qe, Ee].forEach(e => e.addEventListener("input", draw));
  Me.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 33 · #sk-svg — stacking ═══════════ */
(function () {
  const svg = d3.select("#sk-svg");
  if (svg.empty()) return;
  const W = 760, H = 410;
  const Le = document.getElementById("sk-L"), Te = document.getElementById("sk-T"),
    He = document.getElementById("sk-dh"), Se = document.getElementById("sk-s"),
    Ce = document.getElementById("sk-c");

  function draw() {
    const L = +Le.value, T = +Te.value, dh = +He.value, style = Se.value, cell = Ce.value;
    document.getElementById("sk-Lv").textContent = L;
    document.getElementById("sk-Tv").textContent = T;
    document.getElementById("sk-dhv").textContent = dh;
    const f = DL.frame(svg, W, H, { l: 50, r: 186, t: 24, b: 34 }), g = f.g;

    /* run the stack for real */
    const dx = 2, X0 = RN.seq(T, dx, 861);
    const layers = [];
    let inp = X0, din = dx;
    for (let l = 0; l < L; l++) {
      const p = DL.CELL[cell].init(din, dh, { seed: 50 + l, wScale: cell === "vanilla" ? null : 0.5,
        rho: cell === "vanilla" ? 1.0 : null, uScale: 0.4 / Math.sqrt(din), forgetBias: 1 });
      const st = DL.CELL[cell].forward(p, inp);
      layers.push({ p, st });
      inp = st.H; din = dh;
    }
    /* per-layer autocorrelation of the state */
    const maxLag = Math.min(40, T - 2);
    const acorr = layers.map(ly => {
      const Hs = ly.st.H;
      const mu = DL.zeros(dh);
      Hs.forEach(h => h.forEach((v, i) => mu[i] += v / Hs.length));
      const out = [];
      for (let lag = 0; lag <= maxLag; lag++) {
        let num = 0, den = 0;
        for (let t = 0; t + lag < Hs.length; t++) for (let i = 0; i < dh; i++) {
          num += (Hs[t][i] - mu[i]) * (Hs[t + lag][i] - mu[i]);
        }
        for (let t = 0; t < Hs.length; t++) for (let i = 0; i < dh; i++) den += (Hs[t][i] - mu[i]) * (Hs[t][i] - mu[i]);
        out.push(den > 0 ? num / den * (Hs.length / Math.max(1, Hs.length - lag)) : 0);
      }
      const n0 = out[0] || 1;
      return out.map(v => v / n0);
    });

    /* ---- diagram ---- */
    const gw = 300, step = Math.min(26, gw / Math.min(T, 14)), showT = Math.min(T, 14);
    RN.title(g, 0, 0, "the stack, one row per layer");
    const lyH = 26;
    for (let l = 0; l < L; l++) for (let t = 0; t < showT; t++) {
      const x = 6 + t * step, y = 16 + (L - 1 - l) * lyH;
      if (t > 0) DL.arrow(g, x - step + 6, y, x - 6, y, { color: DC.a2, w: 1, head: 3, op: 0.6 });
      if (l > 0) DL.arrow(g, x, y + lyH - 6, x, y + 6, { color: DC.accent, w: 1, head: 3, op: 0.6 });
      if (style === "skip" && l > 1) {
        g.append("path").attr("d", `M${x + 5},${y + 2 * lyH - 4} C${x + 16},${y + lyH} ${x + 16},${y + lyH} ${x + 5},${y + 4}`)
          .attr("fill", "none").attr("stroke", DC.good).attr("stroke-width", 1).attr("stroke-opacity", 0.7);
      }
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", 5)
        .attr("fill", DC.panel2).attr("stroke", DC.a2).attr("stroke-width", 1);
      if (style === "trans" && t > 0) g.append("circle").attr("cx", x - step / 2).attr("cy", y).attr("r", 2)
        .attr("fill", DC.violet);
    }
    if (T > showT) RN.note(g, 6 + showT * step + 4, 16 + (L - 1) * lyH / 2, "⋯", DC.muted, 12);
    for (let l = 0; l < L; l++) RN.note(g, -34, 20 + (L - 1 - l) * lyH, "layer " + (l + 1), DC.muted, 9);
    /* the highlighted shortest path from an input to the top layer, d steps later */
    const dsel = Math.min(8, showT - 1);
    const mult = style === "trans" ? 2 : 1;
    const pathLen = (style === "skip" ? Math.max(1, Math.ceil(L / 2)) : L) + dsel * mult;
    RN.note(g, 0, 16 + L * lyH + 14, `shortest path from an input to the top layer ${dsel} steps later: ${pathLen} transformations`, DC.violet, 10);

    /* ---- autocorrelation ---- */
    const ay = 16 + L * lyH + 34, ah = f.ih - ay - 6;
    RN.title(g, 0, ay - 8, "measured state autocorrelation, one curve per layer");
    const gg = g.append("g").attr("transform", `translate(0,${ay})`);
    const x = d3.scaleLinear().domain([0, maxLag]).range([0, f.iw]);
    const y = d3.scaleLinear().domain([Math.min(-0.1, d3.min(acorr, a => d3.min(a))), 1]).range([ah, 0]);
    DL.gridY(gg, y, f.iw, 4);
    DL.axisB(gg, x, ah, 6, "lag");
    DL.axisL(gg, y, 4, "autocorrelation");
    gg.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y(0.5)).attr("y2", y(0.5))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
    const halfs = [];
    acorr.forEach((a, l) => {
      const col = d3.interpolateTurbo(L === 1 ? 0.5 : l / (L - 1));
      DL.curve(gg, a.map((v, i) => [x(i), y(v)]), { stroke: col, w: 1.9 });
      let h = null;
      for (let i = 0; i < a.length; i++) if (a[i] < 0.5) { h = i; break; }
      halfs.push(h);
      if (h !== null) gg.append("circle").attr("cx", x(h)).attr("cy", y(a[h])).attr("r", 2.6).attr("fill", col);
    });
    DL.legend(gg, acorr.map((_, l) => ({ label: "layer " + (l + 1), color: d3.interpolateTurbo(L === 1 ? 0.5 : l / (L - 1)) })),
      f.iw - 60, 4, { vertical: true, gap: 13, font: 9 });

    const nPar = (() => { let tot = 0, di = dx; for (let l = 0; l < L; l++) { tot += DL.cellParams(cell, di, dh, 1).total * (style === "trans" ? 2 : 1); di = dh; } return tot; })();
    const sx = f.iw + 16;
    const kv = DL.kv(g, sx, 6, { keyW: 128, size: 10.5, lead: 16 });
    kv("layers", String(L));
    kv("cell", DL.CELL[cell].label);
    kv("parameters", DL.commas(nPar));
    kv("shortest path, d = " + dsel, String(pathLen), pathLen > 20 ? DC.bad : DC.ink);
    kv("path per extra step", String(mult) + (style === "trans" ? "  (doubled)" : ""), style === "trans" ? DC.bad : DC.ink);
    halfs.forEach((h, l) => kv("layer " + (l + 1) + " half-life", h === null ? "> " + maxLag : String(h) + " steps",
      d3.interpolateTurbo(L === 1 ? 0.5 : l / (L - 1))));

    const grew = halfs.filter(v => v !== null);
    document.getElementById("sk-readout").innerHTML =
      `${L} layer${L === 1 ? "" : "s"} of ${DL.CELL[cell].label}, ${Se.options[Se.selectedIndex].text} · ` +
      `${DL.commas(nPar)} parameters · measured autocorrelation half-life per layer: ` +
      `<b>${halfs.map(h => h === null ? "> " + maxLag : h).join(", ")}</b> steps` +
      (grew.length > 1
        ? (grew[grew.length - 1] > grew[0]
          ? " — <b>increasing with depth</b>, which is the effect this section claims"
          : (grew.every(v => v === grew[0])
            ? ` — <b>all equal</b>: at this setting the state decorrelates within ${grew[0]} step${grew[0] === 1 ? "" : "s"} at every layer, so no broadening appears at all. A plain tanh cell at a contracting spectral radius does not show the effect; switch the cell to a gated one, which does.`
            : " — <b>not monotone</b> here"))
        : "") +
      ` · the shortest path from an input to the top layer ${dsel} steps later is <b>${pathLen}</b> transformations, ` +
      `and a deeper transition would ${style === "trans" ? "have" : ""} double${style === "trans" ? "d" : " "} the per-step cost.`;
  }
  [Le, Te, He].forEach(e => e.addEventListener("input", draw));
  [Se, Ce].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 34 · #ss-svg — the encoder-decoder pair ═══════════ */
(function () {
  const svg = d3.select("#ss-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Xe = document.getElementById("ss-nx"), Ye = document.getElementById("ss-ny"),
    Ce = document.getElementById("ss-c"), Ee = document.getElementById("ss-e"),
    He = document.getElementById("ss-dh");

  function draw() {
    const nx = +Xe.value, ny = +Ye.value, cmode = Ce.value, emode = Ee.value, dh = +He.value;
    document.getElementById("ss-nxv").textContent = nx;
    document.getElementById("ss-nyv").textContent = ny;
    document.getElementById("ss-dhv").textContent = dh;
    const f = DL.frame(svg, W, H, { l: 16, r: 16, t: 24, b: 16 }), g = f.g;

    const dx = 32, dy = 32;
    const bidir = emode === "bi", rev = emode === "rev";
    const ctxDim = dh * (bidir ? 2 : 1);
    const step = Math.min(44, (f.iw - 90) / (nx + ny + 1));
    const ex = i => 16 + i * step, dxp = i => 16 + (nx + 1 + i) * step;
    const yH = 96, yB = 132, yX = 186, yO = 50;
    const r = 11;

    RN.title(g, 0, 0, "encoder");
    RN.title(g, dxp(0) - 10, 0, "decoder");
    for (let i = 0; i < nx; i++) {
      const X = ex(i);
      if (i > 0) DL.arrow(g, ex(i - 1) + r + 1, yH, X - r - 1, yH, { color: DC.a2, w: 1.3, head: 4 });
      if (bidir && i < nx - 1) DL.arrow(g, ex(i + 1) - r - 1, yB, X + r + 1, yB, { color: DC.violet, w: 1.2, head: 4 });
      DL.arrow(g, X, yX - r - 1, X, (bidir ? yB : yH) + r + 1, { color: DC.line, w: 1, head: 3, op: 0.6 });
      if (bidir) DL.arrow(g, X, yB - r - 1, X, yH + r + 1, { color: DC.line, w: 0.8, head: 3, op: 0.4 });
      [[yH, DC.a2]].concat(bidir ? [[yB, DC.violet]] : []).forEach(d => {
        g.append("circle").attr("cx", X).attr("cy", d[0]).attr("r", r).attr("fill", DC.panel2).attr("stroke", d[1]);
      });
      g.append("circle").attr("cx", X).attr("cy", yX).attr("r", r).attr("fill", DC.panel2).attr("stroke", DC.accent);
      g.append("text").attr("x", X).attr("y", yX + 3.5).attr("text-anchor", "middle").attr("font-size", 8)
        .attr("fill", DC.muted).text("x" + (rev ? nx - i : i + 1));
    }
    /* the context node */
    const cX = ex(nx - 1) + step * 0.6;
    g.append("rect").attr("x", cX - 16).attr("y", yH - 15).attr("width", 32).attr("height", 30).attr("rx", 6)
      .attr("fill", DC.teal).attr("fill-opacity", 0.22).attr("stroke", DC.teal).attr("stroke-width", 2);
    g.append("text").attr("x", cX).attr("y", yH + 4).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", DC.teal).text("C");
    DL.arrow(g, ex(nx - 1) + r + 1, yH, cX - 17, yH, { color: DC.teal, w: 3, head: 7 });
    if (bidir) DL.arrow(g, ex(0) - 4, yB, cX - 12, yH + 12, { color: DC.violet, w: 2, head: 5 });
    RN.note(g, cX - 26, yH - 22, ctxDim + " numbers", DC.teal, 9);

    for (let i = 0; i < ny; i++) {
      const X = dxp(i);
      if (i > 0) DL.arrow(g, dxp(i - 1) + r + 1, yH, X - r - 1, yH, { color: DC.good, w: 1.3, head: 4 });
      else DL.arrow(g, cX + 17, yH, X - r - 1, yH, { color: cmode === "every" ? DC.line : DC.teal, w: cmode === "every" ? 1 : 3, head: 6 });
      if (cmode !== "init") DL.arrow(g, cX + 8, yH + 14, X - 4, yH + r + 2, { color: DC.teal, w: 1, head: 3, op: 0.55, dash: "3 2" });
      g.append("circle").attr("cx", X).attr("cy", yH).attr("r", r).attr("fill", DC.panel2).attr("stroke", DC.good);
      DL.arrow(g, X, yH - r - 1, X, yO + r + 1, { color: DC.good, w: 1.1, head: 4 });
      g.append("circle").attr("cx", X).attr("cy", yO).attr("r", r).attr("fill", DC.panel2).attr("stroke", DC.good);
      g.append("text").attr("x", X).attr("y", yO + 3.5).attr("text-anchor", "middle").attr("font-size", 8)
        .attr("fill", DC.muted).text("y" + (i + 1));
    }
    if (rev) {
      const a1 = ex(nx - 1), a2 = dxp(0);
      g.append("path").attr("d", `M${a1},${yX + 16} C${(a1 + a2) / 2},${yX + 52} ${(a1 + a2) / 2},${yO + 40} ${a2},${yO + 16}`)
        .attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
      RN.note(g, (a1 + a2) / 2 - 60, yX + 44, "the first SOURCE word is now adjacent to the first TARGET word", DC.a2, 9.5);
    }

    /* the compression bar */
    const by = 236;
    RN.title(g, 0, by - 8, "how much has to fit through C");
    const srcNums = nx * dx, ctxNums = ctxDim;
    const bw = 420;
    g.append("rect").attr("x", 130).attr("y", by).attr("width", bw).attr("height", 20).attr("rx", 3)
      .attr("fill", DC.accent).attr("fill-opacity", 0.5).attr("stroke", DC.accent);
    RN.note(g, 0, by + 14, "the source", DC.accent, 10);
    g.append("text").attr("x", 130 + bw + 8).attr("y", by + 14).attr("font-size", 10)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(DL.commas(srcNums) + " numbers");
    g.append("rect").attr("x", 130).attr("y", by + 28).attr("width", Math.max(2, bw * ctxNums / srcNums)).attr("height", 20).attr("rx", 3)
      .attr("fill", DC.teal).attr("fill-opacity", 0.6).attr("stroke", DC.teal);
    RN.note(g, 0, by + 42, "the context C", DC.teal, 10);
    g.append("text").attr("x", 130 + Math.max(2, bw * ctxNums / srcNums) + 8).attr("y", by + 42).attr("font-size", 10)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(DL.commas(ctxNums) + " numbers");

    const pathLen = (rev ? 1 : nx) + ny;
    const encPar = DL.cellParams("lstm", dx, dh, 1).total * (bidir ? 2 : 1);
    const decPar = DL.cellParams("lstm", dy + (cmode !== "init" ? ctxDim : 0), dh, 1).total + dh * dy + dy;
    const kv = DL.kv(g, 0, 306, { keyW: 250, size: 10.5, lead: 15 });
    kv("compression ratio, source : context", DL.fmt(srcNums / ctxNums, 1) + " : 1", srcNums / ctxNums > 8 ? DC.bad : DC.ink);
    kv("gradient path, last loss → first source word", String(pathLen) + " recurrent steps", pathLen > 20 ? DC.bad : DC.ink);
    kv("context dimension", String(ctxDim));
    kv("parameters, encoder + decoder", DL.commas(encPar) + " + " + DL.commas(decPar) + " = " + DL.commas(encPar + decPar));
    kv("rank bound on what survives", DL.fmt(Math.min(1, ctxNums / srcNums), 4) + "  (§36)", DC.a2);

    document.getElementById("ss-readout").innerHTML =
      `source ${nx} steps × ${dx} features = <b>${DL.commas(srcNums)}</b> numbers, context = <b>${DL.commas(ctxNums)}</b> numbers, ` +
      `a compression of <b>${DL.fmt(srcNums / ctxNums, 1)} : 1</b> · the gradient from the last decoder loss to the first source ` +
      `word travels <b>${pathLen}</b> recurrent steps${rev ? " — reversing the source cut this to 1 for the FIRST word" : ""} · ` +
      `parameters <b>${DL.commas(encPar + decPar)}</b> · the rank argument of §36 caps the average recoverable fraction of the ` +
      `source at <b>${DL.fmt(Math.min(1, ctxNums / srcNums), 4)}</b>.`;
  }
  [Xe, Ye, He].forEach(e => e.addEventListener("input", draw));
  [Ce, Ee].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════ 35 · #tf-svg — teacher forcing against free running ═══════════ */
(function () {
  const svg = d3.select("#tf-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Ee = document.getElementById("tf-e"), Te = document.getElementById("tf-T"),
    Re = document.getElementById("tf-r"), Ve = document.getElementById("tf-v");

  function draw() {
    const e = +Ee.value, T = +Te.value, rec = +Re.value, view = Ve.value;
    document.getElementById("tf-ev").textContent = DL.fmt(e, 3);
    document.getElementById("tf-Tv").textContent = T;
    document.getElementById("tf-rv").textContent = DL.fmt(rec, 2);
    const f = DL.frame(svg, W, H, { l: 54, r: 200, t: 24, b: 40 }), g = f.g;

    /* a two-state Markov chain: on-distribution ↔ drifted */
    const onDist = [1], errs = [0];
    let p = 1, ex = 0;
    for (let t = 1; t <= T; t++) {
      /* P(on next) = P(on)·(1−e) + P(off)·recovery */
      p = p * (1 - e) + (1 - p) * rec;
      ex += (1 - p) + p * e;
      onDist.push(p); errs.push(ex);
    }
    const noRec = [1];
    { let q = 1; for (let t = 1; t <= T; t++) { q = q * (1 - e); noRec.push(q); } }

    if (view === "both") {
      RN.title(g, 0, -8, "the same decoder, two wirings for one set of arrows");
      const step = Math.min(56, (f.iw - 40) / 6), r = 12;
      const draw1 = (oy, lab, fromData) => {
        RN.note(g, 0, oy - 20, lab, fromData ? DC.good : DC.bad, 10.5);
        for (let t = 0; t < 6; t++) {
          const x = 40 + t * step;
          if (t > 0) DL.arrow(g, x - step + r, oy, x - r, oy, { color: DC.a2, w: 1.2, head: 4 });
          g.append("circle").attr("cx", x).attr("cy", oy).attr("r", r).attr("fill", DC.panel2).attr("stroke", DC.a2);
          g.append("text").attr("x", x).attr("y", oy + 3.5).attr("text-anchor", "middle").attr("font-size", 8).attr("fill", DC.ink).text("h" + (t + 1));
          g.append("circle").attr("cx", x).attr("cy", oy - 38).attr("r", r).attr("fill", DC.panel2).attr("stroke", DC.good);
          g.append("text").attr("x", x).attr("y", oy - 34.5).attr("text-anchor", "middle").attr("font-size", 8).attr("fill", DC.muted).text("ŷ" + (t + 1));
          DL.arrow(g, x, oy - r - 1, x, oy - 38 + r + 1, { color: DC.good, w: 1, head: 3 });
          /* the input arrow: from the data row, or from the previous output */
          if (t > 0) {
            if (fromData) {
              g.append("rect").attr("x", x - 12).attr("y", oy + 30).attr("width", 24).attr("height", 16).attr("rx", 3)
                .attr("fill", DC.good).attr("fill-opacity", 0.25).attr("stroke", DC.good);
              g.append("text").attr("x", x).attr("y", oy + 42).attr("text-anchor", "middle").attr("font-size", 8).attr("fill", DC.good).text("y" + t);
              DL.arrow(g, x, oy + 29, x, oy + r + 1, { color: DC.good, w: 1.2, head: 4 });
            } else {
              g.append("path").attr("d", `M${x - step},${oy - 38 - r} C${x - step},${oy - 78} ${x},${oy - 78} ${x},${oy - 38 - r}`)
                .attr("fill", "none").attr("stroke", DC.bad).attr("stroke-width", 1.2);
              DL.arrow(g, x - 1, oy - 60, x, oy - 38 - r - 1, { color: DC.bad, w: 1.2, head: 4 });
            }
          }
        }
        RN.note(g, 0, oy + 4, fromData ? "training" : "inference", DC.muted, 9.5);
      };
      draw1(112, "teacher forcing — the input is the TRUE previous target", true);
      draw1(250, "free running — the input is the model's OWN previous output", false);
    } else {
      const x = d3.scaleLinear().domain([0, T]).range([0, f.iw]);
      const y = d3.scaleLinear().domain([0, 1]).range([f.ih, 0]);
      DL.gridY(g, y, f.iw, 5);
      DL.axisB(g, x, f.ih, 6, "generation length");
      DL.axisL(g, y, 5, "P(still on the training distribution)");
      g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y(0.5)).attr("y2", y(0.5))
        .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
      DL.curve(g, onDist.map((v, i) => [x(i), y(v)]), { stroke: DC.good, w: 2.2 });
      DL.curve(g, noRec.map((v, i) => [x(i), y(v)]), { stroke: DC.bad, w: 2 });
      const half = v => { for (let i = 0; i < v.length; i++) if (v[i] < 0.5) return i; return null; };
      const h1 = half(onDist), h2 = half(noRec);
      [[h1, DC.good], [h2, DC.bad]].forEach(d => {
        if (d[0] !== null) {
          g.append("line").attr("x1", x(d[0])).attr("x2", x(d[0])).attr("y1", 0).attr("y2", f.ih)
            .attr("stroke", d[1]).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.7);
          RN.note(g, x(d[0]) + 4, 14, "half at " + d[0], d[1], 9);
        }
      });
      DL.legend(g, [{ label: "with recovery, probability " + DL.fmt(rec, 2), color: DC.good },
      { label: "no recovery — every error is permanent", color: DC.bad }],
        4, f.ih + 30, { vertical: false, step: 240, font: 9.5 });
    }

    /* the no-recovery half-life has a closed form and always exists; the curve
       WITH recovery has a steady state that may sit above one half, in which
       case it never crosses and we say so instead of printing a made-up length. */
    const halfNo = er => Math.log(0.5) / Math.log(1 - er);
    const steady = rec / Math.max(1e-12, e + rec);
    const halfOf = (er, rc) => { let q = 1; for (let t = 1; t <= 20000; t++) { q = q * (1 - er) + (1 - q) * rc; if (q < 0.5) return t; } return null; };
    const sx = f.iw + 16;
    const kv = DL.kv(g, sx, 6, { keyW: 150, size: 10.5, lead: 15 });
    kv("per-step error rate", DL.fmt(e, 4));
    kv("P(on distribution) at T", DL.fmt(onDist[T], 4), onDist[T] < 0.5 ? DC.bad : DC.good);
    kv("expected errors over T", DL.fmt(errs[T], 2));
    kv("steady state, with recovery", DL.fmt(steady, 4) + (steady > 0.5 ? "  (never crosses ½)" : ""), steady > 0.5 ? DC.good : DC.bad);
    kv("half-length, NO recovery", DL.fmt(halfNo(e), 1) + " steps", DC.bad);
    kv("the same at a tenth the error", DL.fmt(halfNo(e / 10), 1) + " steps", DC.good);
    kv("the same at a hundredth", DL.fmt(halfNo(e / 100), 1) + " steps", DC.good);
    RN.note(g, sx, 116, "the last three rows are the point:", DC.a2, 9.5);
    RN.note(g, sx, 128, "exposure bias is a statement about a", DC.muted, 9.5);
    RN.note(g, sx, 140, "compounding error rate, and driving", DC.muted, 9.5);
    RN.note(g, sx, 152, "that rate down moves the problem", DC.muted, 9.5);
    RN.note(g, sx, 164, "past any length people generate.", DC.muted, 9.5);

    document.getElementById("tf-readout").innerHTML =
      `per-step error rate <b>${DL.fmt(e, 4)}</b>, recovery probability <b>${DL.fmt(rec, 2)}</b> · ` +
      `after ${T} steps the generated prefix is still on the training distribution with probability ` +
      `<b>${DL.fmt(onDist[T], 4)}</b>, with <b>${DL.fmt(errs[T], 2)}</b> errors expected · ` +
      `with recovery the process settles at a steady state of <b>${DL.fmt(steady, 4)}</b>` +
      (steady > 0.5 ? ", which is above one half, so that curve <b>never</b> crosses and no crossing length is reported" :
        `, and it crosses one half at <b>${halfOf(e, rec) === null ? "> 20000" : halfOf(e, rec)}</b> steps`) +
      ` · with NO recovery the crossing is exact and is <b>${DL.fmt(halfNo(e), 1)}</b> steps, rising to ` +
      `<b>${DL.fmt(halfNo(e / 10), 1)}</b> at a tenth of the error rate and <b>${DL.fmt(halfNo(e / 100), 1)}</b> at a hundredth — ` +
      `which is the whole argument that a sufficiently accurate model does not suffer from this over the lengths people generate.`;
  }
  [Ee, Te, Re].forEach(e => e.addEventListener("input", draw));
  Ve.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 36 · #bn-svg — what survives in the context vector ═══════════ */
(function () {
  const svg = d3.select("#bn-svg");
  if (svg.empty()) return;
  const W = 760, H = 410;
  const Le = document.getElementById("bn-L"), He = document.getElementById("bn-dh"),
    Xe = document.getElementById("bn-dx"), Re = document.getElementById("bn-rho"),
    Ee = document.getElementById("bn-e");

  /* EXACT, for the linear encoder: R²(k) = tr(Mₖ Σ⁺ Mₖᵀ)/dₓ with Mₖ = U W^(L−1−k) */
  function exact(dx, dh, rho, L, seed) {
    const r = DL.rng(seed || 12);
    const U = DL.zeros2(dx, dh);
    for (let i = 0; i < dx; i++) for (let j = 0; j < dh; j++) U[i][j] = DL.randn(r) / Math.sqrt(dx);
    const Wm = RN.mat(dh, (seed || 12) + 1, rho);
    const M = [];
    for (let k = 0; k < L; k++) M.push(DL.matmul(U, DL.matPow(Wm, L - 1 - k)));
    const S = DL.zeros2(dh, dh);
    M.forEach(Mk => { const P = DL.matmul(DL.transpose(Mk), Mk); for (let i = 0; i < dh; i++) for (let j = 0; j < dh; j++) S[i][j] += P[i][j]; });
    const Si = DL.pinvSym(S, 1e-12);
    return M.map(Mk => { const P = DL.matmul(DL.matmul(Mk, Si), DL.transpose(Mk)); let t = 0; for (let i = 0; i < dx; i++) t += P[i][i]; return DL.clamp(t / dx, 0, 1); });
  }
  /* MEASURED, through the tanh encoder, by ridge regression */
  function measured(dx, dh, rho, L, n, seed) {
    const net = RN.net("vanilla", dx, dh, 1, { seed: seed || 5, rho: rho, uScale: 1 / Math.sqrt(dx) });
    const r = DL.rng((seed || 5) + 77), H2 = [], Xs = [];
    for (let i = 0; i < n; i++) {
      const X = DL.zeros2(L, dx).map(row => row.map(() => DL.randn(r)));
      H2.push(DL.CELL.vanilla.forward(net.p, X).H[L - 1].concat([1]));
      Xs.push(X);
    }
    const out = [];
    for (let k = 0; k < L; k++) {
      const Y = Xs.map(X => X[k]);
      const P = DL.matmul(H2, DL.ridge(H2, Y, 1e-6));
      let sse = 0, sst = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < dx; j++) { sse += (P[i][j] - Y[i][j]) ** 2; sst += Y[i][j] ** 2; }
      out.push(DL.clamp(1 - sse / Math.max(1e-12, sst), 0, 1));
    }
    return out;
  }

  function draw() {
    const L = +Le.value, dh = +He.value, dx = +Xe.value, rho = +Re.value, enc = Ee.value;
    document.getElementById("bn-Lv").textContent = L;
    document.getElementById("bn-dhv").textContent = dh;
    document.getElementById("bn-dxv").textContent = dx;
    document.getElementById("bn-rhov").textContent = DL.fmt(rho, 2);
    const f = DL.frame(svg, W, H, { l: 50, r: 14, t: 24, b: 44 }), g = f.g;

    const per = enc === "lin" ? exact(dx, dh, rho, L, 12) : measured(dx, dh, rho, L, 1200, 5);
    const mean = d3.mean(per), bound = Math.min(1, dh / (L * dx));

    /* ---- left: per-position profile ---- */
    const gw = 330, gh = 250;
    RN.title(g, 0, -8, "recoverable fraction of each source position");
    const x = d3.scaleLinear().domain([1, Math.max(2, L)]).range([0, gw]);
    const y = d3.scaleLog().domain([Math.max(1e-7, Math.min(...per.filter(v => v > 0), 1) / 3), 1.4]).range([gh, 0]).clamp(true);
    DL.gridY(g, y, gw, 5);
    DL.axisB(g, x, gh, 6, "source position");
    DL.axisL(g, y, 5, "R²", v => DL.fmtE(v, 0));
    DL.curve(g, per.map((v, i) => [x(i + 1), y(Math.max(v, 1e-9))]), { stroke: DC.accent, w: 2.2 });
    const ref = per.map((v, i) => [x(i + 1), y(Math.max(1e-9, per[L - 1] * Math.pow(rho, 2 * (L - 1 - i))))]);
    DL.curve(g, ref, { stroke: DC.muted, w: 1.2, dash: "4 3" });
    RN.note(g, 4, 12, "dashed: ρ² per step back from the end", DC.muted, 9);
    DL.legend(g, [{ label: enc === "lin" ? "exact, linear encoder" : "measured, tanh encoder (ridge)", color: DC.accent }],
      4, gh + 32, { vertical: false, step: 200, font: 9.5 });

    /* ---- right: mean against L, and the rank bound ---- */
    const rx = gw + 60, rw = f.iw - rx;
    RN.title(g, rx, -8, "mean recoverable fraction against source length");
    const gg = g.append("g").attr("transform", `translate(${rx},0)`);
    const Ls = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64];
    const pts = Ls.map(l => ({ l: l, v: d3.mean(enc === "lin" ? exact(dx, dh, rho, l, 12) : measured(dx, dh, rho, l, 500, 5)), b: Math.min(1, dh / (l * dx)) }));
    const xl = d3.scaleLog().domain([2, 64]).range([0, rw - 10]);
    const yl = d3.scaleLog().domain([Math.max(1e-4, Math.min(...pts.map(p => p.v).filter(v => v > 0)) / 2), 1.6]).range([gh, 0]).clamp(true);
    DL.gridY(gg, yl, rw - 10, 4);
    DL.axisB(gg, xl, gh, 4, "source length L", d3.format("d"));
    DL.axisL(gg, yl, 4, "mean R²", v => DL.fmtE(v, 0));
    DL.curve(gg, pts.map(p => [xl(p.l), yl(Math.max(p.b, 1e-9))]), { stroke: DC.a2, w: 1.6, dash: "4 3" });
    DL.curve(gg, pts.map(p => [xl(p.l), yl(Math.max(p.v, 1e-9))]), { stroke: DC.accent, w: 2.2 });
    pts.forEach(p => gg.append("circle").attr("cx", xl(p.l)).attr("cy", yl(Math.max(p.v, 1e-9))).attr("r", 2.4).attr("fill", DC.accent));
    const Lsing = Math.max(2, Math.ceil(dh / dx) + 1);
    if (Lsing >= 2 && Lsing <= 64) {
      gg.append("line").attr("x1", xl(Lsing)).attr("x2", xl(Lsing)).attr("y1", 0).attr("y2", gh)
        .attr("stroke", DC.bad).attr("stroke-dasharray", "3 3");
      RN.note(gg, xl(Lsing) + 4, 14, "L·dₓ > dₕ from here: the map is singular", DC.bad, 9);
    }
    gg.append("line").attr("x1", xl(DL.clamp(L, 2, 64))).attr("x2", xl(DL.clamp(L, 2, 64))).attr("y1", 0).attr("y2", gh)
      .attr("stroke", DC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.6);
    DL.legend(gg, [{ label: "measured mean", color: DC.accent }, { label: "the rank bound dₕ/(L·dₓ)", color: DC.a2, dash: "4 3" }],
      4, gh + 32, { vertical: false, step: 150, font: 9.5 });

    const kv = DL.kv(g, 0, gh + 62, { keyW: 250, size: 10.5, lead: 15 });
    kv("mean R² at L = " + L, DL.fmt(mean, 6), DC.accent);
    kv("the rank bound dₕ/(L·dₓ)", DL.fmt(bound, 6), DC.a2);
    kv("measured / bound", DL.fmt(mean / Math.max(1e-12, bound), 4), enc === "lin" ? DC.good : DC.a2);
    kv("R² at the FIRST position", DL.fmtE(per[0], 3), per[0] < 0.01 ? DC.bad : DC.ink);
    kv("R² at the LAST position", DL.fmt(per[L - 1], 4), DC.good);
    kv("last / first", per[0] > 0 ? DL.fmtE(per[L - 1] / per[0], 2) : "∞", DC.bad);
    kv("L at which the map becomes singular", String(Lsing));

    document.getElementById("bn-readout").innerHTML =
      `${enc === "lin" ? "exact, linear encoder" : "measured through a tanh encoder"} · dₓ = ${dx}, dₕ = ${dh}, L = ${L} · ` +
      `mean recoverable fraction <b>${DL.fmt(mean, 6)}</b> against the rank bound dₕ/(L·dₓ) = <b>${DL.fmt(bound, 6)}</b> ` +
      `(<b>${DL.fmt(100 * mean / Math.max(1e-12, bound), 1)}%</b> of what counting permits) · ` +
      `the LAST source position is <b>${DL.fmt(100 * per[L - 1], 1)}%</b> recoverable and the FIRST is ` +
      `<b>${DL.fmt(100 * per[0], 3)}%</b> — a ratio of <b>${per[0] > 0 ? DL.fmtE(per[L - 1] / per[0], 2) : "∞"}</b> · ` +
      `the map stops being injective at L = <b>${Lsing}</b>.`;
  }
  [Le, He, Xe, Re].forEach(e => e.addEventListener("input", draw));
  Ee.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 37 · #dc-svg — greedy, beam and exhaustive search ═══════════ */
(function () {
  const svg = d3.select("#dc-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const Ve = document.getElementById("dc-V"), Le = document.getElementById("dc-L"),
    Ke = document.getElementById("dc-k"), Te = document.getElementById("dc-t"),
    Ne = document.getElementById("dc-n"), Se = document.getElementById("dc-s");

  function draw() {
    const V = +Ve.value, L = +Le.value, k = +Ke.value, tau = +Te.value, norml = Ne.value === "on", seed = +Se.value;
    document.getElementById("dc-Vv").textContent = V;
    document.getElementById("dc-Lv").textContent = L;
    document.getElementById("dc-kv").textContent = k;
    document.getElementById("dc-tv").textContent = DL.fmt(tau, 1);
    document.getElementById("dc-sv").textContent = seed;
    const f = DL.frame(svg, W, H, { l: 16, r: 220, t: 24, b: 20 }), g = f.g;

    const tab = new Map(), r = DL.rng(seed * 131 + 7);
    const get = pre => {
      const key = pre.join(",");
      if (!tab.has(key)) { const z = []; for (let i = 0; i < V; i++) z.push(DL.randn(r) * tau); tab.set(key, DL.softmax(z)); }
      return tab.get(key);
    };
    /* enumerate every leaf */
    const leaves = [];
    (function rec(pre, lp) {
      if (pre.length === L) { leaves.push({ s: pre.slice(), lp: lp }); return; }
      const p = get(pre);
      for (let i = 0; i < V; i++) rec(pre.concat([i]), lp + Math.log(p[i]));
    })([], 0);
    const bestLeaf = leaves.reduce((a, b) => (b.lp > a.lp ? b : a));
    /* greedy */
    let gp = [], glp = 0;
    for (let t = 0; t < L; t++) { const p = get(gp); let a = 0; for (let i = 1; i < V; i++) if (p[i] > p[a]) a = i; glp += Math.log(p[a]); gp.push(a); }
    /* beam */
    let B = [{ s: [], lp: 0 }];
    const beamTrace = [];
    for (let t = 0; t < L; t++) {
      const C = [];
      B.forEach(b => { const p = get(b.s); for (let i = 0; i < V; i++) C.push({ s: b.s.concat([i]), lp: b.lp + Math.log(p[i]) }); });
      C.sort((a, b) => (norml ? b.lp / b.s.length - a.lp / a.s.length : b.lp - a.lp));
      B = C.slice(0, k);
      beamTrace.push(B.map(b => b.s.join(",")));
    }
    const beamBest = B.reduce((a, b) => ((norml ? b.lp / L > a.lp / L : b.lp > a.lp) ? b : a));

    /* ---- the tree ---- */
    const levels = [];
    for (let t = 0; t <= L; t++) levels.push(Math.pow(V, t));
    const nLeaf = levels[L];
    const lw = f.iw / (L + 0.4), lh = Math.min(300, f.ih - 40);
    const ypos = (t, idx) => 16 + (idx + 0.5) * lh / levels[t];
    const label = i => String.fromCharCode(65 + i);
    const onPath = (pre, path) => pre.every((v, i) => v === path[i]);
    for (let t = 0; t < L; t++) {
      const cnt = levels[t];
      for (let idx = 0; idx < cnt; idx++) {
        const pre = [];
        let rem = idx;
        for (let d = t - 1; d >= 0; d--) { const pw = Math.pow(V, d); pre.unshift(Math.floor(rem / pw)); rem %= pw; }
        const pr = get(pre);
        for (let i = 0; i < V; i++) {
          const cidx = idx * V + i;
          const isG = onPath(pre.concat([i]), gp);
          const isB = onPath(pre.concat([i]), bestLeaf.s);
          const isBm = beamTrace[t] && beamTrace[t].indexOf(pre.concat([i]).join(",")) >= 0;
          g.append("line").attr("x1", 16 + t * lw).attr("y1", ypos(t, idx))
            .attr("x2", 16 + (t + 1) * lw).attr("y2", ypos(t + 1, cidx))
            .attr("stroke", isB ? DC.good : (isG ? DC.bad : (isBm ? DC.a2 : DC.line)))
            .attr("stroke-width", isB ? 2.6 : (isG ? 2 : (isBm ? 1.4 : 0.7)))
            .attr("stroke-opacity", (isB || isG || isBm) ? 0.9 : Math.max(0.12, pr[i]));
          if (L <= 4 && V <= 3 && cnt <= 9)
            g.append("text").attr("x", 16 + (t + 0.5) * lw).attr("y", (ypos(t, idx) + ypos(t + 1, cidx)) / 2 - 2)
              .attr("text-anchor", "middle").attr("font-size", 7.5).attr("fill", DC.muted).text(DL.fmt(pr[i], 2));
        }
      }
    }
    leaves.forEach((lf, i) => {
      const p = Math.exp(lf.lp);
      const isB = lf === bestLeaf, isG = lf.s.join() === gp.join();
      g.append("circle").attr("cx", 16 + L * lw).attr("cy", ypos(L, i)).attr("r", isB ? 4 : 2.6)
        .attr("fill", isB ? DC.good : (isG ? DC.bad : DC.muted)).attr("fill-opacity", 0.9);
      if (nLeaf <= 32) g.append("text").attr("x", 16 + L * lw + 8).attr("y", ypos(L, i) + 3)
        .attr("font-size", 8).attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", isB ? DC.good : (isG ? DC.bad : DC.muted))
        .text(lf.s.map(label).join("") + "  " + DL.fmt(p, 4));
    });

    /* first divergence */
    let div = null;
    for (let t = 0; t < L; t++) if (gp[t] !== bestLeaf.s[t]) { div = t + 1; break; }
    if (div !== null) {
      const idx = (() => { let a = 0; for (let d = 0; d < div - 1; d++) a = a * V + gp[d]; return a; })();
      g.append("circle").attr("cx", 16 + (div - 1) * lw).attr("cy", ypos(div - 1, idx)).attr("r", 9)
        .attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 2);
    }

    const sx = f.iw + 16;
    const pg = Math.exp(glp), pb = Math.exp(bestLeaf.lp), pbm = Math.exp(beamBest.lp);
    RN.title(g, sx, -4, "what each method returned");
    [["greedy", gp, pg, DC.bad], ["beam " + k, beamBest.s, pbm, DC.a2], ["exhaustive", bestLeaf.s, pb, DC.good]].forEach((d, i) => {
      const yy = 12 + i * 40;
      RN.note(g, sx, yy, d[0], d[3], 10);
      g.append("text").attr("x", sx + 66).attr("y", yy).attr("font-size", 11)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.ink).text(d[1].map(label).join(""));
      g.append("rect").attr("x", sx).attr("y", yy + 6).attr("width", Math.max(1, 150 * d[2] / pb)).attr("height", 11)
        .attr("fill", d[3]).attr("fill-opacity", 0.7);
      g.append("text").attr("x", sx + 156).attr("y", yy + 15).attr("font-size", 9)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", DC.muted)
        .text(DL.fmt(d[2], 5) + "  (" + DL.fmt(100 * d[2] / pb, 1) + "%)");
    });
    const kv = DL.kv(g, sx, 148, { keyW: 158, size: 10.5, lead: 15 });
    kv("greedy = the true argmax?", pg >= pb - 1e-12 ? "YES" : "no", pg >= pb - 1e-12 ? DC.good : DC.bad);
    kv("beam " + k + " = the true argmax?", pbm >= pb - 1e-12 ? "YES" : "no", pbm >= pb - 1e-12 ? DC.good : DC.bad);
    kv("greedy first diverges at step", div === null ? "—" : String(div), DC.a2);
    kv("log-probability greedy loses", DL.fmt(bestLeaf.lp - glp, 4));
    kv("candidates scored, greedy", DL.commas(L * V));
    kv("candidates scored, beam " + k, DL.commas(L * k * V));
    kv("candidates scored, exhaustive", DL.commas(nLeaf));
    kv("length normalisation", norml ? "on" : "off", norml ? DC.a2 : DC.muted);

    document.getElementById("dc-readout").innerHTML =
      `|V| = ${V}, L = ${L}, ${nLeaf} complete sequences · greedy returned <b>${gp.map(label).join("")}</b> with probability ` +
      `<b>${DL.fmt(pg, 5)}</b>; the true argmax is <b>${bestLeaf.s.map(label).join("")}</b> with <b>${DL.fmt(pb, 5)}</b> ` +
      `(<b>${DL.fmt(100 * pg / pb, 1)}%</b> of the optimum) · beam ${k} returned <b>${beamBest.s.map(label).join("")}</b> ` +
      `at <b>${DL.fmt(100 * pbm / pb, 1)}%</b> · ` +
      (div === null ? "greedy found the optimum here." : `greedy first diverged at step <b>${div}</b>, taking the locally better token.`) +
      ` Candidates scored: greedy <b>${DL.commas(L * V)}</b>, beam <b>${DL.commas(L * k * V)}</b>, exhaustive <b>${DL.commas(nLeaf)}</b>.`;
  }
  [Ve, Le, Ke, Te, Se].forEach(e => e.addEventListener("input", draw));
  Ne.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 38 · #tw-svg — chain against tree ═══════════ */
(function () {
  const svg = d3.select("#tw-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Te = document.getElementById("tw-T"), Se = document.getElementById("tw-s"),
    Ge = document.getElementById("tw-g"), Ll = document.getElementById("tw-l");

  /* build a tree over [0, n): returns nodes with depth per leaf */
  function build(n, style, r) {
    let id = 0;
    const nodes = [];
    function rec(lo, hi, depth) {
      const me = { id: id++, lo, hi, depth, leaf: hi - lo === 1, kids: [] };
      nodes.push(me);
      if (me.leaf) return me;
      let mid;
      if (style === "left") mid = hi - 1;
      else if (style === "parse") mid = lo + 1 + Math.floor(r() * (hi - lo - 1));
      else mid = lo + Math.ceil((hi - lo) / 2);
      mid = Math.max(lo + 1, Math.min(hi - 1, mid));
      me.kids = [rec(lo, mid, depth + 1), rec(mid, hi, depth + 1)];
      return me;
    }
    const root = rec(0, n, 0);
    const leafDepth = new Array(n).fill(0);
    nodes.filter(nd => nd.leaf).forEach(nd => leafDepth[nd.lo] = nd.depth);
    return { root, nodes, leafDepth };
  }

  function draw() {
    const T = +Te.value;
    Ll.max = String(T);
    const style = Se.value, gain = +Ge.value, sel = Math.min(+Ll.value, T);
    document.getElementById("tw-Tv").textContent = T;
    document.getElementById("tw-gv").textContent = DL.fmt(gain, 2);
    document.getElementById("tw-lv").textContent = sel;
    const f = DL.frame(svg, W, H, { l: 50, r: 190, t: 24, b: 40 }), g = f.g;

    const tr = build(T, style, DL.rng(303));
    const maxD = Math.max(...tr.leafDepth);
    const chainDepth = i => T - i;                 // leaf i to the end of the chain

    /* ---- the tree ---- */
    const th = 150, tw2 = f.iw;
    RN.title(g, 0, -8, "the tree (top) and the equivalent chain (below it)");
    const lx = i => (i + 0.5) * tw2 / T;
    const ly = d => 14 + d * (th - 24) / Math.max(1, maxD);
    const pathTo = [];
    (function mark(nd) {
      if (nd.leaf) return nd.lo === sel - 1;
      const hit = nd.kids.some(k => mark(k));
      if (hit) pathTo.push(nd);
      return hit;
    })(tr.root);
    tr.nodes.forEach(nd => {
      if (nd.leaf) return;
      const cx = (lx(nd.lo) + lx(nd.hi - 1)) / 2;
      nd.kids.forEach(k => {
        const kx = k.leaf ? lx(k.lo) : (lx(k.lo) + lx(k.hi - 1)) / 2;
        const on = pathTo.indexOf(nd) >= 0 && (k.leaf ? k.lo === sel - 1 : pathTo.indexOf(k) >= 0);
        g.append("line").attr("x1", cx).attr("y1", ly(nd.depth)).attr("x2", kx).attr("y2", ly(k.depth))
          .attr("stroke", on ? DC.violet : DC.line).attr("stroke-width", on ? 2.6 : 0.9).attr("stroke-opacity", on ? 0.95 : 0.5);
      });
      g.append("circle").attr("cx", cx).attr("cy", ly(nd.depth)).attr("r", pathTo.indexOf(nd) >= 0 ? 3.6 : 2.2)
        .attr("fill", pathTo.indexOf(nd) >= 0 ? DC.violet : DC.muted);
    });
    for (let i = 0; i < T; i++) {
      g.append("circle").attr("cx", lx(i)).attr("cy", ly(tr.leafDepth[i])).attr("r", i === sel - 1 ? 4.4 : 2.6)
        .attr("fill", i === sel - 1 ? DC.a2 : DC.accent);
    }
    const cy = th + 26;
    for (let i = 0; i < T; i++) {
      if (i > 0) g.append("line").attr("x1", lx(i - 1)).attr("x2", lx(i)).attr("y1", cy).attr("y2", cy)
        .attr("stroke", i - 1 >= sel - 1 ? DC.bad : DC.line).attr("stroke-width", i - 1 >= sel - 1 ? 2.4 : 0.9)
        .attr("stroke-opacity", i - 1 >= sel - 1 ? 0.9 : 0.5);
      g.append("circle").attr("cx", lx(i)).attr("cy", cy).attr("r", i === sel - 1 ? 4.4 : 2.4)
        .attr("fill", i === sel - 1 ? DC.a2 : DC.bad).attr("fill-opacity", 0.85);
    }
    RN.note(g, 0, cy + 18, "the chain: every leaf is as deep as its distance from the end", DC.bad, 9.5);

    /* ---- surviving gradient per leaf ---- */
    const by = cy + 40, bh = f.ih - by - 6;
    RN.title(g, 0, by - 8, "gradient surviving the journey to the root, per leaf");
    const gg = g.append("g").attr("transform", `translate(0,${by})`);
    const treeV = tr.leafDepth.map(d => Math.pow(gain, d));
    const chainV = d3.range(T).map(i => Math.pow(gain, chainDepth(i)));
    const x = d3.scaleBand().domain(d3.range(T)).range([0, f.iw]).padding(0.15);
    const allv = treeV.concat(chainV).filter(v => v > 0);
    const y = d3.scaleLog().domain([Math.min(...allv) / 3, Math.max(...allv) * 2]).range([bh, 0]).clamp(true);
    DL.gridY(gg, y, f.iw, 4);
    DL.axisB(gg, d3.scaleLinear().domain([1, T]).range([0, f.iw]), bh, 6, "leaf");
    DL.axisL(gg, y, 4, "survives", v => DL.fmtE(v, 0));
    for (let i = 0; i < T; i++) {
      gg.append("rect").attr("x", x(i)).attr("y", y(treeV[i])).attr("width", x.bandwidth() / 2)
        .attr("height", Math.max(0, bh - y(treeV[i]))).attr("fill", DC.violet).attr("fill-opacity", 0.7);
      gg.append("rect").attr("x", x(i) + x.bandwidth() / 2).attr("y", y(chainV[i])).attr("width", x.bandwidth() / 2)
        .attr("height", Math.max(0, bh - y(chainV[i]))).attr("fill", DC.bad).attr("fill-opacity", 0.7);
    }
    DL.legend(gg, [{ label: "tree", color: DC.violet }, { label: "chain", color: DC.bad }],
      4, bh + 30, { vertical: false, step: 70, font: 9.5 });

    const worstT = Math.min(...treeV), worstC = Math.min(...chainV);
    const sx = f.iw + 16;
    const kv = DL.kv(g, sx, 6, { keyW: 140, size: 10.5, lead: 16 });
    kv("leaf " + sel + ": tree depth", String(tr.leafDepth[sel - 1]), DC.violet);
    kv("leaf " + sel + ": chain depth", String(chainDepth(sel - 1)), DC.bad);
    kv("tree survives", DL.fmtE(treeV[sel - 1], 3), DC.violet);
    kv("chain survives", DL.fmtE(chainV[sel - 1], 3), DC.bad);
    kv("ratio", DL.fmtE(treeV[sel - 1] / Math.max(1e-300, chainV[sel - 1]), 2), DC.good);
    kv("mean tree depth", DL.fmt(d3.mean(tr.leafDepth), 2));
    kv("max tree depth", String(maxD) + "   (log₂T = " + DL.fmt(Math.log2(T), 2) + ")");
    kv("worst leaf: tree / chain", DL.fmtE(worstT, 2) + " / " + DL.fmtE(worstC, 2));
    const rp = 2 * 8 * 8 + 8, cp = 2 * 8 + 8 * 8 + 8;
    kv("cell parameters at dₕ = 8", DL.commas(rp) + " (tree) vs " + DL.commas(cp) + " (chain)", DC.muted);
    let bad = null;
    for (let t = 4; t <= 100000; t++) if (Math.pow(gain, t) < 1e-6) { bad = t; break; }
    kv("chain worst leaf below 10⁻⁶ at T =", bad === null ? "> 100000" : String(bad), DC.bad);

    document.getElementById("tw-readout").innerHTML =
      `${Se.options[Se.selectedIndex].text}, T = ${T}, per-composition factor ${DL.fmt(gain, 2)} · ` +
      `leaf ${sel} is <b>${tr.leafDepth[sel - 1]}</b> compositions from the root in the tree and <b>${chainDepth(sel - 1)}</b> ` +
      `in the chain, so <b>${DL.fmtE(treeV[sel - 1], 3)}</b> survives against <b>${DL.fmtE(chainV[sel - 1], 3)}</b> — ` +
      `a ratio of <b>${DL.fmtE(treeV[sel - 1] / Math.max(1e-300, chainV[sel - 1]), 2)}</b> · ` +
      `mean tree depth <b>${DL.fmt(d3.mean(tr.leafDepth), 2)}</b> against log₂T = ${DL.fmt(Math.log2(T), 2)}` +
      `${style === "left" ? " — a fully left-branching tree IS the chain, and the two bars coincide" : ""}.`;
  }
  [Te, Ge, Ll].forEach(e => e.addEventListener("input", draw));
  Se.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 39 · #nw-svg — recurrent against attention ═══════════ */
(function () {
  const svg = d3.select("#nw-svg");
  if (svg.empty()) return;
  const W = 760, H = 400;
  const Te = document.getElementById("nw-T"), De = document.getElementById("nw-d"),
    Le = document.getElementById("nw-L"), Ce = document.getElementById("nw-c");

  /* every one of these is shape arithmetic, stated once and used everywhere */
  const trainRNN = (T, d, L) => T * 4 * (d * d + d * d) * L;
  const trainATT = (T, d, L) => (4 * T * d * d + 2 * T * T * d + 8 * T * d * d) * L;
  const trainLIN = (T, d, L) => T * 4 * (d * d + d * d) * L;
  const inferRNN = (T, d, L) => 4 * (d * d + d * d) * L;
  const inferATT = (T, d, L) => (12 * d * d + 2 * T * d) * L;
  const memRNN = (T, d, L) => 2 * d * L * 2;
  const memATT = (T, d, L) => 2 * T * d * L * 2;
  const depRNN = (T) => T, depATT = () => 1, depLIN = T => Math.max(1, Math.ceil(Math.log2(T)));

  function draw() {
    const T = +Te.value, d = +De.value, L = +Le.value, cmp = Ce.value;
    document.getElementById("nw-Tv").textContent = T;
    document.getElementById("nw-dv").textContent = d;
    document.getElementById("nw-Lv").textContent = L;
    const f = DL.frame(svg, W, H, { l: 62, r: 214, t: 24, b: 40 }), g = f.g;

    const Ts = d3.range(0, 49).map(i => Math.round(16 * Math.pow(16384 / 16, i / 48)));
    const fns = {
      train: [["recurrent (LSTM)", t => trainRNN(t, d, L), RN.CELLC.lstm],
      ["attention + FFN", t => trainATT(t, d, L), RN.CELLC.attn],
      ["linear recurrence (scan)", t => trainLIN(t, d, L), RN.CELLC.linrec]],
      infer: [["recurrent (LSTM)", t => inferRNN(t, d, L), RN.CELLC.lstm],
      ["attention + KV cache", t => inferATT(t, d, L), RN.CELLC.attn],
      ["linear recurrence", t => inferRNN(t, d, L), RN.CELLC.linrec]],
      mem: [["recurrent state", t => memRNN(t, d, L), RN.CELLC.lstm],
      ["key-value cache", t => memATT(t, d, L), RN.CELLC.attn],
      ["linear-recurrence state", t => memRNN(t, d, L), RN.CELLC.linrec]],
      depth: [["recurrent", t => depRNN(t) * L, RN.CELLC.lstm],
      ["attention", t => depATT(t) * L, RN.CELLC.attn],
      ["linear recurrence (scan)", t => depLIN(t) * L, RN.CELLC.linrec]]
    }[cmp];
    const unit = { train: "MACs per sequence", infer: "MACs per generated token", mem: "bytes of state", depth: "dependent products" }[cmp];

    const x = d3.scaleLog().domain([16, 16384]).range([0, f.iw]);
    const allv = fns.reduce((a, s) => a.concat(Ts.map(s[1])), []).filter(v => v > 0);
    const y = d3.scaleLog().domain([Math.min(...allv) / 2, Math.max(...allv) * 2]).range([f.ih, 0]);
    DL.gridY(g, y, f.iw, 5);
    DL.axisB(g, x, f.ih, 5, "sequence length T", d3.format("~s"));
    DL.axisL(g, y, 5, unit, v => DL.big(v));
    fns.forEach(s => DL.curve(g, Ts.map(t => [x(t), y(Math.max(s[1](t), 1e-9))]), { stroke: s[2], w: 2 }));
    g.append("line").attr("x1", x(DL.clamp(T, 16, 16384))).attr("x2", x(DL.clamp(T, 16, 16384)))
      .attr("y1", 0).attr("y2", f.ih).attr("stroke", DC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.6);
    fns.forEach(s => {
      const v = s[1](T);
      if (v > 0) g.append("circle").attr("cx", x(DL.clamp(T, 16, 16384))).attr("cy", y(v)).attr("r", 3.4).attr("fill", s[2]);
    });
    DL.legend(g, fns.map(s => ({ label: s[0], color: s[2] })), 4, f.ih + 30, { vertical: false, step: 176, font: 9 });

    /* crossovers */
    const cross = (a, b) => {
      for (let i = 0; i + 1 < Ts.length; i++) {
        const d1 = a(Ts[i]) - b(Ts[i]), d2 = a(Ts[i + 1]) - b(Ts[i + 1]);
        if (d1 * d2 < 0) return Ts[i + 1];
      }
      return null;
    };
    const cRA = cross(fns[0][1], fns[1][1]);
    const sx = f.iw + 16;
    const kv = DL.kv(g, sx, 6, { keyW: 148, size: 10.5, lead: 16 });
    fns.forEach(s => kv(s[0], DL.big(s[1](T)) + (cmp === "mem" ? " B" : ""), s[2]));
    kv("attention / recurrent", DL.fmt(fns[1][1](T) / Math.max(1e-9, fns[0][1](T)), 3) + "×", DC.a2);
    kv("crossover length", cRA === null ? "none in range" : DL.commas(cRA));
    if (cmp === "train") kv("attention's quadratic part", DL.big(2 * T * T * d * L) + " (" + DL.fmt(100 * 2 * T * T * d / Math.max(1, 12 * T * d * d + 2 * T * T * d), 1) + "%)");
    if (cmp === "train") kv("where quadratic overtakes linear", DL.commas(6 * d) + "  (= 6d)", DC.a2);
    const verdict = {
      train: fns[1][1](T) < fns[0][1](T) * 3 ? "attention: same order of arithmetic, and it parallelises" : "recurrence is cheaper in arithmetic — but still serial",
      infer: "recurrence: constant cost per token, whatever the history",
      mem: "recurrence: constant state; the cache grows without bound",
      depth: "attention: one product per layer against " + DL.commas(T) + " for recurrence"
    }[cmp];
    RN.note(g, sx, 6 + 6 * 16 + 26, "verdict at this operating point:", DC.muted, 9.5);
    RN.note(g, sx, 6 + 6 * 16 + 38, verdict.slice(0, 34), DC.ink, 9.5);
    RN.note(g, sx, 6 + 6 * 16 + 50, verdict.slice(34, 68), DC.ink, 9.5);
    RN.note(g, sx, 6 + 6 * 16 + 62, verdict.slice(68), DC.ink, 9.5);

    document.getElementById("nw-readout").innerHTML =
      `${Ce.options[Ce.selectedIndex].text}, T = ${DL.commas(T)}, d = ${d}, ${L} layers · ` +
      fns.map(s => `${s[0]} <b>${DL.big(s[1](T))}${cmp === "mem" ? " B" : ""}</b>`).join(" · ") +
      ` · attention costs <b>${DL.fmt(fns[1][1](T) / Math.max(1e-9, fns[0][1](T)), 3)}×</b> the recurrent model here` +
      (cRA === null ? "" : `, and the two curves cross at T ≈ <b>${DL.commas(cRA)}</b>`) +
      (cmp === "train" ? ` · attention's quadratic term overtakes its own linear terms at T = 6d = <b>${DL.commas(6 * d)}</b>.` : ".");
  }
  [Te, De, Le].forEach(e => e.addEventListener("input", draw));
  Ce.addEventListener("change", draw);
  draw();
})();

/* ═══════════ 40 · #dg-svg — the four diagnostics ═══════════ */
(function () {
  const svg = d3.select("#dg-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const Fe = document.getElementById("dg-f"), Ce = document.getElementById("dg-c"),
    Te = document.getElementById("dg-T"), Ne = document.getElementById("dg-n");
  const DETECT = { none: null, explode: 1, overclip: 1, vanish: 2, trunc: 2, bf0: 3 };
  const CACHE = {};

  function run(fault, cell, T, steps) {
    const key = fault + "|" + cell + "|" + T + "|" + steps;
    if (CACHE[key]) return CACHE[key];
    const K = 3, dx = K + 1, dh = 10;
    const rho = fault === "explode" ? 1.6 : (fault === "vanish" ? 0.5 : 1.0);
    const bf = fault === "bf0" ? 0 : 1;
    const clipAt = fault === "explode" ? 1e9 : (fault === "overclip" ? 0.01 : 1.0);
    const trunc = fault === "trunc" ? 3 : null;
    const net = RN.net(cell, dx, dh, K, { seed: 606, out: "softmax", uScale: 0.3,
      rho: cell === "vanilla" ? rho : null, wScale: cell === "vanilla" ? null : 0.3, forgetBias: bf });
    const pk = RN.packer(net), o = DL.OPT.adam, hp = Object.assign({}, o.hp, { lr: 0.03 });
    const stt = o.init(pk.n);
    let v = pk.get();
    const r = DL.rng(717), loss = [], norms = [], spec = [];
    const dhAcc = new Array(T).fill(0);
    let nAcc = 0;
    for (let s = 0; s < steps; s++) {
      const B = RN.memBatch("hold", T, K, 5, r);
      const G = new Array(pk.n).fill(0);
      let L2 = 0;
      for (const b of B) {
        const gr = trunc ? DL.seqBPTTtrunc(net, b.X, b.Y, trunc) : DL.seqBPTT(net, b.X, b.Y);
        const gf = pk.grad(gr);
        for (let i = 0; i < G.length; i++) G[i] += gf[i] / B.length;
        L2 += gr.loss / B.length;
        if (s % 8 === 0) {
          /* per-step gradient at the states, for panel 3. The truncated forms
             do not carry a dH field, so take it from the head directly. */
          const stt2 = gr.st;
          const hd2 = DL.seqHead(net, stt2, b.Y).dH;
          let dn = DL.zeros(dh);
          for (let tt = T - 1; tt >= 0; tt--) {
            const dht = hd2[tt].map((val, j) => val + dn[j]);
            if (trunc && tt < T - trunc) { for (let j = 0; j < dh; j++) dht[j] = hd2[tt][j]; }
            dhAcc[tt] += RN.norm(dht);
            if (cell === "vanilla") {
              const gz = dht.map((val, j) => val * (1 - stt2.H[tt][j] * stt2.H[tt][j]));
              dn = DL.vecmat(gz, DL.transpose(net.p.W));
            } else dn = DL.zeros(dh);
          }
          nAcc++;
        }
      }
      const cl = DL.clipNorm(G, clipAt);
      const dvec = o.step(stt, cl.g, hp);
      for (let i = 0; i < v.length; i++) v[i] += dvec[i];
      pk.set(v);
      loss.push(L2); norms.push(cl.norm);
      if (s % Math.max(1, Math.round(steps / 40)) === 0)
        spec.push([s, cell === "vanilla" ? DL.specRad(net.p.W) : d3.mean(DL.CELL.lstm.forward(net.p, RN.memBatch("hold", T, K, 1, DL.rng(3))[0].X).F.map(a => d3.mean(a)))]);
    }
    /* a diverged run produces non-finite values; record WHERE and keep the
       arrays plottable rather than letting a NaN destroy a scale domain */
    let nanAt = null;
    for (let i = 0; i < loss.length; i++) if (!isFinite(loss[i]) || !isFinite(norms[i])) { nanAt = i; break; }
    const res = { loss: loss.map(v => (isFinite(v) ? v : NaN)), norms: norms.map(v => (isFinite(v) ? v : NaN)),
      spec: spec.filter(p2 => isFinite(p2[1])), perStep: dhAcc.map(x => x / Math.max(1, nAcc)),
      clipAt, trunc, cell, nanAt };
    CACHE[key] = res;
    return res;
  }

  function draw() {
    const fault = Fe.value, cell = Ce.value, T = +Te.value, steps = +Ne.value;
    document.getElementById("dg-Tv").textContent = T;
    document.getElementById("dg-nv").textContent = steps;
    const f = DL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 14 }), g = f.g;
    const R = run(fault, cell, T, steps);
    const det = DETECT[fault];

    const pw = (f.iw - 30) / 2, ph = (f.ih - 44) / 2;
    /* a log scale that cannot be handed an empty or degenerate domain */
    const slog = (vals, h, extra) => {
      const pos = vals.concat(extra || []).filter(v => v > 0 && isFinite(v));
      const lo = pos.length ? Math.min(...pos) : 1e-6, hi = pos.length ? Math.max(...pos) : 1;
      return d3.scaleLog().domain([Math.max(1e-14, lo / 3), Math.max(hi * 3, lo * 10)]).range([h, 0]).clamp(true);
    };
    const panel = (idx, ox, oy, title2, fn) => {
      const on = det === idx;
      g.append("rect").attr("x", ox - 6).attr("y", oy - 16).attr("width", pw + 12).attr("height", ph + 30)
        .attr("rx", 6).attr("fill", "none").attr("stroke", on ? DC.a2 : DC.line).attr("stroke-width", on ? 2 : 0.8);
      RN.title(g, ox, oy - 4, title2, on ? DC.a2 : DC.ink);
      fn(g.append("g").attr("transform", `translate(${ox},${oy + 8})`));
    };
    panel(0, 14, 30, "1 · the training loss", gg => {
      const x = d3.scaleLinear().domain([0, R.loss.length]).range([0, pw - 40]);
      const y = slog(R.loss, ph - 20);
      DL.gridY(gg, y, pw - 40, 3); DL.axisB(gg, x, ph - 20, 3, ""); DL.axisL(gg, y, 3, "", v => DL.fmtE(v, 0));
      const lp = R.loss.map((v, i) => [x(i), y(Math.max(v, 1e-9))]).filter(pt => isFinite(pt[1]));
      if (lp.length > 1) DL.curve(gg, lp, { stroke: DC.accent, w: 1.6 });
      if (R.nanAt !== null) {
        gg.append("line").attr("x1", x(R.nanAt)).attr("x2", x(R.nanAt)).attr("y1", 0).attr("y2", ph - 20)
          .attr("stroke", DC.bad).attr("stroke-width", 2);
        RN.note(gg, x(R.nanAt) + 4, 12, "NaN at step " + R.nanAt, DC.bad, 9);
      }
    });
    panel(1, 14 + pw + 30, 30, "2 · the PRE-CLIP gradient norm", gg => {
      const x = d3.scaleLinear().domain([0, R.norms.length]).range([0, pw - 40]);
      const y = slog(R.norms, ph - 20, R.clipAt < 1e8 ? [R.clipAt] : []);
      DL.gridY(gg, y, pw - 40, 3); DL.axisB(gg, x, ph - 20, 3, ""); DL.axisL(gg, y, 3, "", v => DL.fmtE(v, 0));
      if (R.clipAt < 1e8) {
        gg.append("line").attr("x1", 0).attr("x2", pw - 40).attr("y1", y(R.clipAt)).attr("y2", y(R.clipAt))
          .attr("stroke", DC.bad).attr("stroke-dasharray", "4 3");
        RN.note(gg, 2, y(R.clipAt) - 3, "clip", DC.bad, 8.5);
      }
      const np = R.norms.map((v, i) => [x(i), y(Math.max(v, 1e-12))]).filter(pt => isFinite(pt[1]));
      if (np.length > 1) DL.curve(gg, np, { stroke: DC.a2, w: 1.4 });
      R.norms.forEach((v, i) => { if (isFinite(v) && v > R.clipAt) gg.append("circle").attr("cx", x(i)).attr("cy", y(v)).attr("r", 1.8).attr("fill", DC.bad); });
      if (R.nanAt !== null) {
        gg.append("line").attr("x1", x(R.nanAt)).attr("x2", x(R.nanAt)).attr("y1", 0).attr("y2", ph - 20)
          .attr("stroke", DC.bad).attr("stroke-width", 2);
      }
    });
    panel(2, 14, 30 + ph + 40, "3 · gradient at each time step", gg => {
      const x = d3.scaleLinear().domain([1, T]).range([0, pw - 40]);
      const y = slog(R.perStep, ph - 20);
      DL.gridY(gg, y, pw - 40, 3); DL.axisB(gg, x, ph - 20, 4, "step"); DL.axisL(gg, y, 3, "", v => DL.fmtE(v, 0));
      const bw2 = (pw - 40) / T;
      R.perStep.forEach((v, i) => gg.append("rect").attr("x", x(i + 1) - bw2 / 2).attr("y", y(Math.max(v, 1e-300)))
        .attr("width", Math.max(1, bw2 - 1)).attr("height", Math.max(0, ph - 20 - y(Math.max(v, 1e-300))))
        .attr("fill", R.trunc && i < T - R.trunc ? DC.bad : DC.violet).attr("fill-opacity", 0.7));
      if (R.trunc) {
        gg.append("line").attr("x1", x(T - R.trunc + 1)).attr("x2", x(T - R.trunc + 1)).attr("y1", 0).attr("y2", ph - 20)
          .attr("stroke", DC.bad).attr("stroke-dasharray", "3 3");
        RN.note(gg, 2, 10, "truncation boundary", DC.bad, 8.5);
      }
    });
    panel(3, 14 + pw + 30, 30 + ph + 40, cell === "vanilla" ? "4 · ρ(W) across training" : "4 · mean forget gate across training", gg => {
      const x = d3.scaleLinear().domain([0, steps]).range([0, pw - 40]);
      const vals = R.spec.map(p2 => p2[1]).filter(isFinite);
      const y = d3.scaleLinear().domain([Math.min(0, ...vals.concat([0])) * 0.9, Math.max(1.2, ...vals.concat([1.2])) * 1.1]).range([ph - 20, 0]);
      DL.gridY(gg, y, pw - 40, 3); DL.axisB(gg, x, ph - 20, 3, "step"); DL.axisL(gg, y, 3, "");
      gg.append("line").attr("x1", 0).attr("x2", pw - 40).attr("y1", y(1)).attr("y2", y(1))
        .attr("stroke", DC.muted).attr("stroke-dasharray", "3 3");
      const sp = R.spec.map(p2 => [x(p2[0]), y(p2[1])]).filter(pt => isFinite(pt[0]) && isFinite(pt[1]));
      if (sp.length > 1) DL.curve(gg, sp, { stroke: DC.good, w: 1.8 });
    });

    const names = { none: "none — a healthy run", explode: "recurrent scale too large, clipping off",
      overclip: "clip threshold far too low", vanish: "recurrent scale too small",
      trunc: "truncation window far too short", bf0: "forget-gate bias left at zero" };
    const panelName = ["the loss curve", "the pre-clip gradient norm", "the per-step gradient profile", "the tracked initialisation quantity"];
    const fixes = { none: "—", explode: "turn on global-norm clipping at 1.0",
      overclip: "raise the threshold above the median pre-clip norm",
      vanish: "initialise ρ(W) at 1.0–1.2, or use a gated cell",
      trunc: "widen the window, or use per-loss rather than chunked truncation",
      bf0: "set the forget-gate bias to 1" };
    const fin = R.norms.filter(v => v > 0 && isFinite(v));
    const med = fin.length ? d3.median(fin) : 0;
    const shown = det === null ? "—" : panelName[det];
    const val = det === null ? "—" : det === 1 ? ("median pre-clip norm " + RN.P(med, 3) + " against a threshold of " + (R.clipAt > 1e8 ? "∞" : DL.fmt(R.clipAt, 3)))
      : det === 2 ? ("first-step gradient " + DL.fmtE(R.perStep[0], 2) + " against last-step " + DL.fmtE(R.perStep[T - 1], 2))
        : (R.spec.length === 0 ? "the run diverged before any value could be read"
          : (cell === "vanilla" ? "ρ(W) = " + DL.fmt(R.spec[0][1], 4) : "mean forget gate = " + DL.fmt(R.spec[0][1], 4)));

    document.getElementById("dg-readout").innerHTML =
      `fault: <b>${names[fault]}</b> · detected by <b>${shown}</b> ` +
      `${det === null ? "(nothing to detect)" : "(panel " + (det + 1) + ", outlined)"} · ${val} · ` +
      (R.nanAt === null ? `final loss <b>${RN.P(R.loss[R.loss.length - 1], 4)}</b>` :
        `<b>the run went to NaN at step ${R.nanAt}</b>`) +
      `, largest finite pre-clip gradient norm <b>${fin.length ? RN.P(Math.max(...fin), 3) : "—"}</b>, ` +
      `median <b>${fin.length ? RN.P(med, 3) : "—"}</b> · fix: ${fixes[fault]}.`;
  }
  [Te, Ne].forEach(e => e.addEventListener("input", draw));
  [Fe, Ce].forEach(e => e.addEventListener("change", draw));
  draw();
})();
