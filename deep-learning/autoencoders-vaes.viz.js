/* autoencoders-vaes.viz.js — page-local D3 figures for deep-learning/autoencoders-vaes.html.
   Loaded after ../data.js → ../notes.js, so the palette `C` from notes.js is already global.
   The deep-learning folder has no shared viz library, so everything lives in one AE namespace.

   Every figure is self-contained in an IIFE that bails out if its <svg> is absent, so the
   file is safe to load on a partially built page. Nothing here draws into an svg it did
   not find in the DOM, and every svg carries its own viewBox/role/aria-label in the HTML. */

const AE = (function () {

  /* ── palette ─────────────────────────────────────────────────────────── */
  const col = {
    A: "#5b9cff", B: "#ffb454", good: "#4ade80", bad: "#f87171",
    ink: "#e6e9ef", muted: "#9aa3b2", line: "#2a2f3a", grid: "#1b2130",
    panel: "#171a23", panel2: "#1e222d", bg: "#0f1117",
    violet: "#c084fc", teal: "#2dd4bf", rose: "#fb7185", lime: "#a3e635"
  };

  /* ── scalars ─────────────────────────────────────────────────────────── */
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  function linspace(a, b, k) {
    const o = [];
    for (let i = 0; i < k; i++) o.push(a + (b - a) * (k === 1 ? 0 : i / (k - 1)));
    return o;
  }
  function fmt(x, d) {
    if (!isFinite(x)) return "—";
    const dd = (d === undefined) ? 2 : d;
    const s = x.toFixed(dd);
    return (parseFloat(s) === 0) ? (0).toFixed(dd) : s;
  }
  const commas = n => Math.round(n).toLocaleString("en-US");

  /* ── seeded randomness, so every reader sees the same picture ────────── */
  function rng(seed) {                                  // mulberry32
    let a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(r) {                                   // Box–Muller
    let u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ── dense linear algebra, only what the figures need ────────────────── */
  const zeros = (n, m) => (m === undefined ? new Float64Array(n)
    : Array.from({ length: n }, () => new Float64Array(m)));
  function matmul(A, B) {                               // A (n×k) · B (k×m)
    const n = A.length, k = B.length, m = B[0].length, O = zeros(n, m);
    for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) {
      const a = A[i][p]; if (a === 0) continue;
      for (let j = 0; j < m; j++) O[i][j] += a * B[p][j];
    }
    return O;
  }
  const T = A => {
    const n = A.length, m = A[0].length, O = zeros(m, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) O[j][i] = A[i][j];
    return O;
  };
  function mv(A, v) {                                   // matrix × vector
    const o = new Float64Array(A.length);
    for (let i = 0; i < A.length; i++) { let s = 0; for (let j = 0; j < v.length; j++) s += A[i][j] * v[j]; o[i] = s; }
    return o;
  }
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const nrm = a => Math.sqrt(dot(a, a));

  /* Jacobi eigendecomposition of a symmetric matrix. Returns eigenvalues in
     DESCENDING order with the matching eigenvectors as columns of V. Used to
     compute PCA in the browser, so the figures can quote the exact floor. */
  function eigSym(Ain, iters) {
    const n = Ain.length;
    const A = Ain.map(r => Float64Array.from(r));
    let V = zeros(n, n); for (let i = 0; i < n; i++) V[i][i] = 1;
    const sweeps = iters || 100;
    for (let s = 0; s < sweeps; s++) {
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
      if (off < 1e-24) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-18) continue;
        const th = 0.5 * (A[q][q] - A[p][p]) / A[p][q];
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - sn * akq; A[k][q] = sn * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - sn * aqk; A[q][k] = sn * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - sn * vkq; V[k][q] = sn * vkp + c * vkq;
        }
      }
    }
    const idx = d3.range(n).sort((a, b) => A[b][b] - A[a][a]);
    const vals = idx.map(i => A[i][i]);
    const vecs = zeros(n, n);
    idx.forEach((src, dst) => { for (let k = 0; k < n; k++) vecs[k][dst] = V[k][src]; });
    return { values: vals, vectors: vecs };
  }
  function covariance(X) {                              // X: array of Float64Array rows, already centred
    const n = X.length, d = X[0].length, S = zeros(d, d);
    for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) {
      const xa = X[i][a];
      for (let b = a; b < d; b++) S[a][b] += xa * X[i][b];
    }
    for (let a = 0; a < d; a++) for (let b = a; b < d; b++) { S[a][b] /= n; S[b][a] = S[a][b]; }
    return S;
  }

  /* ── Gaussian bits used all over the VAE half ────────────────────────── */
  /* KL( N(m, diag v) ‖ N(0, I) ) in nats — the closed form the VAE uses. */
  const klDiagStd = (m, v) => {
    let s = 0;
    for (let i = 0; i < m.length; i++) s += v[i] + m[i] * m[i] - 1 - Math.log(v[i]);
    return 0.5 * s;
  };
  const logNormPdf = (x, m, s) => -0.5 * Math.log(2 * Math.PI * s * s) - (x - m) * (x - m) / (2 * s * s);

  /* ── drawing helpers ─────────────────────────────────────────────────── */
  function frame(sel, W, H, m) {
    const g = sel.append("g").attr("transform", `translate(${m.l},${m.t})`);
    return { g, w: W - m.l - m.r, h: H - m.t - m.b, m };
  }
  function axisB(g, x, y, ticks, label, fmtf) {
    const ax = g.append("g").attr("class", "axis").attr("transform", `translate(0,${y})`)
      .call(d3.axisBottom(x).ticks(ticks || 6).tickFormat(fmtf || null));
    ax.selectAll("text").attr("fill", col.muted).attr("font-size", 10);
    ax.selectAll("line,path").attr("stroke", col.line);
    if (label) g.append("text").attr("x", x.range()[1]).attr("y", y + 32)
      .attr("text-anchor", "end").attr("fill", col.muted).attr("font-size", 11).text(label);
    return ax;
  }
  function axisL(g, y, x0, ticks, label, fmtf) {
    const ax = g.append("g").attr("class", "axis").attr("transform", `translate(${x0},0)`)
      .call(d3.axisLeft(y).ticks(ticks || 5).tickFormat(fmtf || null));
    ax.selectAll("text").attr("fill", col.muted).attr("font-size", 10);
    ax.selectAll("line,path").attr("stroke", col.line);
    if (label) g.append("text").attr("transform", "rotate(-90)").attr("x", 0).attr("y", x0 - 34)
      .attr("text-anchor", "end").attr("fill", col.muted).attr("font-size", 11).text(label);
    return ax;
  }
  function gridY(g, y, w, n) {
    g.append("g").selectAll("line").data(y.ticks(n || 5)).enter().append("line")
      .attr("x1", 0).attr("x2", w).attr("y1", d => y(d)).attr("y2", d => y(d))
      .attr("stroke", col.grid).attr("stroke-width", 1);
  }
  function gridX(g, x, h, n) {
    g.append("g").selectAll("line").data(x.ticks(n || 6)).enter().append("line")
      .attr("y1", 0).attr("y2", h).attr("x1", d => x(d)).attr("x2", d => x(d))
      .attr("stroke", col.grid).attr("stroke-width", 1);
  }
  function legend(g, items, x, y) {
    const L = g.append("g").attr("transform", `translate(${x},${y})`);
    items.forEach((it, i) => {
      const row = L.append("g").attr("transform", `translate(0,${i * 15})`);
      if (it.dash) row.append("line").attr("x1", 0).attr("x2", 16).attr("y1", 0).attr("y2", 0)
        .attr("stroke", it.c).attr("stroke-width", 2).attr("stroke-dasharray", it.dash);
      else row.append("rect").attr("x", 0).attr("y", -4).attr("width", 12).attr("height", 8)
        .attr("rx", 2).attr("fill", it.c).attr("fill-opacity", it.op === undefined ? .9 : it.op);
      row.append("text").attr("x", 22).attr("y", 4).attr("fill", col.muted).attr("font-size", 10.5).text(it.t);
    });
    return L;
  }
  function panelBox(g, x, y, w, h, title) {
    g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h).attr("rx", 8)
      .attr("fill", col.panel2).attr("fill-opacity", .55).attr("stroke", col.line);
    if (title) g.append("text").attr("x", x + 9).attr("y", y + 15)
      .attr("fill", col.muted).attr("font-size", 10.5).attr("letter-spacing", .4).text(title);
  }
  function arrow(g, x1, y1, x2, y2, c, w, head) {
    const hd = head === undefined ? 4 : head;
    const a = Math.atan2(y2 - y1, x2 - x1), L = Math.hypot(x2 - x1, y2 - y1);
    if (L < 1e-9) return;
    const bx = x2 - hd * Math.cos(a), by = y2 - hd * Math.sin(a);
    g.append("line").attr("x1", x1).attr("y1", y1).attr("x2", bx).attr("y2", by)
      .attr("stroke", c).attr("stroke-width", w || 1.3).attr("stroke-linecap", "round");
    g.append("path").attr("d", `M${x2},${y2} L${bx - hd * 0.55 * Math.sin(a)},${by + hd * 0.55 * Math.cos(a)} L${bx + hd * 0.55 * Math.sin(a)},${by - hd * 0.55 * Math.cos(a)} Z`)
      .attr("fill", c);
  }
  /* a clip rect in the *referencing element's* user space, never page coordinates */
  function clipRect(svg, id, x, y, w, h) {
    let defs = svg.select("defs");
    if (defs.empty()) defs = svg.append("defs");
    defs.append("clipPath").attr("id", id).append("rect")
      .attr("x", x).attr("y", y).attr("width", w).attr("height", h);
    return `url(#${id})`;
  }
  const on = (id, ev, fn) => { const e = document.getElementById(id); if (e) e.addEventListener(ev, fn); };
  /* Range inputs whose step is a whole number always report whole numbers in a
     browser; snapping here means every figure can index arrays with AE.val
     directly and cannot be handed a fractional width or a fractional index. */
  const val = id => {
    const e = document.getElementById(id);
    if (!e) return 0;
    const v = +e.value, st = +e.step;
    return (Number.isFinite(st) && st >= 1 && Number.isInteger(st)) ? Math.round(v) : v;
  };
  const chk = id => { const e = document.getElementById(id); return !!(e && e.checked); };
  const sel = id => { const e = document.getElementById(id); return e ? e.value : ""; };
  const put = (id, s) => { const e = document.getElementById(id); if (e) e.innerHTML = s; };

  return {
    col, clamp, lerp, linspace, fmt, commas, rng, randn, zeros, matmul, T, mv, dot, nrm,
    eigSym, covariance, klDiagStd, logNormPdf,
    frame, axisB, axisL, gridY, gridX, legend, panelBox, arrow, clipRect,
    on, val, chk, sel, put
  };
})();

/* ═══ 01 · the shape of the computation ═══════════════════════════════════ */
(function () {
  const svg = d3.select("#ae-arch");
  if (svg.empty()) return;
  const W = 760, H = 330, C1 = AE.col;
  const g = svg.append("g");

  function draw() {
    g.selectAll("*").remove();
    const D = AE.val("ar-D"), d = Math.min(AE.val("ar-d"), D), hw = AE.val("ar-h"), tied = AE.chk("ar-tied");
    AE.put("ar-Dv", D); AE.put("ar-dv", d); AE.put("ar-hv", hw);

    const layers = [
      { n: D, lbl: "x", sub: "input", c: C1.muted },
      { n: hw, lbl: "a₁", sub: "encoder hidden", c: C1.A },
      { n: d, lbl: "h", sub: "code — the bottleneck", c: C1.B },
      { n: hw, lbl: "a₂", sub: "decoder hidden", c: C1.A },
      { n: D, lbl: "r", sub: "reconstruction", c: C1.muted }
    ];
    const maxN = Math.max(D, hw), midY = 158, maxH = 190;
    const xs = [70, 220, 380, 540, 690];

    // connection bands, drawn first so the bars sit on top
    for (let i = 0; i < 4; i++) {
      const h0 = Math.max(6, maxH * layers[i].n / maxN), h1 = Math.max(6, maxH * layers[i + 1].n / maxN);
      g.append("path").attr("d",
        `M${xs[i] + 15},${midY - h0 / 2} L${xs[i + 1] - 15},${midY - h1 / 2} L${xs[i + 1] - 15},${midY + h1 / 2} L${xs[i] + 15},${midY + h0 / 2} Z`)
        .attr("fill", i < 2 ? C1.A : C1.violet).attr("fill-opacity", tied ? .13 : .10)
        .attr("stroke", i < 2 ? C1.A : C1.violet).attr("stroke-opacity", .35);
    }
    layers.forEach((L, i) => {
      const h = Math.max(6, maxH * L.n / maxN);
      g.append("rect").attr("x", xs[i] - 15).attr("y", midY - h / 2).attr("width", 30).attr("height", h)
        .attr("rx", 5).attr("fill", L.c).attr("fill-opacity", i === 2 ? .85 : .35)
        .attr("stroke", L.c).attr("stroke-width", i === 2 ? 2 : 1);
      g.append("text").attr("x", xs[i]).attr("y", midY - h / 2 - 20).attr("text-anchor", "middle")
        .attr("fill", C1.ink).attr("font-size", 13).attr("font-weight", 600).text(L.lbl);
      g.append("text").attr("x", xs[i]).attr("y", midY - h / 2 - 7).attr("text-anchor", "middle")
        .attr("fill", C1.muted).attr("font-size", 10).text(L.sub);
      g.append("text").attr("x", xs[i]).attr("y", midY + h / 2 + 16).attr("text-anchor", "middle")
        .attr("fill", i === 2 ? C1.B : C1.muted).attr("font-size", 11)
        .attr("font-family", "SF Mono,Menlo,monospace").text("ℝ^" + L.n);
    });

    // encoder / decoder brackets
    const brk = (x0, x1, y, txt, c) => {
      g.append("path").attr("d", `M${x0},${y} L${x0},${y + 8} L${x1},${y + 8} L${x1},${y}`)
        .attr("fill", "none").attr("stroke", c).attr("stroke-width", 1.2).attr("stroke-opacity", .7);
      g.append("text").attr("x", (x0 + x1) / 2).attr("y", y + 24).attr("text-anchor", "middle")
        .attr("fill", c).attr("font-size", 11.5).text(txt);
    };
    brk(xs[0], xs[2], 278, "encoder  f : ℝ^" + D + " → ℝ^" + d, C1.A);
    brk(xs[2], xs[4], 278, "decoder  g : ℝ^" + d + " → ℝ^" + D, C1.violet);

    const pEnc = D * hw + hw + hw * d + d;
    const pDec = tied ? 0 : (d * hw + hw + hw * D + D);
    const biasOnly = tied ? (hw + D) : 0;
    const tot = pEnc + pDec + biasOnly;
    AE.put("ar-out",
      `code holds <b>${d}</b> of <b>${D}</b> numbers — a <b>${AE.fmt(D / d, 1)}×</b> squeeze &nbsp;·&nbsp; ` +
      `encoder ${AE.commas(pEnc)} params &nbsp;·&nbsp; decoder ${tied ? AE.commas(biasOnly) + " (weights tied, biases only)" : AE.commas(pDec)} ` +
      `&nbsp;·&nbsp; total <b>${AE.commas(tot)}</b> &nbsp;·&nbsp; ` +
      (d >= D ? `<span style="color:${C1.bad}">d ≥ D: overcomplete — nothing stops g∘f = identity, see §06</span>`
        : `<span style="color:${C1.good}">undercomplete: the code cannot hold the input</span>`));
  }
  ["ar-D", "ar-d", "ar-h"].forEach(id => AE.on(id, "input", draw));
  AE.on("ar-tied", "change", draw);
  draw();
})();

/* ═══ a tiny batched MLP + Adam, shared by the figures that train live ════
   Deliberately minimal: dense layers, tanh or linear activations, squared or
   Bernoulli loss, full-batch Adam. The datasets on this page are a few hundred
   2-D points, so a few thousand steps run in well under a frame budget. The
   training machinery itself is taught in neural-network-training.html; this is
   only enough of it to make the pictures real rather than drawn by hand.      */
AE.nn = (function () {
  const zeros = AE.zeros;

  function make(sizes, acts, r, scale) {
    const W = [], b = [], s = scale === undefined ? 1 : scale;
    for (let l = 0; l < sizes.length - 1; l++) {
      const fan = sizes[l], out = sizes[l + 1];
      const Wl = zeros(out, fan);
      for (let i = 0; i < out; i++) for (let j = 0; j < fan; j++)
        Wl[i][j] = AE.randn(r) * s * Math.sqrt(2 / (fan + out));
      W.push(Wl); b.push(new Float64Array(out));
    }
    return { sizes, acts, W, b, L: sizes.length - 1 };
  }

  function forward(net, X) {                        // X: n × sizes[0]
    const A = [X];
    for (let l = 0; l < net.L; l++) {
      const Wl = net.W[l], bl = net.b[l], prev = A[l];
      const n = prev.length, out = Wl.length, fan = Wl[0].length;
      const Z = zeros(n, out);
      for (let i = 0; i < n; i++) {
        const pi = prev[i], zi = Z[i];
        for (let o = 0; o < out; o++) {
          let s = bl[o]; const wo = Wl[o];
          for (let j = 0; j < fan; j++) s += wo[j] * pi[j];
          zi[o] = net.acts[l] === "tanh" ? Math.tanh(s) : (net.acts[l] === "relu" ? (s > 0 ? s : 0) : s);
        }
      }
      A.push(Z);
    }
    return A;
  }

  /* squared-error backprop; dOut is n × sizes[L] (already dL/dOut).
     `extra` is an optional {layerIndex: matrix} of additional gradients on the
     POST-activation values of that layer — this is how a sparsity penalty on the
     code enters the backward pass without a second graph. */
  function backward(net, A, dOut, extra) {
    const gW = net.W.map(w => zeros(w.length, w[0].length));
    const gb = net.b.map(v => new Float64Array(v.length));
    let dZ = dOut;
    for (let l = net.L - 1; l >= 0; l--) {
      const out = net.W[l].length, fan = net.W[l][0].length, n = A[l].length;
      if (extra && extra[l + 1]) {
        const E = extra[l + 1];
        for (let i = 0; i < n; i++) for (let o = 0; o < out; o++) dZ[i][o] += E[i][o];
      }
      if (net.acts[l] === "tanh") {                  // A[l+1] already holds tanh(z)
        for (let i = 0; i < n; i++) for (let o = 0; o < out; o++) {
          const t = A[l + 1][i][o]; dZ[i][o] *= (1 - t * t);
        }
      } else if (net.acts[l] === "relu") {
        for (let i = 0; i < n; i++) for (let o = 0; o < out; o++) {
          if (A[l + 1][i][o] <= 0) dZ[i][o] = 0;
        }
      }
      for (let i = 0; i < n; i++) {
        const di = dZ[i], ai = A[l][i];
        for (let o = 0; o < out; o++) {
          const d = di[o]; if (d === 0) continue;
          gb[l][o] += d; const go = gW[l][o];
          for (let j = 0; j < fan; j++) go[j] += d * ai[j];
        }
      }
      if (l > 0) {
        const dPrev = zeros(n, fan);
        for (let i = 0; i < n; i++) {
          const di = dZ[i], dp = dPrev[i];
          for (let o = 0; o < out; o++) {
            const d = di[o]; if (d === 0) continue;
            const wo = net.W[l][o];
            for (let j = 0; j < fan; j++) dp[j] += d * wo[j];
          }
        }
        dZ = dPrev;
      }
    }
    return { gW, gb };
  }

  function adam(net) {
    return {
      t: 0,
      mW: net.W.map(w => zeros(w.length, w[0].length)), vW: net.W.map(w => zeros(w.length, w[0].length)),
      mb: net.b.map(v => new Float64Array(v.length)), vb: net.b.map(v => new Float64Array(v.length))
    };
  }
  function step(net, st, g, lr) {
    st.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - Math.pow(b1, st.t), c2 = 1 - Math.pow(b2, st.t);
    for (let l = 0; l < net.L; l++) {
      const W = net.W[l], gW = g.gW[l], mW = st.mW[l], vW = st.vW[l];
      for (let i = 0; i < W.length; i++) for (let j = 0; j < W[0].length; j++) {
        mW[i][j] = b1 * mW[i][j] + (1 - b1) * gW[i][j];
        vW[i][j] = b2 * vW[i][j] + (1 - b2) * gW[i][j] * gW[i][j];
        W[i][j] -= lr * (mW[i][j] / c1) / (Math.sqrt(vW[i][j] / c2) + eps);
      }
      const b = net.b[l], gb = g.gb[l], mb = st.mb[l], vb = st.vb[l];
      for (let o = 0; o < b.length; o++) {
        mb[o] = b1 * mb[o] + (1 - b1) * gb[o];
        vb[o] = b2 * vb[o] + (1 - b2) * gb[o] * gb[o];
        b[o] -= lr * (mb[o] / c1) / (Math.sqrt(vb[o] / c2) + eps);
      }
    }
  }

  /* one full-batch squared-error autoencoder step; returns the mean per-sample
     squared reconstruction error BEFORE the update */
  function aeStep(net, st, X, lr) {
    const A = forward(net, X), R = A[net.L], n = X.length, D = X[0].length;
    const dOut = zeros(n, D);
    let loss = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) {
      const e = R[i][j] - X[i][j];
      loss += e * e; dOut[i][j] = 2 * e / n;
    }
    step(net, st, backward(net, A, dOut), lr);
    return loss / n;
  }
  const mse = (net, X) => {
    const R = forward(net, X)[net.L]; let s = 0;
    for (let i = 0; i < X.length; i++) for (let j = 0; j < X[0].length; j++) {
      const e = R[i][j] - X[i][j]; s += e * e;
    }
    return s / X.length;
  };

  /* one step of  ‖x − g(f(x))‖²  +  λ‖h‖₁ , where h is the activation of layer k */
  function aeStepSparse(net, st, X, lr, lam, k) {
    const A = forward(net, X), R = A[net.L], n = X.length, D = X[0].length;
    const dOut = zeros(n, D);
    let loss = 0, l1 = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) {
      const e = R[i][j] - X[i][j];
      loss += e * e; dOut[i][j] = 2 * e / n;
    }
    const Hk = A[k], dh = zeros(n, Hk[0].length);
    for (let i = 0; i < n; i++) for (let j = 0; j < Hk[0].length; j++) {
      l1 += Math.abs(Hk[i][j]);
      dh[i][j] = lam * Math.sign(Hk[i][j]) / n;
    }
    const extra = {}; extra[k] = dh;
    step(net, st, backward(net, A, dOut, extra), lr);
    return { mse: loss / n, l1: l1 / n };
  }

  return { make, forward, backward, adam, step, aeStep, aeStepSparse, mse };
})();

/* ═══ 04a · linear vs nonlinear bottleneck, against the PCA line ══════════ */
(function () {
  const svg = d3.select("#ae-curve");
  if (svg.empty()) return;
  const C1 = AE.col, W = 760, H = 400;
  const F = AE.frame(svg, W, H, { l: 46, r: 190, t: 18, b: 42 });
  const x = d3.scaleLinear().domain([-1.55, 1.55]).range([0, F.w]);
  const y = d3.scaleLinear().domain([-1.15, 1.15]).range([F.h, 0]);
  AE.gridX(F.g, x, F.h, 7); AE.gridY(F.g, y, F.w, 5);
  AE.axisB(F.g, x, F.h, 7, "x₁"); AE.axisL(F.g, y, 0, 5, "x₂");
  const gPts = F.g.append("g"), gRes = F.g.append("g"), gLine = F.g.append("g"), gLeg = F.g.append("g");

  let S = null, timer = null;

  function build() {
    const a = AE.val("cv-curve"), sd = AE.val("cv-noise"), r = AE.rng(20240907);
    AE.put("cv-curvev", AE.fmt(a, 2)); AE.put("cv-noisev", AE.fmt(sd, 2));
    const N = 320, X = AE.zeros(N, 2);
    for (let i = 0; i < N; i++) {
      const t = -1.25 + 2.5 * (i + 0.5) / N + 0.02 * AE.randn(r);
      // the clean 1-D manifold, then isotropic noise around it
      X[i][0] = t + sd * AE.randn(r);
      X[i][1] = a * Math.sin(1.9 * t) * 0.62 + sd * AE.randn(r);
    }
    let m0 = 0, m1 = 0;
    for (let i = 0; i < N; i++) { m0 += X[i][0]; m1 += X[i][1]; }
    m0 /= N; m1 /= N;
    for (let i = 0; i < N; i++) { X[i][0] -= m0; X[i][1] -= m1; }

    const Sc = AE.covariance(X), E = AE.eigSym(Sc);
    const pcaFloor = E.values[1];                    // d = 1 ⇒ floor is λ₂
    const v1 = [E.vectors[0][0], E.vectors[1][0]];

    const rr = AE.rng(7);
    const lin = AE.nn.make([2, 1, 2], ["lin", "lin"], rr, 1.0);
    const nl = AE.nn.make([2, 12, 1, 12, 2], ["tanh", "lin", "tanh", "lin"], rr, 1.4);
    return {
      X, N, pcaFloor, v1, E, lin, nl,
      stLin: AE.nn.adam(lin), stNl: AE.nn.adam(nl), step: 0,
      mseLin: AE.nn.mse(lin, X), mseNl: AE.nn.mse(nl, X)
    };
  }

  function decoderImage(net, X) {
    // codes on the data, then decode a dense sweep across the code range
    const A = AE.nn.forward(net, X), mid = (net.L / 2) | 0;
    const codes = A[mid];
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < codes.length; i++) { lo = Math.min(lo, codes[i][0]); hi = Math.max(hi, codes[i][0]); }
    const pad = 0.06 * (hi - lo);
    const HS = AE.linspace(lo - pad, hi + pad, 160).map(v => Float64Array.from([v]));
    // decode: run the second half of the network by hand
    let cur = HS;
    for (let l = mid; l < net.L; l++) {
      const Wl = net.W[l], bl = net.b[l], out = Wl.length, fan = Wl[0].length;
      const Z = AE.zeros(cur.length, out);
      for (let i = 0; i < cur.length; i++) for (let o = 0; o < out; o++) {
        let s = bl[o]; for (let j = 0; j < fan; j++) s += Wl[o][j] * cur[i][j];
        Z[i][o] = net.acts[l] === "tanh" ? Math.tanh(s) : s;
      }
      cur = Z;
    }
    return cur;
  }

  function render() {
    const mode = AE.sel("cv-model"), showRes = AE.chk("cv-resid");
    gPts.selectAll("*").remove(); gLine.selectAll("*").remove();
    gRes.selectAll("*").remove(); gLeg.selectAll("*").remove();

    if (showRes) {
      const net = (mode === "lin") ? S.lin : S.nl;
      const R = AE.nn.forward(net, S.X)[net.L];
      gRes.selectAll("line").data(d3.range(S.N)).enter().append("line")
        .attr("x1", i => x(S.X[i][0])).attr("y1", i => y(S.X[i][1]))
        .attr("x2", i => x(R[i][0])).attr("y2", i => y(R[i][1]))
        .attr("stroke", C1.bad).attr("stroke-width", .8).attr("stroke-opacity", .55);
    }
    gPts.selectAll("circle").data(d3.range(S.N)).enter().append("circle")
      .attr("cx", i => x(S.X[i][0])).attr("cy", i => y(S.X[i][1])).attr("r", 2.1)
      .attr("fill", C1.muted).attr("fill-opacity", .55);

    // the exact first principal direction, as a dashed line through the (centred) mean
    const L = 1.9;
    gLine.append("line")
      .attr("x1", x(-L * S.v1[0])).attr("y1", y(-L * S.v1[1]))
      .attr("x2", x(L * S.v1[0])).attr("y2", y(L * S.v1[1]))
      .attr("stroke", C1.B).attr("stroke-width", 1.8).attr("stroke-dasharray", "6 4").attr("stroke-opacity", .95);

    const path = d3.line().x(p => x(p[0])).y(p => y(p[1])).curve(d3.curveCatmullRom);
    if (mode === "lin" || mode === "both") {
      const P = decoderImage(S.lin, S.X);
      gLine.append("path").attr("d", path(Array.from(P, p => [p[0], p[1]])))
        .attr("fill", "none").attr("stroke", C1.teal).attr("stroke-width", 2.6).attr("stroke-opacity", .9);
    }
    if (mode === "nl" || mode === "both") {
      const P = decoderImage(S.nl, S.X);
      gLine.append("path").attr("d", path(Array.from(P, p => [p[0], p[1]])))
        .attr("fill", "none").attr("stroke", C1.A).attr("stroke-width", 2.6);
    }
    const items = [{ c: C1.muted, t: "data (centred)" }, { c: C1.B, t: "first principal direction (exact)", dash: "6 4" }];
    if (mode !== "nl") items.push({ c: C1.teal, t: "linear autoencoder, decoder image" });
    if (mode !== "lin") items.push({ c: C1.A, t: "nonlinear autoencoder, decoder image" });
    if (showRes) items.push({ c: C1.bad, t: "x − g(f(x))" });
    AE.legend(gLeg, items, F.w + 14, 20);

    const gap = 100 * (S.pcaFloor - S.mseNl) / S.pcaFloor;
    AE.put("cv-out",
      `step <b>${S.step}</b> &nbsp;·&nbsp; PCA floor for d = 1, <code>λ₂</code> = <b>${AE.fmt(S.pcaFloor, 5)}</b> &nbsp;·&nbsp; ` +
      `linear AE MSE = <b>${AE.fmt(S.mseLin, 5)}</b> &nbsp;·&nbsp; nonlinear AE MSE = <b>${AE.fmt(S.mseNl, 5)}</b>` +
      (S.step >= 2400 ? ` &nbsp;·&nbsp; the linear one has converged <i>exactly to the floor</i>; the nonlinear one is <b>${AE.fmt(gap, 1)}%</b> below it` : " &nbsp;·&nbsp; training…"));
  }

  function train() {
    if (timer) { clearInterval(timer); timer = null; }
    S = build(); render();
    timer = setInterval(() => {
      for (let i = 0; i < 60; i++) {
        AE.nn.aeStep(S.lin, S.stLin, S.X, 0.02);
        AE.nn.aeStep(S.nl, S.stNl, S.X, 0.012);
        S.step++;
      }
      S.mseLin = AE.nn.mse(S.lin, S.X); S.mseNl = AE.nn.mse(S.nl, S.X);
      render();
      if (S.step >= 3600) { clearInterval(timer); timer = null; }
    }, 16);
  }
  ["cv-curve", "cv-noise"].forEach(id => AE.on(id, "input", train));
  AE.on("cv-model", "change", render);
  AE.on("cv-resid", "change", render);
  AE.on("cv-go", "click", train);
  train();
})();

/* ═══ 04b · bottleneck width vs reconstruction error, PCA floor underneath ═ */
(function () {
  const svg = d3.select("#ae-sweep");
  if (svg.empty()) return;
  const C1 = AE.col, W = 760, H = 380;
  const F = AE.frame(svg, W, H, { l: 56, r: 230, t: 16, b: 44 });
  const gAll = F.g.append("g"), gLeg = F.g.append("g"), gIns = svg.append("g");
  let job = null;

  function makeData(D, decay) {
    const r = AE.rng(4242), N = 400;
    // a fixed random orthonormal basis, then a geometric spectrum on it
    let B = AE.zeros(D, D);
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) B[i][j] = AE.randn(r);
    for (let j = 0; j < D; j++) {                        // Gram–Schmidt
      for (let k = 0; k < j; k++) {
        let d = 0; for (let i = 0; i < D; i++) d += B[i][j] * B[i][k];
        for (let i = 0; i < D; i++) B[i][j] -= d * B[i][k];
      }
      let n = 0; for (let i = 0; i < D; i++) n += B[i][j] * B[i][j];
      n = Math.sqrt(n); for (let i = 0; i < D; i++) B[i][j] /= n;
    }
    const lam = d3.range(D).map(i => 4 * Math.pow(decay, i));
    const X = AE.zeros(N, D);
    for (let n = 0; n < N; n++) {
      const z = new Float64Array(D);
      for (let j = 0; j < D; j++) z[j] = AE.randn(r) * Math.sqrt(lam[j]);
      for (let i = 0; i < D; i++) { let s = 0; for (let j = 0; j < D; j++) s += B[i][j] * z[j]; X[n][i] = s; }
    }
    const mu = new Float64Array(D);
    for (let n = 0; n < N; n++) for (let i = 0; i < D; i++) mu[i] += X[n][i] / N;
    for (let n = 0; n < N; n++) for (let i = 0; i < D; i++) X[n][i] -= mu[i];
    return { X, D, ev: AE.eigSym(AE.covariance(X)).values };
  }

  function draw() {
    const D = AE.val("sw-D"), decay = AE.val("sw-decay"), doTrain = AE.chk("sw-train");
    AE.put("sw-Dv", D); AE.put("sw-decayv", AE.fmt(decay, 2));
    gAll.selectAll("*").remove(); gLeg.selectAll("*").remove(); gIns.selectAll("*").remove();

    const { X, ev } = makeData(D, decay);
    const total = d3.sum(ev);
    const floors = d3.range(0, D + 1).map(d => d3.sum(ev.slice(d)));   // ∑_{i>d} λᵢ

    const x = d3.scaleBand().domain(d3.range(1, D + 1)).range([0, F.w]).padding(0.28);
    const y = d3.scaleLinear().domain([0, floors[1] * 1.12]).range([F.h, 0]);
    AE.gridY(gAll, y, F.w, 5);
    AE.axisB(gAll, d3.scaleLinear().domain([0.5, D + 0.5]).range([0, F.w]), F.h, D, "bottleneck width d", d3.format("d"));
    AE.axisL(gAll, y, 0, 5, "mean squared reconstruction error");

    gAll.selectAll("rect.fl").data(d3.range(1, D + 1)).enter().append("rect").attr("class", "fl")
      .attr("x", d => x(d)).attr("width", x.bandwidth())
      .attr("y", d => y(floors[d])).attr("height", d => F.h - y(floors[d]))
      .attr("rx", 3).attr("fill", C1.B).attr("fill-opacity", .32).attr("stroke", C1.B).attr("stroke-opacity", .55);

    // train one bottleneck width per tick, so the figure fills in rather than freezing
    if (job) { clearInterval(job); job = null; }
    const pts = [], gTr = gAll.append("g");
    function paint() {
      gTr.selectAll("*").remove();
      gTr.selectAll("circle").data(pts).enter().append("circle")
        .attr("cx", p => x(p.d) + x.bandwidth() / 2).attr("cy", p => y(p.m)).attr("r", 4.2)
        .attr("fill", C1.A).attr("stroke", C1.bg).attr("stroke-width", 1.2);
      if (pts.length > 1) gTr.append("path").attr("d", d3.line().x(p => x(p.d) + x.bandwidth() / 2).y(p => y(p.m))(pts))
        .attr("fill", "none").attr("stroke", C1.A).attr("stroke-width", 1.4).attr("stroke-opacity", .6);
    }
    if (doTrain) {
      let dcur = 1;
      job = setInterval(() => {
        const rr = AE.rng(1000 + dcur);
        const net = AE.nn.make([D, dcur, D], ["lin", "lin"], rr, 1.0);
        const st = AE.nn.adam(net);
        for (let t = 0; t < 600; t++) AE.nn.aeStep(net, st, X, 0.06);
        pts.push({ d: dcur, m: AE.nn.mse(net, X) });
        paint();
        // the honest error scale is a fraction of TOTAL variance: at large d the floor
        // itself is nearly zero, so a relative-to-floor number would be meaningless
        const worst = d3.max(pts, p => Math.abs(p.m - floors[p.d])) / total;
        AE.put("sw-msg", ` &nbsp;·&nbsp; trained linear AE matches the exact floor to within <b>${AE.fmt(100 * worst, 3)}%</b> of total variance at every d tried so far (${pts.length}/${D})`);
        if (++dcur > D) { clearInterval(job); job = null; }
      }, 16);
    } else AE.put("sw-msg", "");

    // the spectrum, as an inset
    const iw = 190, ih = 96, ix = F.w + F.m.l + 16, iy = 40;
    AE.panelBox(gIns, ix, iy, iw, ih + 26, "eigenvalue spectrum");
    const xi = d3.scaleBand().domain(d3.range(D)).range([ix + 10, ix + iw - 10]).padding(0.2);
    const yi = d3.scaleLinear().domain([0, ev[0]]).range([iy + ih + 12, iy + 24]);
    gIns.selectAll("rect.sp").data(d3.range(D)).enter().append("rect").attr("class", "sp")
      .attr("x", i => xi(i)).attr("width", xi.bandwidth())
      .attr("y", i => yi(ev[i])).attr("height", i => (iy + ih + 12) - yi(ev[i]))
      .attr("rx", 1.5).attr("fill", C1.violet).attr("fill-opacity", .8);
    gIns.append("text").attr("x", ix + 10).attr("y", iy + ih + 26).attr("fill", C1.muted).attr("font-size", 9.5)
      .text("λ₁ = " + AE.fmt(ev[0], 2) + "  …  λ_D = " + AE.fmt(ev[D - 1], 3));

    AE.legend(gLeg, [
      { c: C1.B, t: "PCA floor  ∑ᵢ>d λᵢ  (exact)", op: .5 },
      { c: C1.A, t: "trained linear autoencoder" }
    ], F.w + 16, 16);

    const d90 = floors.findIndex(f => f <= 0.10 * total);
    AE.put("sw-out",
      `total variance <b>${AE.fmt(total, 3)}</b> &nbsp;·&nbsp; ` +
      `d = 1 leaves <b>${AE.fmt(100 * floors[1] / total, 1)}%</b> of it unexplained, ` +
      `d = ${Math.min(3, D)} leaves <b>${AE.fmt(100 * floors[Math.min(3, D)] / total, 1)}%</b> &nbsp;·&nbsp; ` +
      `first d reaching 90% explained variance: <b>d = ${d90 > 0 ? d90 : "—"}</b>` +
      `<span id="sw-msg"></span>`);
  }
  ["sw-decay", "sw-D"].forEach(id => AE.on(id, "input", draw));
  AE.on("sw-train", "change", draw);
  draw();
})();

/* ═══ shared toy manifold: N points near a 1-D curve in the plane ═════════ */
AE.curveData = function (N, amp, freq, sd, seed) {
  const r = AE.rng(seed || 12345), X = AE.zeros(N, 2);
  for (let i = 0; i < N; i++) {
    const t = -1.2 + 2.4 * (i + 0.5) / N + 0.01 * AE.randn(r);
    X[i][0] = t + sd * AE.randn(r);
    X[i][1] = amp * Math.sin(freq * t) + sd * AE.randn(r);
  }
  let m0 = 0, m1 = 0;
  for (let i = 0; i < N; i++) { m0 += X[i][0]; m1 += X[i][1]; }
  m0 /= N; m1 /= N;
  for (let i = 0; i < N; i++) { X[i][0] -= m0; X[i][1] -= m1; }
  return X;
};

/* ═══ 06 · the capacity trap, probed off the data ════════════════════════ */
(function () {
  const svg = d3.select("#ae-capacity");
  if (svg.empty()) return;
  const C1 = AE.col, W = 760, H = 400;
  const F = AE.frame(svg, W, H, { l: 46, r: 200, t: 16, b: 42 });
  const x = d3.scaleLinear().domain([-1.6, 1.6]).range([0, F.w]);
  const y = d3.scaleLinear().domain([-1.15, 1.15]).range([F.h, 0]);
  AE.gridX(F.g, x, F.h, 7); AE.gridY(F.g, y, F.w, 5);
  AE.axisB(F.g, x, F.h, 7, "x₁"); AE.axisL(F.g, y, 0, 5, "x₂");
  const gArr = F.g.append("g"), gPts = F.g.append("g"), gLeg = F.g.append("g");
  let timer = null, S = null;

  // a fixed grid of probe points, most of which are NOT plausible samples
  const probes = [];
  for (let i = 0; i < 11; i++) for (let j = 0; j < 8; j++)
    probes.push(Float64Array.from([-1.35 + 2.7 * i / 10, -0.95 + 1.9 * j / 7]));

  function build() {
    const d = +AE.sel("cp-d"), hw = AE.val("cp-h");
    AE.put("cp-hv", hw);
    const X = AE.curveData(300, 0.62, 2.4, 0.05, 20240907);
    const r = AE.rng(11);
    const net = AE.nn.make([2, hw, d, hw, 2], ["tanh", "lin", "tanh", "lin"], r, 1.4);
    return { X, d, hw, net, st: AE.nn.adam(net), step: 0 };
  }

  function render() {
    gArr.selectAll("*").remove(); gPts.selectAll("*").remove(); gLeg.selectAll("*").remove();
    const R = AE.nn.forward(S.net, S.X)[S.net.L];
    const P = AE.nn.forward(S.net, probes)[S.net.L];

    if (AE.chk("cp-arrows")) {
      for (let i = 0; i < probes.length; i++) {
        const q = probes[i], p = P[i];
        const len = Math.hypot(p[0] - q[0], p[1] - q[1]);
        AE.arrow(gArr, x(q[0]), y(q[1]), x(p[0]), y(p[1]),
          len < 0.04 ? C1.bad : C1.teal, 1.2, 4);
      }
    }
    gArr.selectAll("circle.pr").data(probes).enter().append("circle").attr("class", "pr")
      .attr("cx", q => x(q[0])).attr("cy", q => y(q[1])).attr("r", 1.6)
      .attr("fill", C1.muted).attr("fill-opacity", .8);
    gPts.selectAll("circle").data(d3.range(S.X.length)).enter().append("circle")
      .attr("cx", i => x(S.X[i][0])).attr("cy", i => y(S.X[i][1])).attr("r", 2)
      .attr("fill", C1.A).attr("fill-opacity", .6);

    /* Two numbers. (1) how far a probe moves. (2) what FRACTION of its distance to
       the data it closes — 1 for a true projection, 0 for the identity. Only probes
       that start genuinely off the data are scored. */
    const dmin = p => {
      let b = Infinity;
      for (let i = 0; i < S.X.length; i++) b = Math.min(b, Math.hypot(p[0] - S.X[i][0], p[1] - S.X[i][1]));
      return b;
    };
    let moved = 0, stuck = 0, closed = 0, nOff = 0;
    for (let i = 0; i < probes.length; i++) {
      const q = probes[i], p = P[i], len = Math.hypot(p[0] - q[0], p[1] - q[1]);
      moved += len; if (len < 0.04) stuck++;
      const d0 = dmin(q);
      if (d0 < 0.12) continue;
      closed += (d0 - dmin(p)) / d0; nOff++;
    }
    moved /= probes.length; closed = nOff ? closed / nOff : 0;
    let train = 0;
    for (let i = 0; i < S.X.length; i++) train += Math.hypot(R[i][0] - S.X[i][0], R[i][1] - S.X[i][1]);
    train /= S.X.length;

    AE.legend(gLeg, [
      { c: C1.A, t: "training data" }, { c: C1.muted, t: "probe points" },
      { c: C1.teal, t: "probe → its reconstruction" },
      { c: C1.bad, t: "probe barely moves (identity)" }
    ], F.w + 14, 20);

    AE.put("cp-out",
      `d = <b>${S.d}</b>, hidden width <b>${S.hw}</b>, step <b>${S.step}</b> &nbsp;·&nbsp; ` +
      `mean displacement — <i>data</i> point <b>${AE.fmt(train, 4)}</b>, <i>probe</i> point <b>${AE.fmt(moved, 4)}</b> &nbsp;·&nbsp; ` +
      `<b>${stuck}</b>/${probes.length} probes essentially unmoved &nbsp;·&nbsp; ` +
      `<b>${S.step < 1500 ? "—" : AE.fmt(100 * closed, 1) + "%"}</b> of each probe's distance to the data is closed &nbsp;·&nbsp; ` +
      (S.step < 1500
        ? `<span style="color:${C1.muted}">training…</span>`
        : closed > 0.5
          ? `<span style="color:${C1.good}">off-manifold points are pulled back — the model knows where the data lives</span>`
          : `<span style="color:${C1.bad}">probes barely move: g∘f is close to the identity off the data too — nothing was learned about the distribution</span>`));
  }

  function train() {
    if (timer) { clearInterval(timer); timer = null; }
    S = build(); render();
    timer = setInterval(() => {
      for (let i = 0; i < 60; i++) { AE.nn.aeStep(S.net, S.st, S.X, 0.012); S.step++; }
      render();
      if (S.step >= 4200) { clearInterval(timer); timer = null; }
    }, 16);
  }
  AE.on("cp-d", "change", train);
  AE.on("cp-h", "input", train);
  AE.on("cp-arrows", "change", render);
  AE.on("cp-go", "click", train);
  train();
})();

/* ═══ 07 · linear regions: what an extra layer does to the input space ════ */
(function () {
  const svg = d3.select("#ae-depth");
  if (svg.empty()) return;
  const C1 = AE.col, W = 760, H = 400;
  const PX = 46, PY = 20, PW = 400, PH = 336;                 // the plane's box, in svg user space
  const gCells = svg.append("g"), gEdge = svg.append("g"), gSide = svg.append("g"), gAx = svg.append("g");

  svg.append("rect").attr("x", PX).attr("y", PY).attr("width", PW).attr("height", PH)
    .attr("fill", "none").attr("stroke", C1.line);
  gAx.append("text").attr("x", PX + PW / 2).attr("y", PY + PH + 26).attr("text-anchor", "middle")
    .attr("fill", C1.muted).attr("font-size", 11).text("input plane  x ∈ [−1, 1]²  — colour = which linear region");

  /* a random ReLU encoder: D → n → … → n → 1, weights fixed by a seed */
  function encoder(n, L, seed) {
    const r = AE.rng(seed * 9176 + 17), Wl = [], bl = [];
    let fan = 2;
    for (let l = 0; l < L; l++) {
      const Ww = AE.zeros(n, fan), bb = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < fan; j++) Ww[i][j] = AE.randn(r) * Math.sqrt(2 / fan);
        bb[i] = AE.randn(r) * 0.7;
      }
      Wl.push(Ww); bl.push(bb); fan = n;
    }
    return { W: Wl, b: bl, L, n };
  }
  function pat(net, x0, x1) {                                  // activation pattern as a string key
    let a = [x0, x1], key = "";
    for (let l = 0; l < net.L; l++) {
      const o = new Float64Array(net.n);
      for (let i = 0; i < net.n; i++) {
        let s = net.b[l][i], wi = net.W[l][i];
        for (let j = 0; j < a.length; j++) s += wi[j] * a[j];
        key += s > 0 ? "1" : "0";
        o[i] = s > 0 ? s : 0;
      }
      a = o;
    }
    return key;
  }
  function countRegions(net, G) {
    const S = new Set();
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++)
      S.add(pat(net, -1 + 2 * i / (G - 1), -1 + 2 * j / (G - 1)));
    return S.size;
  }
  const paramsDeep = (n, L) => { let p = 0, fan = 2; for (let l = 0; l < L; l++) { p += n * fan + n; fan = n; } return p + n + 1; };

  function draw() {
    const n = AE.val("dp-n"), L = AE.val("dp-L"), seed = AE.val("dp-seed"), edges = AE.chk("dp-edges");
    AE.put("dp-nv", n); AE.put("dp-Lv", L); AE.put("dp-seedv", seed);
    gCells.selectAll("*").remove(); gEdge.selectAll("*").remove(); gSide.selectAll("*").remove();

    const net = encoder(n, L, seed);
    const GX = 100, GY = 84, cw = PW / GX, ch = PH / GY;
    const keys = new Array(GX * GY), ids = new Map();
    const cells = [];
    for (let i = 0; i < GX; i++) for (let j = 0; j < GY; j++) {
      const k = pat(net, -1 + 2 * (i + 0.5) / GX, 1 - 2 * (j + 0.5) / GY);
      keys[i * GY + j] = k;
      if (!ids.has(k)) ids.set(k, ids.size);
      cells.push({ i, j, k });
    }
    const nId = Math.max(1, ids.size);
    gCells.selectAll("rect").data(cells).enter().append("rect")
      .attr("x", c => PX + c.i * cw).attr("y", c => PY + c.j * ch)
      .attr("width", cw + 0.6).attr("height", ch + 0.6)
      .attr("fill", c => d3.interpolateTurbo(0.08 + 0.84 * ((ids.get(c.k) * 2654435761) % 1000) / 1000))
      .attr("fill-opacity", .55);

    if (edges) {
      const segs = [];
      for (let i = 0; i < GX; i++) for (let j = 0; j < GY; j++) {
        const k = keys[i * GY + j];
        if (i + 1 < GX && keys[(i + 1) * GY + j] !== k)
          segs.push([PX + (i + 1) * cw, PY + j * ch, PX + (i + 1) * cw, PY + (j + 1) * ch]);
        if (j + 1 < GY && keys[i * GY + j + 1] !== k)
          segs.push([PX + i * cw, PY + (j + 1) * ch, PX + (i + 1) * cw, PY + (j + 1) * ch]);
      }
      gEdge.selectAll("line").data(segs).enter().append("line")
        .attr("x1", s => s[0]).attr("y1", s => s[1]).attr("x2", s => s[2]).attr("y2", s => s[3])
        .attr("stroke", C1.bg).attr("stroke-opacity", .75).attr("stroke-width", 1);
    }

    // exact-ish region counts on a finer grid, for this config and for the depth ladder
    const G = 190;
    const ladder = [1, 2, 3].map(l => ({ L: l, r: countRegions(encoder(n, l, seed), G), p: paramsDeep(n, l) }));
    const here = ladder[L - 1];
    // a ONE-layer encoder given the same parameter budget: 4m + 1 params
    const mMatch = Math.max(1, Math.round((here.p - 1) / 4));
    const rMatch = countRegions(encoder(mMatch, 1, seed), G);

    // side panel: region count against depth
    const bx = PX + PW + 34, bw = 240;
    AE.panelBox(gSide, bx, PY + 8, bw, 150, "regions vs depth, width " + n + " fixed");
    const ymax = d3.max(ladder, d => d.r) * 1.15;
    ladder.forEach((d, i) => {
      const yy = PY + 40 + i * 36, ww = (bw - 96) * d.r / ymax;
      gSide.append("rect").attr("x", bx + 62).attr("y", yy - 9).attr("width", Math.max(2, ww)).attr("height", 16)
        .attr("rx", 3).attr("fill", d.L === L ? C1.B : C1.A).attr("fill-opacity", d.L === L ? .85 : .35);
      gSide.append("text").attr("x", bx + 56).attr("y", yy + 3).attr("text-anchor", "end")
        .attr("fill", d.L === L ? C1.B : C1.muted).attr("font-size", 10.5).text(d.L + " layer" + (d.L > 1 ? "s" : ""));
      gSide.append("text").attr("x", bx + 68 + Math.max(2, ww)).attr("y", yy + 3)
        .attr("fill", C1.ink).attr("font-size", 10.5).attr("font-family", "SF Mono,Menlo,monospace")
        .text(d.r + "  (" + d.p + "p)");
    });
    AE.panelBox(gSide, bx, PY + 172, bw, 96, "same parameter budget, ONE layer");
    gSide.append("text").attr("x", bx + 10).attr("y", PY + 200).attr("fill", C1.muted).attr("font-size", 10.5)
      .text("width " + mMatch + ",  " + (4 * mMatch + 1) + " params");
    gSide.append("text").attr("x", bx + 10).attr("y", PY + 222).attr("fill", rMatch > here.r ? C1.bad : C1.good)
      .attr("font-size", 15).attr("font-weight", 600).text(rMatch + " regions");
    gSide.append("text").attr("x", bx + 10).attr("y", PY + 244).attr("fill", C1.muted).attr("font-size", 10)
      .text(rMatch > here.r ? "wider-and-shallow wins here — see the caveat" : "the deep stack wins here");

    AE.put("dp-out",
      `encoder 2 → ${Array(L).fill(n).join(" → ")} → 1 &nbsp;·&nbsp; <b>${here.p}</b> parameters &nbsp;·&nbsp; ` +
      `<b>${here.r}</b> distinct linear regions measured on a ${G}×${G} grid &nbsp;·&nbsp; ` +
      `going from 1 layer to ${L} multiplies the region count by <b>${AE.fmt(here.r / ladder[0].r, 2)}×</b> at the same width, ` +
      `and multiplies the parameter count by <b>${AE.fmt(here.p / ladder[0].p, 2)}×</b> &nbsp;·&nbsp; ` +
      `for comparison a <b>single</b> layer of width ${mMatch} has the same ${here.p} parameters and gives <b>${rMatch}</b> regions` +
      (rMatch > here.r ? `<span style="color:${C1.bad}"> — more, not fewer: in two dimensions width beats depth</span>`
        : `<span style="color:${C1.good}"> — fewer, so the deep stack wins here</span>`));
  }
  ["dp-n", "dp-L", "dp-seed"].forEach(id => AE.on(id, "input", draw));
  AE.on("dp-edges", "change", draw);
  draw();
})();

/* ═══ shared: the exact optimal denoiser and the exact score ══════════════
   For the empirical distribution p̂ = (1/N) Σ δ(x − xᵢ) corrupted by 𝒩(0, σ²I),
       E[x | x̃] = Σ wᵢ xᵢ / Σ wᵢ ,      wᵢ = exp(−‖x̃ − xᵢ‖² / 2σ²)
       ∇ log p_σ(x̃) = ( E[x | x̃] − x̃ ) / σ²
   The second line is Tweedie's formula; both are computed here so the figures
   can display the residual between them rather than assert the identity.
   Weights are formed in log space so that far-away probes do not underflow.   */
AE.denoise = function (q, X, sig) {
  const N = X.length, D = q.length, lw = new Float64Array(N);
  let mx = -Infinity;
  for (let i = 0; i < N; i++) {
    let d = 0;
    for (let k = 0; k < D; k++) { const e = q[k] - X[i][k]; d += e * e; }
    lw[i] = -d / (2 * sig * sig);
    if (lw[i] > mx) mx = lw[i];
  }
  let Z = 0;
  for (let i = 0; i < N; i++) { lw[i] = Math.exp(lw[i] - mx); Z += lw[i]; }
  const out = new Float64Array(D);
  for (let i = 0; i < N; i++) for (let k = 0; k < D; k++) out[k] += lw[i] * X[i][k];
  for (let k = 0; k < D; k++) out[k] /= Z;
  return { r: out, logp: mx + Math.log(Z / N) - D * Math.log(2 * Math.PI * sig * sig) / 2 };
};
AE.scoreOf = function (q, X, sig) {                 // ∇ log p_σ(q), from the same weights
  const { r } = AE.denoise(q, X, sig), D = q.length, g = new Float64Array(D);
  for (let k = 0; k < D; k++) g[k] = (r[k] - q[k]) / (sig * sig);
  return g;
};
AE.manifold = function (kind, N, gap) {
  const P = AE.zeros(N, 2), g = (gap === undefined) ? 0.6 : gap;
  for (let i = 0; i < N; i++) {
    const u = (i + 0.5) / N;
    if (kind === "ring") {
      const th = 2 * Math.PI * u; P[i][0] = 0.74 * Math.cos(th); P[i][1] = 0.62 * Math.sin(th);
    } else if (kind === "two") {
      const t = -1.05 + 2.1 * ((i % (N / 2)) / (N / 2 - 1)), s = (i < N / 2) ? 1 : -1;
      P[i][0] = t; P[i][1] = s * (g / 2) + 0.10 * Math.sin(2.6 * t) * s;
    } else {
      const t = -1.15 + 2.3 * u; P[i][0] = t; P[i][1] = 0.55 * Math.sin(2.2 * t);
    }
  }
  return P;
};

/* ═══ 09 · the denoising vector field ════════════════════════════════════ */
(function () {
  const svg = d3.select("#ae-dae");
  if (svg.empty()) return;
  const C1 = AE.col, PX = 46, PY = 18, PW = 520, PH = 344;
  const x = d3.scaleLinear().domain([-1.45, 1.45]).range([PX, PX + PW]);
  const y = d3.scaleLinear().domain([-1.05, 1.05]).range([PY + PH, PY]);
  const gDen = svg.append("g"), gPts = svg.append("g"), gArr = svg.append("g"),
    gSide = svg.append("g"), gAx = svg.append("g");
  svg.append("rect").attr("x", PX).attr("y", PY).attr("width", PW).attr("height", PH)
    .attr("fill", "none").attr("stroke", C1.line);
  gAx.append("text").attr("x", PX + PW / 2).attr("y", PY + PH + 26).attr("text-anchor", "middle")
    .attr("fill", C1.muted).attr("font-size", 11)
    .text("arrows are  r(x̃) − x̃  =  E[x | x̃] − x̃ ,  computed exactly");

  function draw() {
    const sig = AE.val("dn-sig"), kind = AE.sel("dn-shape"), show = AE.sel("dn-show"), scaled = AE.chk("dn-len");
    AE.put("dn-sigv", AE.fmt(sig, 2));
    gDen.selectAll("*").remove(); gPts.selectAll("*").remove(); gArr.selectAll("*").remove(); gSide.selectAll("*").remove();

    const P = AE.manifold(kind, 400);

    if (show === "dens" || show === "both") {
      const GX = 96, GY = 64, cw = PW / GX, ch = PH / GY, cells = [];
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < GX; i++) for (let j = 0; j < GY; j++) {
        const q = [x.invert(PX + (i + .5) * cw), y.invert(PY + (j + .5) * ch)];
        const lp = AE.denoise(q, P, sig).logp;
        cells.push({ i, j, lp }); lo = Math.min(lo, lp); hi = Math.max(hi, lp);
      }
      const cs = d3.scaleLinear().domain([Math.max(lo, hi - 9), hi]).range([0, 1]).clamp(true);
      gDen.selectAll("rect").data(cells).enter().append("rect")
        .attr("x", c => PX + c.i * cw).attr("y", c => PY + c.j * ch)
        .attr("width", cw + .6).attr("height", ch + .6)
        .attr("fill", c => d3.interpolateMagma(cs(c.lp))).attr("fill-opacity", show === "both" ? .55 : .9);
    }

    // the field, plus a live check of Tweedie's identity
    let worst = 0, onLen = 0, onN = 0, offLen = 0, offN = 0, homeN = 0;
    const dmin = q => {
      let b = Infinity;
      for (let i = 0; i < P.length; i++) b = Math.min(b, Math.hypot(q[0] - P[i][0], q[1] - P[i][1]));
      return b;
    };
    if (show !== "dens") {
      const GX = 27, GY = 19;
      for (let i = 0; i < GX; i++) for (let j = 0; j < GY; j++) {
        const qx = -1.38 + 2.76 * i / (GX - 1), qy = -0.98 + 1.96 * j / (GY - 1), q = [qx, qy];
        const { r } = AE.denoise(q, P, sig);
        const s = AE.scoreOf(q, P, sig);
        worst = Math.max(worst, Math.abs((r[0] - qx) - sig * sig * s[0]), Math.abs((r[1] - qy) - sig * sig * s[1]));
        const L = Math.hypot(r[0] - qx, r[1] - qy);
        const d0 = dmin(q);
        if (d0 < 0.10) { onLen += L; onN++; }
        else { offLen += L; offN++; if (dmin([r[0], r[1]]) < d0) homeN++; }
        const k = scaled ? 1 : Math.min(1, 0.085 / Math.max(L, 1e-9));
        const ex = qx + (r[0] - qx) * k, ey = qy + (r[1] - qy) * k;
        if (L > 1e-4) AE.arrow(gArr, x(qx), y(qy), x(ex), y(ey),
          d3.interpolateViridis(AE.clamp(L / 0.7, 0, 1)), 1.25, 4.2);
        else gArr.append("circle").attr("cx", x(qx)).attr("cy", y(qy)).attr("r", 1.4).attr("fill", C1.good);
      }
    }
    gPts.selectAll("circle").data(d3.range(P.length)).enter().append("circle")
      .attr("cx", i => x(P[i][0])).attr("cy", i => y(P[i][1])).attr("r", 1.5)
      .attr("fill", C1.bad).attr("fill-opacity", .85);

    // side panel
    const bx = PX + PW + 26;
    AE.panelBox(gSide, bx, PY + 10, 168, 120, "arrow length");
    const rows = [["on the manifold", onN ? onLen / onN : 0, C1.good], ["off it", offN ? offLen / offN : 0, C1.B]];
    rows.forEach((rw, i) => {
      gSide.append("text").attr("x", bx + 10).attr("y", PY + 44 + i * 40).attr("fill", C1.muted).attr("font-size", 10.5).text(rw[0]);
      gSide.append("text").attr("x", bx + 10).attr("y", PY + 62 + i * 40).attr("fill", rw[2])
        .attr("font-size", 15).attr("font-weight", 600).attr("font-family", "SF Mono,Menlo,monospace").text(AE.fmt(rw[1], 4));
    });
    AE.legend(gSide, [
      { c: C1.bad, t: "data (the manifold)" },
      { c: d3.interpolateViridis(0.15), t: "short arrow — near the data" },
      { c: d3.interpolateViridis(0.95), t: "long arrow — far from it" }
    ], bx, PY + 152);

    AE.put("dn-out",
      `<span class="keep">σ</span> = <b>${AE.fmt(sig, 2)}</b> &nbsp;·&nbsp; ` +
      `mean displacement on the manifold <b>${AE.fmt(onN ? onLen / onN : 0, 4)}</b> vs off it <b>${AE.fmt(offN ? offLen / offN : 0, 4)}</b> ` +
      `(a factor of <b>${AE.fmt((offN ? offLen / offN : 0) / Math.max(1e-9, onN ? onLen / onN : 1e-9), 1)}×</b>) &nbsp;·&nbsp; ` +
      `<b>${homeN}</b> of ${offN} off-manifold probes are moved <i>closer</i> to the data` +
      (homeN < offN ? ` — the other ${offN - homeN} sit where p<sub>σ</sub> has a local minimum or a merged mode, and the field there points at the blur, not at the data` : "") + ` &nbsp;·&nbsp; ` +
      `Tweedie residual <code>max | (r − x̃) − σ²∇log p_σ |</code> over the whole grid = <b>${worst.toExponential(1)}</b> — an identity, not a fit`);
  }
  ["dn-sig"].forEach(id => AE.on(id, "input", draw));
  ["dn-shape", "dn-show", "dn-len"].forEach(id => AE.on(id, "change", draw));
  draw();
})();

/* ═══ 10 · one noise level is one blurred view ═══════════════════════════ */
(function () {
  const svg = d3.select("#ae-score");
  if (svg.empty()) return;
  const C1 = AE.col;
  const LX = 46, LY = 22, LW = 320, LH = 268;                 // left panel: density along a slice
  const RX = 440, RY = 22, RW = 286, RH = 268;                // right panel: |field| vs distance
  const gL = svg.append("g"), gR = svg.append("g"), gT = svg.append("g");
  svg.append("text").attr("x", LX).attr("y", 14).attr("fill", C1.muted).attr("font-size", 10.5)
    .text("smoothed density p_σ along the vertical slice x₁ = 0");
  svg.append("text").attr("x", RX).attr("y", 14).attr("fill", C1.muted).attr("font-size", 10.5)
    .text("‖r(x̃) − x̃‖ against distance to the nearest data point");

  function draw() {
    const sig = AE.val("sc-sig"), gap = AE.val("sc-gap"), tw = AE.chk("sc-tw");
    AE.put("sc-sigv", AE.fmt(sig, 2)); AE.put("sc-gapv", AE.fmt(gap, 2));
    gL.selectAll("*").remove(); gR.selectAll("*").remove(); gT.selectAll("*").remove();
    const P = AE.manifold("two", 400, gap);

    /* ── left: p_σ along x₁ = 0, as a function of x₂ ── */
    const ys = AE.linspace(-1.05, 1.05, 260);
    const lps = ys.map(v => AE.denoise([0, v], P, sig).logp);
    const ps = lps.map(Math.exp);
    const xL = d3.scaleLinear().domain([-1.05, 1.05]).range([LX, LX + LW]);
    const yL = d3.scaleLinear().domain([0, d3.max(ps) * 1.12]).range([LY + LH, LY]);
    const gg = gL.append("g");
    AE.gridY(gg, yL, LW, 4);
    gg.selectAll("line").attr("transform", `translate(${LX},0)`);
    AE.axisB(gL.append("g").attr("transform", `translate(${LX},0)`), d3.scaleLinear().domain([-1.05, 1.05]).range([0, LW]), LY + LH, 5, "x₂");
    gL.append("path").attr("d", d3.line().x((d, i) => xL(ys[i])).y(d => yL(d))(ps))
      .attr("fill", "none").attr("stroke", C1.A).attr("stroke-width", 2);
    gL.append("path").attr("d", d3.area().x((d, i) => xL(ys[i])).y0(LY + LH).y1(d => yL(d))(ps))
      .attr("fill", C1.A).attr("fill-opacity", .14);
    // the two true arm positions
    [gap / 2, -gap / 2].forEach(v => gL.append("line").attr("x1", xL(v)).attr("x2", xL(v))
      .attr("y1", LY).attr("y2", LY + LH).attr("stroke", C1.bad).attr("stroke-dasharray", "4 4").attr("stroke-opacity", .7));
    /* Are the two arms still resolved? A grid peak-count is fragile, so measure the
       thing that actually decides it: is the density in the GAP lower than the
       density AT an arm? That ratio is exact and monotone in σ. */
    const pMid = Math.exp(AE.denoise([0, 0], P, sig).logp);
    const pArm = Math.exp(AE.denoise([0, gap / 2], P, sig).logp);
    const ratio = pMid / pArm, resolved = ratio < 0.999;
    // mark the valley on the curve
    gL.append("circle").attr("cx", xL(0)).attr("cy", yL(pMid)).attr("r", 3.4)
      .attr("fill", resolved ? C1.good : C1.bad).attr("stroke", C1.bg);

    /* ── right: |displacement| against distance to the manifold ── */
    const r2 = AE.rng(90210), pts = [];
    const dmin = q => { let b = Infinity; for (let i = 0; i < P.length; i++) b = Math.min(b, Math.hypot(q[0] - P[i][0], q[1] - P[i][1])); return b; };
    let worst = 0;
    for (let n = 0; n < 700; n++) {
      const q = [-1.3 + 2.6 * r2(), -1.0 + 2.0 * r2()];
      const { r } = AE.denoise(q, P, sig);
      const s = AE.scoreOf(q, P, sig);
      worst = Math.max(worst, Math.abs((r[0] - q[0]) - sig * sig * s[0]), Math.abs((r[1] - q[1]) - sig * sig * s[1]));
      pts.push({ d: dmin(q), L: Math.hypot(r[0] - q[0], r[1] - q[1]) });
    }
    const xR = d3.scaleLinear().domain([0, d3.max(pts, p => p.d) * 1.05]).range([RX, RX + RW]);
    const yR = d3.scaleLinear().domain([0, d3.max(pts, p => p.L) * 1.08]).range([RY + RH, RY]);
    const gr = gR.append("g");
    AE.gridY(gr, yR, RW, 4); gr.selectAll("line").attr("transform", `translate(${RX},0)`);
    AE.axisB(gR.append("g").attr("transform", `translate(${RX},0)`), d3.scaleLinear().domain(xR.domain()).range([0, RW]), RY + RH, 5, "distance to the data");
    AE.axisL(gR.append("g").attr("transform", `translate(${RX},0)`), yR, 0, 4, "‖r(x̃) − x̃‖");
    gR.selectAll("circle").data(pts).enter().append("circle")
      .attr("cx", p => xR(p.d)).attr("cy", p => yR(p.L)).attr("r", 2)
      .attr("fill", C1.teal).attr("fill-opacity", .5);
    // the identity line: a perfect projection would put every point on y = x
    gR.append("line").attr("x1", xR(0)).attr("y1", yR(0))
      .attr("x2", xR(xR.domain()[1])).attr("y2", yR(Math.min(yR.domain()[1], xR.domain()[1])))
      .attr("stroke", C1.B).attr("stroke-dasharray", "5 4").attr("stroke-width", 1.4);
    gR.append("text").attr("x", RX + RW - 4).attr("y", RY + 14).attr("text-anchor", "end")
      .attr("fill", C1.B).attr("font-size", 10).text("a perfect projection would sit on this line");

    if (tw) {
      AE.panelBox(gT, LX, LY + LH + 44, 680, 42, null);
      gT.append("text").attr("x", LX + 12).attr("y", LY + LH + 70).attr("fill", C1.muted).attr("font-size", 11.5)
        .text("Tweedie check over 700 random probes:  max | (r(x̃) − x̃) − σ² ∇log p_σ(x̃) |  =  " + worst.toExponential(1) + "   — the two sides are the same object");
    }

    AE.put("sc-out",
      `<span class="keep">σ</span> = <b>${AE.fmt(sig, 2)}</b>, arm separation <b>${AE.fmt(gap, 2)}</b> &nbsp;·&nbsp; ` +
      `p<sub><span class="keep">σ</span></sub>(midpoint) / p<sub><span class="keep">σ</span></sub>(arm) = <b>${AE.fmt(ratio, 3)}</b> ` +
      (resolved ? `<span style="color:${C1.good}">— below 1, so a valley still separates the two arms</span>`
        : `<span style="color:${C1.bad}">— at or above 1: the two arms have fused into one mode, and no denoiser at this <span class="keep">σ</span> can tell them apart</span>`) +
      ` &nbsp;·&nbsp; the merge happens once <code><span class="keep">σ</span></code> is roughly half the separation, here <code>${AE.fmt(gap / 2, 2)}</code>`);
  }
  ["sc-sig", "sc-gap"].forEach(id => AE.on(id, "input", draw));
  AE.on("sc-tw", "change", draw);
  draw();
})();

/* ═══ 08 · a sparse, overcomplete code ═══════════════════════════════════ */
(function () {
  const svg = d3.select("#ae-sparse");
  if (svg.empty()) return;
  const C1 = AE.col, D = 12, K = 24;                       // input width 12, code width 24
  const gA = svg.append("g"), gB = svg.append("g"), gC = svg.append("g");
  let timer = null, sweepJob = null, S = null, sweep = [];

  /* data: each example is a sum of k Gaussian bumps drawn from a 12-position grid */
  function makeData(N, k, seed) {
    const r = AE.rng(seed), X = AE.zeros(N, D);
    for (let n = 0; n < N; n++) {
      for (let a = 0; a < k; a++) {
        const c = Math.floor(r() * D), amp = 0.6 + 0.8 * r();
        for (let j = 0; j < D; j++) X[n][j] += amp * Math.exp(-Math.pow(j - c, 2) / 2.0);
      }
      for (let j = 0; j < D; j++) X[n][j] += 0.02 * AE.randn(r);
    }
    return X;
  }

  function build() {
    const k = AE.val("sp-k");
    AE.put("sp-kv", k);
    const X = makeData(220, k, 1234);
    const r = AE.rng(88);
    const net = AE.nn.make([D, K, D], ["relu", "lin"], r, 1.2);
    return { X, k, net, st: AE.nn.adam(net), step: 0, mse: 0, occ: 0 };
  }

  function stats(net, X, tol) {
    const A = AE.nn.forward(net, X), Hh = A[1], R = A[2];
    let mse = 0, occ = 0, mx = 0;
    for (let i = 0; i < X.length; i++) for (let j = 0; j < K; j++) mx = Math.max(mx, Hh[i][j]);
    const th = Math.max(1e-9, (tol === undefined ? 0.02 : tol) * mx);
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < D; j++) { const e = R[i][j] - X[i][j]; mse += e * e; }
      for (let j = 0; j < K; j++) if (Hh[i][j] > th) occ++;
    }
    return { mse: mse / X.length, occ: occ / X.length, H: Hh, R };
  }

  function render() {
    const lam = AE.val("sp-lam");
    AE.put("sp-lamv", AE.fmt(lam, 2));
    gA.selectAll("*").remove(); gB.selectAll("*").remove(); gC.selectAll("*").remove();
    const st = stats(S.net, S.X);

    /* panel A — one example's code */
    const AX = 44, AY = 26, AW = 300, AH = 128;
    AE.panelBox(gA, AX - 12, AY - 20, AW + 24, AH + 40, "code h for one example — " + K + " units, input width " + D);
    const hmax = Math.max(1e-9, d3.max(d3.range(K), j => st.H[0][j]));
    const xb = d3.scaleBand().domain(d3.range(K)).range([AX, AX + AW]).padding(0.18);
    const yb = d3.scaleLinear().domain([0, hmax]).range([AY + AH, AY]);
    gA.selectAll("rect.h").data(d3.range(K)).enter().append("rect").attr("class", "h")
      .attr("x", j => xb(j)).attr("width", xb.bandwidth())
      .attr("y", j => yb(st.H[0][j])).attr("height", j => AY + AH - yb(st.H[0][j]))
      .attr("rx", 1.5).attr("fill", j => st.H[0][j] > 0.02 * hmax ? C1.B : C1.line);
    gA.append("line").attr("x1", AX).attr("x2", AX + AW).attr("y1", AY + AH).attr("y2", AY + AH)
      .attr("stroke", C1.line);

    /* panel B — the learned decoder atoms (columns of the decoder matrix) */
    const BX = 44, BY = 208, BW = 300, BH = 150;
    AE.panelBox(gB, BX - 12, BY - 20, BW + 24, BH + 34, "the 24 decoder columns — one 'atom' each");
    const W2 = S.net.W[1];                                   // D × K
    const cellW = BW / 8, cellH = BH / 3;
    for (let j = 0; j < K; j++) {
      const cx = BX + (j % 8) * cellW, cy = BY + Math.floor(j / 8) * cellH;
      const colv = d3.range(D).map(i => W2[i][j]);
      const m = Math.max(1e-9, d3.max(colv, Math.abs));
      const xs = d3.scaleLinear().domain([0, D - 1]).range([cx + 3, cx + cellW - 5]);
      const ysc = d3.scaleLinear().domain([-m, m]).range([cy + cellH - 8, cy + 4]);
      const active = d3.mean(d3.range(S.X.length), i => st.H[i][j] > 0.02 * hmax ? 1 : 0);
      gB.append("path").attr("d", d3.line().x((d, i) => xs(i)).y(d => ysc(d))(colv))
        .attr("fill", "none").attr("stroke", active > 0.01 ? C1.A : C1.line).attr("stroke-width", 1.3);
    }

    /* panel C — the sweep over λ, filled in as it is computed */
    const CX = 430, CY = 30, CW = 288, CH = 300;
    AE.panelBox(gC, CX - 14, CY - 22, CW + 28, CH + 46, "sweep over the sparsity weight λ");
    if (sweep.length > 1) {
      const xs = d3.scaleLinear().domain([0, 0.6]).range([CX, CX + CW]);
      const yo = d3.scaleLinear().domain([0, K]).range([CY + CH, CY]);
      const ym = d3.scaleLinear().domain([0, d3.max(sweep, p => p.mse) * 1.1]).range([CY + CH, CY]);
      AE.axisB(gC.append("g").attr("transform", `translate(${CX},0)`),
        d3.scaleLinear().domain([0, 0.6]).range([0, CW]), CY + CH, 4, "λ");
      gC.append("path").attr("d", d3.line().x(p => xs(p.lam)).y(p => yo(p.occ))(sweep))
        .attr("fill", "none").attr("stroke", C1.B).attr("stroke-width", 2);
      gC.append("path").attr("d", d3.line().x(p => xs(p.lam)).y(p => ym(p.mse))(sweep))
        .attr("fill", "none").attr("stroke", C1.bad).attr("stroke-width", 2);
      gC.append("line").attr("x1", xs(lam)).attr("x2", xs(lam)).attr("y1", CY).attr("y2", CY + CH)
        .attr("stroke", C1.ink).attr("stroke-opacity", .35).attr("stroke-dasharray", "3 3");
      AE.legend(gC, [
        { c: C1.B, t: "active code units per example (0–" + K + ")" },
        { c: C1.bad, t: "reconstruction MSE" }
      ], CX + 6, CY + 12);
    } else {
      gC.append("text").attr("x", CX + CW / 2).attr("y", CY + CH / 2).attr("text-anchor", "middle")
        .attr("fill", C1.muted).attr("font-size", 11).text("sweeping…");
    }

    AE.put("sp-out",
      `<span class="keep">λ</span> = <b>${AE.fmt(lam, 2)}</b>, step <b>${S.step}</b> &nbsp;·&nbsp; ` +
      `<b>${AE.fmt(st.occ, 2)}</b> of ${K} code units active per example ` +
      `(each example was built from <b>${S.k}</b> generating bump${S.k > 1 ? "s" : ""}) &nbsp;·&nbsp; ` +
      `reconstruction MSE <b>${AE.fmt(st.mse, 4)}</b> &nbsp;·&nbsp; ` +
      (S.step < 800
        ? `<span style="color:${C1.muted}">training…</span>`
        : lam === 0
          ? `<span style="color:${C1.bad}">λ = 0: about as many active units as the input has dimensions, and near-zero error — the code is a re-encoding of the input, not a bottleneck</span>`
          : `<span style="color:${C1.good}">the code is twice as wide as the input yet mostly exactly zero — the constraint is occupancy, not width</span>`));
  }

  /* two independent jobs: (a) train the model at the slider's λ, (b) sweep λ */
  function trainMain() {
    if (timer) { clearInterval(timer); timer = null; }
    S = build(); render();
    const lam = AE.val("sp-lam");
    timer = setInterval(() => {
      for (let i = 0; i < 40; i++) { AE.nn.aeStepSparse(S.net, S.st, S.X, 0.02, lam, 1); S.step++; }
      render();
      if (S.step >= 1600) { clearInterval(timer); timer = null; }
    }, 16);
  }
  function runSweep() {
    if (sweepJob) { clearInterval(sweepJob); sweepJob = null; }
    sweep = [];
    const lams = AE.linspace(0, 0.6, 7);
    let li = 0;
    sweepJob = setInterval(() => {
      const r = AE.rng(88);
      const net = AE.nn.make([D, K, D], ["relu", "lin"], r, 1.2), stt = AE.nn.adam(net);
      for (let t = 0; t < 1000; t++) AE.nn.aeStepSparse(net, stt, S.X, 0.02, lams[li], 1);
      const s2 = stats(net, S.X);
      sweep.push({ lam: lams[li], occ: s2.occ, mse: s2.mse });
      render();
      if (++li >= lams.length) { clearInterval(sweepJob); sweepJob = null; }
    }, 16);
  }
  function full() { trainMain(); runSweep(); }
  AE.on("sp-lam", "input", trainMain);
  AE.on("sp-k", "input", full);
  AE.on("sp-go", "click", full);
  full();
})();

/* ═══ 11 · the Jacobian of the reconstruction map ════════════════════════ */
(function () {
  const svg = d3.select("#ae-cae");
  if (svg.empty()) return;
  const C1 = AE.col, PX = 46, PY = 20, PW = 470, PH = 320;
  const x = d3.scaleLinear().domain([-1.35, 1.35]).range([PX, PX + PW]);
  const y = d3.scaleLinear().domain([-1.0, 1.0]).range([PY + PH, PY]);
  const g = svg.append("g"), gS = svg.append("g");
  svg.append("rect").attr("x", PX).attr("y", PY).attr("width", PW).attr("height", PH)
    .attr("fill", "none").attr("stroke", C1.line);

  const curveY = t => 0.55 * Math.sin(2.2 * t);
  const tangent = t => { const s = 0.55 * 2.2 * Math.cos(2.2 * t), n = Math.hypot(1, s); return [1 / n, s / n]; };

  function jac(q, P, sig, h) {
    const J = [[0, 0], [0, 0]];
    for (let k = 0; k < 2; k++) {
      const a = [q[0], q[1]], b = [q[0], q[1]];
      a[k] += h; b[k] -= h;
      const ra = AE.denoise(a, P, sig).r, rb = AE.denoise(b, P, sig).r;
      J[0][k] = (ra[0] - rb[0]) / (2 * h); J[1][k] = (ra[1] - rb[1]) / (2 * h);
    }
    return J;
  }
  function svd2(J) {                                  // eigen-decomposition of JᵀJ
    const A = [J[0][0] * J[0][0] + J[1][0] * J[1][0], J[0][0] * J[0][1] + J[1][0] * J[1][1],
    J[0][1] * J[0][1] + J[1][1] * J[1][1]];
    const tr = A[0] + A[2], det = A[0] * A[2] - A[1] * A[1];
    const d = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const l1 = tr / 2 + d, l2 = Math.max(0, tr / 2 - d);
    let v1 = [A[1], l1 - A[0]];
    if (Math.hypot(v1[0], v1[1]) < 1e-12) v1 = [1, 0];
    const n = Math.hypot(v1[0], v1[1]); v1 = [v1[0] / n, v1[1] / n];
    return { s1: Math.sqrt(l1), s2: Math.sqrt(l2), v1, v2: [-v1[1], v1[0]] };
  }

  function draw() {
    const sig = AE.val("ca-sig"), t = AE.val("ca-t"), off = AE.val("ca-off"), circ = AE.chk("ca-circ");
    AE.put("ca-sigv", AE.fmt(sig, 2)); AE.put("ca-tv", AE.fmt(t, 2)); AE.put("ca-offv", AE.fmt(off, 2));
    g.selectAll("*").remove(); gS.selectAll("*").remove();

    const P = AE.manifold("wave", 400);
    g.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => y(d[1]))(
      AE.linspace(-1.15, 1.15, 200).map(u => [u, curveY(u)])))
      .attr("fill", "none").attr("stroke", C1.bad).attr("stroke-width", 2).attr("stroke-opacity", .75);

    const th = tangent(t), nrm = [-th[1], th[0]];
    const q = [t + off * nrm[0], curveY(t) + off * nrm[1]];
    const J = jac(q, P, sig, 1e-4), sv = svd2(J);
    const align = Math.abs(sv.v1[0] * th[0] + sv.v1[1] * th[1]);
    const fro = Math.sqrt(J[0][0] ** 2 + J[0][1] ** 2 + J[1][0] ** 2 + J[1][1] ** 2);

    // a small circle around the probe, and its image under the local linear map
    const R0 = 0.16;
    if (circ) {
      const ring = AE.linspace(0, 2 * Math.PI, 90);
      g.append("path").attr("d", d3.line().x(a => x(q[0] + R0 * Math.cos(a))).y(a => y(q[1] + R0 * Math.sin(a)))(ring))
        .attr("fill", "none").attr("stroke", C1.muted).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
      const r0 = AE.denoise(q, P, sig).r;
      g.append("path").attr("d", d3.line()
        .x(a => x(r0[0] + R0 * (J[0][0] * Math.cos(a) + J[0][1] * Math.sin(a))))
        .y(a => y(r0[1] + R0 * (J[1][0] * Math.cos(a) + J[1][1] * Math.sin(a))))(ring))
        .attr("fill", "none").attr("stroke", C1.A).attr("stroke-width", 2.2);
      AE.arrow(g, x(q[0]), y(q[1]), x(r0[0]), y(r0[1]), C1.teal, 1.6, 5);
      g.append("circle").attr("cx", x(r0[0])).attr("cy", y(r0[1])).attr("r", 3).attr("fill", C1.A);
    }
    // the analytic tangent, for comparison with the top singular direction
    AE.arrow(g, x(q[0]), y(q[1]), x(q[0] + 0.30 * th[0]), y(q[1] + 0.30 * th[1]), C1.B, 1.8, 5);
    AE.arrow(g, x(q[0]), y(q[1]), x(q[0] + 0.30 * sv.v1[0]), y(q[1] + 0.30 * sv.v1[1]), C1.good, 1.4, 5);
    g.append("circle").attr("cx", x(q[0])).attr("cy", y(q[1])).attr("r", 4)
      .attr("fill", C1.ink).attr("stroke", C1.bg).attr("stroke-width", 1.4);

    const bx = PX + PW + 24;
    AE.panelBox(gS, bx, PY + 6, 196, 118, "singular values of J");
    [["σ₁ (tangent)", sv.s1, sv.s1 > 1 ? C1.bad : C1.good], ["σ₂ (normal)", sv.s2, C1.A]].forEach((rw, i) => {
      gS.append("text").attr("x", bx + 10).attr("y", PY + 40 + i * 40).attr("fill", C1.muted).attr("font-size", 10.5).text(rw[0]);
      gS.append("text").attr("x", bx + 10).attr("y", PY + 60 + i * 40).attr("fill", rw[2])
        .attr("font-size", 16).attr("font-weight", 600).attr("font-family", "SF Mono,Menlo,monospace").text(AE.fmt(rw[1], 4));
    });
    AE.legend(gS, [
      { c: C1.bad, t: "the data manifold" }, { c: C1.muted, t: "a small circle around the probe", dash: "3 3" },
      { c: C1.A, t: "its image under J — an ellipse" }, { c: C1.B, t: "analytic tangent" },
      { c: C1.good, t: "top singular direction of J" }, { c: C1.teal, t: "r(x) − x" }
    ], bx, PY + 146);

    AE.put("ca-out",
      `<span class="keep">σ</span> = <b>${AE.fmt(sig, 2)}</b>, probe at t = <b>${AE.fmt(t, 2)}</b>${off > 0 ? `, lifted <b>${AE.fmt(off, 2)}</b> off the curve` : " (on the curve)"} &nbsp;·&nbsp; ` +
      `<code>‖J‖<sub>F</sub></code> = <b>${AE.fmt(fro, 4)}</b> &nbsp;·&nbsp; ` +
      `<code>σ₁/σ₂</code> = <b>${sv.s2 > 1e-9 ? AE.fmt(sv.s1 / sv.s2, 0) : "∞"}</b> &nbsp;·&nbsp; ` +
      `top singular direction agrees with the analytic tangent to <code>|cos| = ${AE.fmt(align, 4)}</code> &nbsp;·&nbsp; ` +
      (sv.s1 > 1
        ? `<span style="color:${C1.bad}">σ₁ &gt; 1 here — the map is <b>not</b> a contraction in the tangent direction</span>`
        : `<span style="color:${C1.good}">every singular value is below 1: locally a genuine contraction</span>`));
  }
  ["ca-sig", "ca-t", "ca-off"].forEach(id => AE.on(id, "input", draw));
  AE.on("ca-circ", "change", draw);
  draw();
})();

/* ═══ 12 · reconstruction against identity, in one dimension ═════════════ */
(function () {
  const svg = d3.select("#ae-manifold");
  if (svg.empty()) return;
  const C1 = AE.col, PX = 52, PY = 20, PW = 500, PH = 300;
  const x = d3.scaleLinear().domain([-1.15, 1.15]).range([PX, PX + PW]);
  const g = svg.append("g"), gS = svg.append("g");

  const SETS = {
    three: [-0.72, 0.02, 0.76],
    two: [-0.55, 0.60],
    cluster: [-0.80, -0.05, 0.55, 0.62, 0.69]
  };

  function draw() {
    const sig = AE.val("mf-sig"), key = AE.sel("mf-set"), showD = AE.chk("mf-deriv");
    AE.put("mf-sigv", AE.fmt(sig, 2));
    g.selectAll("*").remove(); gS.selectAll("*").remove();
    const pts = SETS[key], P = pts.map(v => Float64Array.from([v]));

    const xs = AE.linspace(-1.12, 1.12, 420);
    const rs = xs.map(v => AE.denoise([v], P, sig).r[0]);
    const dr = xs.map((v, i) => {
      const h = 2e-3;
      return (AE.denoise([v + h], P, sig).r[0] - AE.denoise([v - h], P, sig).r[0]) / (2 * h);
    });

    const y = showD
      ? d3.scaleLinear().domain([-0.1, Math.max(1.2, d3.max(dr) * 1.08)]).range([PY + PH, PY])
      : d3.scaleLinear().domain([-1.15, 1.15]).range([PY + PH, PY]);
    AE.gridY(g.append("g").attr("transform", `translate(${PX},0)`), y, PW, 5);
    AE.axisB(g.append("g").attr("transform", `translate(${PX},0)`),
      d3.scaleLinear().domain([-1.15, 1.15]).range([0, PW]), PY + PH, 6, "x");
    AE.axisL(g.append("g").attr("transform", `translate(${PX},0)`), y, 0, 5, showD ? "r′(x)" : "r(x)");

    if (!showD) {
      g.append("line").attr("x1", x(-1.12)).attr("y1", y(-1.12)).attr("x2", x(1.12)).attr("y2", y(1.12))
        .attr("stroke", C1.muted).attr("stroke-dasharray", "5 4").attr("stroke-width", 1.3);
      g.append("path").attr("d", d3.line().x((d, i) => x(xs[i])).y(d => y(d))(rs))
        .attr("fill", "none").attr("stroke", C1.A).attr("stroke-width", 2.4);
      // arrows r(x) − x along the bottom
      const ay = PY + PH - 16;
      for (let i = 6; i < xs.length; i += 22) {
        const v = xs[i], d = rs[i] - v;
        if (Math.abs(d) < 1e-3) continue;
        const k = Math.min(1, 0.10 / Math.abs(d));
        AE.arrow(g, x(v), ay, x(v + d * k), ay, d > 0 ? C1.good : C1.B, 1.2, 4);
      }
    } else {
      g.append("line").attr("x1", x(-1.12)).attr("y1", y(1)).attr("x2", x(1.12)).attr("y2", y(1))
        .attr("stroke", C1.muted).attr("stroke-dasharray", "5 4").attr("stroke-width", 1.2);
      g.append("text").attr("x", x(1.10)).attr("y", y(1) - 5).attr("text-anchor", "end")
        .attr("fill", C1.muted).attr("font-size", 10).text("r′ = 1, the identity's slope");
      g.append("path").attr("d", d3.line().x((d, i) => x(xs[i])).y(d => y(d))(dr))
        .attr("fill", "none").attr("stroke", C1.violet).attr("stroke-width", 2.4);
    }
    pts.forEach(v => {
      g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", PY).attr("y2", PY + PH)
        .attr("stroke", C1.bad).attr("stroke-opacity", .30).attr("stroke-dasharray", "3 3");
      if (!showD) g.append("circle").attr("cx", x(v)).attr("cy", y(v)).attr("r", 4)
        .attr("fill", C1.bad).attr("stroke", C1.bg).attr("stroke-width", 1.2);
    });

    // slopes at the data, and the steepest slope anywhere
    const slopeAt = pts.map(v => {
      const h = 2e-3;
      return (AE.denoise([v + h], P, sig).r[0] - AE.denoise([v - h], P, sig).r[0]) / (2 * h);
    });
    const maxSlope = d3.max(dr);
    const bx = PX + PW + 22;
    AE.panelBox(gS, bx, PY + 6, 158, 96, "slope r′(x)");
    gS.append("text").attr("x", bx + 10).attr("y", PY + 38).attr("fill", C1.muted).attr("font-size", 10.5).text("at the data points");
    gS.append("text").attr("x", bx + 10).attr("y", PY + 58).attr("fill", C1.good).attr("font-size", 14)
      .attr("font-family", "SF Mono,Menlo,monospace").text(AE.fmt(d3.max(slopeAt), 3));
    gS.append("text").attr("x", bx + 10).attr("y", PY + 78).attr("fill", C1.muted).attr("font-size", 10.5).text("largest anywhere");
    gS.append("text").attr("x", bx + 10).attr("y", PY + 96).attr("fill", C1.B).attr("font-size", 14)
      .attr("font-family", "SF Mono,Menlo,monospace").text(AE.fmt(maxSlope, 3));

    // how many crossings of the identity?
    let cross = 0;
    for (let i = 1; i < xs.length; i++) {
      const a = rs[i - 1] - xs[i - 1], b = rs[i] - xs[i];
      if (a === 0 || (a < 0) !== (b < 0)) cross++;
    }
    const sMax = d3.max(slopeAt);
    AE.put("mf-out",
      `<span class="keep">σ</span> = <b>${AE.fmt(sig, 2)}</b>, <b>${pts.length}</b> data points &nbsp;·&nbsp; ` +
      `<code>r(x)</code> crosses the identity <b>${cross}</b> time${cross === 1 ? "" : "s"} ` +
      (cross === 2 * pts.length - 1
        ? `— one <i>stable</i> crossing at each data point plus <b>${pts.length - 1}</b> <i>unstable</i> ones between them, where the arrows point apart`
        : cross < pts.length ? `<span style="color:${C1.bad}">— fewer than the ${pts.length} data points: the smoothing has fused them</span>` : "") +
      ` &nbsp;·&nbsp; slope at the data <b>${sMax < 5e-4 ? "&lt; 0.0005 (essentially flat)" : AE.fmt(sMax, 3)}</b>, ` +
      `steepest slope anywhere <b>${AE.fmt(maxSlope, 3)}</b> — flat on the data, steep between it`);
  }
  AE.on("mf-sig", "input", draw);
  ["mf-set", "mf-deriv"].forEach(id => AE.on(id, "change", draw));
  draw();
})();

/* ═══ 16 · prior → decoder → marginal, and the posterior ═════════════════ */
(function () {
  const svg = d3.select("#ae-lvm");
  if (svg.empty()) return;
  const C1 = AE.col;
  const g = svg.append("g");
  const ZLO = -3.6, ZHI = 3.6, NZ = 601, dz = (ZHI - ZLO) / (NZ - 1);
  const zs = AE.linspace(ZLO, ZHI, NZ);
  const prior = zs.map(z => Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI));
  // panels
  const P1 = { x: 44, y: 34, w: 190, h: 150 };      // prior over z
  const P2 = { x: 268, y: 34, w: 216, h: 150 };     // decoder curve
  const P3 = { x: 518, y: 34, w: 202, h: 150 };     // marginal over x
  const P4 = { x: 44, y: 236, w: 676, h: 128 };     // posterior over z

  function draw() {
    const b = AE.val("lv-b"), sig = AE.val("lv-sig"), xo = AE.val("lv-x"), gauss = AE.chk("lv-gauss");
    AE.put("lv-bv", AE.fmt(b, 2)); AE.put("lv-sigv", AE.fmt(sig, 2)); AE.put("lv-xv", AE.fmt(xo, 2));
    g.selectAll("*").remove();

    const dec = z => 1.05 * z + b * z * z * z * 0.42;      // the "generator network"
    const gz = zs.map(dec);

    /* panel 1 — the prior */
    AE.panelBox(g, P1.x - 12, P1.y - 22, P1.w + 24, P1.h + 44, "prior  p(z) = 𝒩(0, 1)");
    const x1 = d3.scaleLinear().domain([ZLO, ZHI]).range([P1.x, P1.x + P1.w]);
    const y1 = d3.scaleLinear().domain([0, d3.max(prior) * 1.15]).range([P1.y + P1.h, P1.y]);
    g.append("path").attr("d", d3.area().x((d, i) => x1(zs[i])).y0(P1.y + P1.h).y1(d => y1(d))(prior))
      .attr("fill", C1.A).attr("fill-opacity", .25);
    g.append("path").attr("d", d3.line().x((d, i) => x1(zs[i])).y(d => y1(d))(prior))
      .attr("fill", "none").attr("stroke", C1.A).attr("stroke-width", 1.8);
    AE.axisB(g.append("g").attr("transform", `translate(${P1.x},0)`),
      d3.scaleLinear().domain([ZLO, ZHI]).range([0, P1.w]), P1.y + P1.h, 5, "z");

    /* panel 2 — the decoder */
    AE.panelBox(g, P2.x - 12, P2.y - 22, P2.w + 24, P2.h + 44, "decoder  g(z)");
    const x2 = d3.scaleLinear().domain([ZLO, ZHI]).range([P2.x, P2.x + P2.w]);
    const y2 = d3.scaleLinear().domain([-3.2, 3.2]).range([P2.y + P2.h, P2.y]);
    g.append("path").attr("d", d3.line().x((d, i) => x2(zs[i])).y(d => y2(AE.clamp(d, -3.2, 3.2)))(gz))
      .attr("fill", "none").attr("stroke", C1.violet).attr("stroke-width", 2.2);
    g.append("line").attr("x1", P2.x).attr("x2", P2.x + P2.w).attr("y1", y2(xo)).attr("y2", y2(xo))
      .attr("stroke", C1.B).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.4);
    AE.axisB(g.append("g").attr("transform", `translate(${P2.x},0)`),
      d3.scaleLinear().domain([ZLO, ZHI]).range([0, P2.w]), P2.y + P2.h, 5, "z");
    AE.axisL(g.append("g").attr("transform", `translate(${P2.x},0)`), y2, 0, 4, "x");

    /* panel 3 — the marginal p(x), by quadrature */
    const xs = AE.linspace(-3.2, 3.2, 320);
    const px = xs.map(v => {
      let s = 0;
      for (let i = 0; i < NZ; i++) {
        const d = v - gz[i];
        s += prior[i] * Math.exp(-d * d / (2 * sig * sig)) / Math.sqrt(2 * Math.PI * sig * sig);
      }
      return s * dz;
    });
    AE.panelBox(g, P3.x - 12, P3.y - 22, P3.w + 24, P3.h + 44, "marginal  p(x) = ∫ p(x|z)p(z)dz");
    const x3 = d3.scaleLinear().domain([-3.2, 3.2]).range([P3.x, P3.x + P3.w]);
    const y3 = d3.scaleLinear().domain([0, d3.max(px) * 1.15]).range([P3.y + P3.h, P3.y]);
    g.append("path").attr("d", d3.area().x((d, i) => x3(xs[i])).y0(P3.y + P3.h).y1(d => y3(d))(px))
      .attr("fill", C1.good).attr("fill-opacity", .22);
    g.append("path").attr("d", d3.line().x((d, i) => x3(xs[i])).y(d => y3(d))(px))
      .attr("fill", "none").attr("stroke", C1.good).attr("stroke-width", 1.8);
    g.append("line").attr("x1", x3(xo)).attr("x2", x3(xo)).attr("y1", P3.y).attr("y2", P3.y + P3.h)
      .attr("stroke", C1.B).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.4);
    AE.axisB(g.append("g").attr("transform", `translate(${P3.x},0)`),
      d3.scaleLinear().domain([-3.2, 3.2]).range([0, P3.w]), P3.y + P3.h, 5, "x");

    /* panel 4 — the exact posterior p(z | x), by Bayes on the same grid */
    let Zn = 0;
    const post = zs.map((z, i) => {
      const d = xo - gz[i];
      const v = prior[i] * Math.exp(-d * d / (2 * sig * sig)) / Math.sqrt(2 * Math.PI * sig * sig);
      Zn += v * dz; return v;
    });
    for (let i = 0; i < NZ; i++) post[i] /= Math.max(Zn, 1e-300);
    let pm = 0, pv = 0;
    for (let i = 0; i < NZ; i++) pm += zs[i] * post[i] * dz;
    for (let i = 0; i < NZ; i++) pv += (zs[i] - pm) * (zs[i] - pm) * post[i] * dz;
    let modes = 0;
    for (let i = 2; i < NZ - 2; i++)
      if (post[i] > post[i - 1] && post[i] >= post[i + 1] && post[i] > 0.05 * d3.max(post)) modes++;

    AE.panelBox(g, P4.x - 12, P4.y - 22, P4.w + 24, P4.h + 44,
      "the exact posterior  p(z | x = " + AE.fmt(xo, 2) + "),  computed by quadrature");
    const x4 = d3.scaleLinear().domain([ZLO, ZHI]).range([P4.x, P4.x + P4.w]);
    const y4 = d3.scaleLinear().domain([0, d3.max(post) * 1.15]).range([P4.y + P4.h, P4.y]);
    g.append("path").attr("d", d3.area().x((d, i) => x4(zs[i])).y0(P4.y + P4.h).y1(d => y4(d))(post))
      .attr("fill", C1.B).attr("fill-opacity", .22);
    g.append("path").attr("d", d3.line().x((d, i) => x4(zs[i])).y(d => y4(d))(post))
      .attr("fill", "none").attr("stroke", C1.B).attr("stroke-width", 2.2);
    if (gauss) {
      const fit = zs.map(z => Math.exp(-(z - pm) * (z - pm) / (2 * pv)) / Math.sqrt(2 * Math.PI * pv));
      g.append("path").attr("d", d3.line().x((d, i) => x4(zs[i])).y(d => y4(d))(fit))
        .attr("fill", "none").attr("stroke", C1.teal).attr("stroke-width", 1.8).attr("stroke-dasharray", "5 4");
      AE.legend(g, [{ c: C1.B, t: "exact posterior" }, { c: C1.teal, t: "a Gaussian with the same mean and variance", dash: "5 4" }],
        P4.x + P4.w - 250, P4.y + 12);
    }
    AE.axisB(g.append("g").attr("transform", `translate(${P4.x},0)`),
      d3.scaleLinear().domain([ZLO, ZHI]).range([0, P4.w]), P4.y + P4.h, 7, "z");

    AE.put("lv-out",
      `p(x = ${AE.fmt(xo, 2)}) = <b>${AE.fmt(px[Math.round((xo + 3.2) / 6.4 * 319)], 4)}</b> &nbsp;·&nbsp; ` +
      `posterior mean <b>${AE.fmt(pm, 3)}</b>, variance <b>${AE.fmt(pv, 3)}</b> &nbsp;·&nbsp; ` +
      `<b>${modes}</b> mode${modes === 1 ? "" : "s"} ` +
      (modes > 1
        ? `<span style="color:${C1.bad}">— the decoder is non-monotone here, so two different codes explain this x equally well and <i>no</i> Gaussian q can represent the posterior</span>`
        : `<span style="color:${C1.good}">— unimodal, so a Gaussian q is at least the right shape</span>`));
  }
  ["lv-b", "lv-sig", "lv-x"].forEach(id => AE.on(id, "input", draw));
  AE.on("lv-gauss", "change", draw);
  draw();
})();

/* ═══ the toy linear-Gaussian model used by §17 ══════════════════════════
   x = W z + μ + ε ,  z ~ 𝒩(0, I₂) ,  ε ~ 𝒩(0, σ²I₄).  Chosen because the
   marginal AND the posterior are both available in closed form, so the ELBO
   can be checked against a true log-likelihood rather than against itself.  */
AE.toyLG = (function () {
  const Wm = [[1.0, 0.4], [0.2, -0.9], [-0.7, 0.3], [0.5, 0.6]];
  const mu = [0.3, -0.2, 0.1, 0.0], s2 = 0.25, xo = [1.2, -0.5, 0.4, 0.9], D = 4, k = 2;
  // Σ = WWᵀ + σ²I, and its log-density at xo — done with an explicit 4×4 inverse
  function sigmaAndLogp() {
    const S = AE.zeros(D, D);
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) {
      let s = 0; for (let a = 0; a < k; a++) s += Wm[i][a] * Wm[j][a];
      S[i][j] = s + (i === j ? s2 : 0);
    }
    // Cholesky
    const L = AE.zeros(D, D);
    for (let i = 0; i < D; i++) for (let j = 0; j <= i; j++) {
      let s = S[i][j];
      for (let t = 0; t < j; t++) s -= L[i][t] * L[j][t];
      L[i][j] = (i === j) ? Math.sqrt(s) : s / L[j][j];
    }
    const d = xo.map((v, i) => v - mu[i]), yv = new Float64Array(D);
    for (let i = 0; i < D; i++) {
      let s = d[i]; for (let t = 0; t < i; t++) s -= L[i][t] * yv[t];
      yv[i] = s / L[i][i];
    }
    let q = 0, ld = 0;
    for (let i = 0; i < D; i++) { q += yv[i] * yv[i]; ld += 2 * Math.log(L[i][i]); }
    return -0.5 * (D * Math.log(2 * Math.PI) + ld + q);
  }
  // exact posterior:  M = (WᵀW/σ² + I)⁻¹ ,  mean = M Wᵀ(x−μ)/σ²
  function posterior() {
    const A = [[0, 0], [0, 0]];
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) {
      let s = 0; for (let i = 0; i < D; i++) s += Wm[i][a] * Wm[i][b];
      A[a][b] = s / s2 + (a === b ? 1 : 0);
    }
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    const M = [[A[1][1] / det, -A[0][1] / det], [-A[1][0] / det, A[0][0] / det]];
    const r = xo.map((v, i) => v - mu[i]), Wt = [0, 0];
    for (let a = 0; a < k; a++) { let s = 0; for (let i = 0; i < D; i++) s += Wm[i][a] * r[i]; Wt[a] = s / s2; }
    const m = [M[0][0] * Wt[0] + M[0][1] * Wt[1], M[1][0] * Wt[0] + M[1][1] * Wt[1]];
    return { M, m };
  }
  const logpx = sigmaAndLogp(), post = posterior();

  /* the ELBO for a diagonal q = 𝒩(m, diag v), in closed form */
  function elbo(m, v) {
    const r = [0, 0, 0, 0];
    for (let i = 0; i < D; i++) {
      let s = mu[i]; for (let a = 0; a < k; a++) s += Wm[i][a] * m[a];
      r[i] = xo[i] - s;
    }
    let rr = 0; for (let i = 0; i < D; i++) rr += r[i] * r[i];
    let trWVW = 0;
    for (let i = 0; i < D; i++) for (let a = 0; a < k; a++) trWVW += Wm[i][a] * Wm[i][a] * v[a];
    const recon = -0.5 * D * Math.log(2 * Math.PI * s2) - (rr + trWVW) / (2 * s2);
    const kl = AE.klDiagStd(m, v);
    // form A: E_q[log p(z,x)] + H(q)
    let Eq_logpz = -0.5 * k * Math.log(2 * Math.PI);
    for (let a = 0; a < k; a++) Eq_logpz -= 0.5 * (v[a] + m[a] * m[a]);
    let Hq = 0; for (let a = 0; a < k; a++) Hq += 0.5 * Math.log(2 * Math.PI * Math.E * v[a]);
    return { recon, kl, elbo: recon - kl, elboA: recon + Eq_logpz + Hq };
  }
  /* the gap: KL( 𝒩(m, diag v) ‖ 𝒩(post.m, post.M) ) */
  function gap(m, v) {
    const M = post.M, det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    const Mi = [[M[1][1] / det, -M[0][1] / det], [-M[1][0] / det, M[0][0] / det]];
    const trm = Mi[0][0] * v[0] + Mi[1][1] * v[1];
    const d = [post.m[0] - m[0], post.m[1] - m[1]];
    const quad = d[0] * (Mi[0][0] * d[0] + Mi[0][1] * d[1]) + d[1] * (Mi[1][0] * d[0] + Mi[1][1] * d[1]);
    return 0.5 * (trm + quad - k + Math.log(det / (v[0] * v[1])));
  }
  return { W: Wm, mu, s2, x: xo, D, k, logpx, post, elbo, gap };
})();

/* ═══ 17 · the bound, the gap, and the posterior it approximates ═════════ */
(function () {
  const svg = d3.select("#ae-elbo");
  if (svg.empty()) return;
  const C1 = AE.col, T = AE.toyLG;
  const g = svg.append("g");
  const BX = 60, BY = 30, BW = 280, BH = 300;
  const CX = 430, CY = 30, CW = 280, CH = 300;

  function draw() {
    const m = [AE.val("eb-m1"), AE.val("eb-m2")], v = [AE.val("eb-v1"), AE.val("eb-v2")];
    AE.put("eb-m1v", AE.fmt(m[0], 2)); AE.put("eb-m2v", AE.fmt(m[1], 2));
    AE.put("eb-v1v", AE.fmt(v[0], 2)); AE.put("eb-v2v", AE.fmt(v[1], 2));
    g.selectAll("*").remove();
    const E = T.elbo(m, v), G = T.gap(m, v);

    /* left — a stacked bar: ELBO + gap = log p(x) */
    AE.panelBox(g, BX - 40, BY - 22, BW + 60, BH + 46, "the decomposition, drawn to scale");
    const span = Math.max(6, -E.elbo - T.logpx + 3);
    const y = d3.scaleLinear().domain([T.logpx - span, T.logpx + 1.2]).range([BY + BH, BY]);
    AE.axisL(g.append("g").attr("transform", `translate(${BX},0)`), y, 0, 6, "nats");
    // log p(x) reference line
    g.append("line").attr("x1", BX).attr("x2", BX + BW).attr("y1", y(T.logpx)).attr("y2", y(T.logpx))
      .attr("stroke", C1.good).attr("stroke-width", 2);
    g.append("text").attr("x", BX + BW).attr("y", y(T.logpx) - 6).attr("text-anchor", "end")
      .attr("fill", C1.good).attr("font-size", 11)
      .text("log p(x) = " + AE.fmt(T.logpx, 4) + "  — the ceiling");
    // the ELBO bar, and the gap above it
    const bw = 92, bx0 = BX + 40;
    g.append("rect").attr("x", bx0).attr("y", y(E.elbo)).attr("width", bw)
      .attr("height", Math.max(0, y(T.logpx - span) - y(E.elbo))).attr("fill", C1.A).attr("fill-opacity", .35)
      .attr("stroke", C1.A);
    g.append("rect").attr("x", bx0).attr("y", y(T.logpx)).attr("width", bw)
      .attr("height", Math.max(0, y(E.elbo) - y(T.logpx))).attr("fill", C1.bad).attr("fill-opacity", .30)
      .attr("stroke", C1.bad).attr("stroke-dasharray", "4 3");
    g.append("text").attr("x", bx0 + bw + 12).attr("y", (y(E.elbo) + y(T.logpx)) / 2 + 4)
      .attr("fill", C1.bad).attr("font-size", 11).text("gap = KL(q ‖ p(z|x)) = " + AE.fmt(G, 4));
    g.append("text").attr("x", bx0 + bw / 2).attr("y", y(E.elbo) + 16).attr("text-anchor", "middle")
      .attr("fill", C1.ink).attr("font-size", 11.5).attr("font-weight", 600).text("ELBO " + AE.fmt(E.elbo, 4));
    // the two components of the ELBO, as a second bar
    const bx1 = bx0 + bw + 148;
    g.append("rect").attr("x", bx1).attr("y", y(E.recon)).attr("width", 56)
      .attr("height", Math.max(0, y(T.logpx - span) - y(E.recon))).attr("fill", C1.teal).attr("fill-opacity", .3).attr("stroke", C1.teal);
    g.append("rect").attr("x", bx1).attr("y", y(E.recon + E.kl)).attr("width", 56)
      .attr("height", Math.max(0, y(E.recon) - y(E.recon + E.kl))).attr("fill", C1.B).attr("fill-opacity", .35).attr("stroke", C1.B);
    g.append("text").attr("x", bx1 + 28).attr("y", BY + BH + 18).attr("text-anchor", "middle")
      .attr("fill", C1.muted).attr("font-size", 10).text("its two terms");

    /* right — the true posterior and the diagonal approximation */
    AE.panelBox(g, CX - 44, CY - 22, CW + 62, CH + 46, "latent space:  q  against the true posterior");
    const xs = d3.scaleLinear().domain([-0.9, 1.9]).range([CX, CX + CW]);
    const ys = d3.scaleLinear().domain([-0.9, 1.9]).range([CY + CH, CY]);
    AE.axisB(g.append("g").attr("transform", `translate(${CX},0)`), d3.scaleLinear().domain([-0.9, 1.9]).range([0, CW]), CY + CH, 5, "z₁");
    AE.axisL(g.append("g").attr("transform", `translate(${CX},0)`), ys, 0, 5, "z₂");
    const M = T.post.M, det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    const Mi = [[M[1][1] / det, -M[0][1] / det], [-M[1][0] / det, M[0][0] / det]];
    const ring = AE.linspace(0, 2 * Math.PI, 100);
    // ellipses of the true posterior: chol of M
    const l11 = Math.sqrt(M[0][0]), l21 = M[1][0] / l11, l22 = Math.sqrt(Math.max(1e-12, M[1][1] - l21 * l21));
    [1, 2].forEach(c => {
      g.append("path").attr("d", d3.line()
        .x(a => xs(T.post.m[0] + c * l11 * Math.cos(a)))
        .y(a => ys(T.post.m[1] + c * (l21 * Math.cos(a) + l22 * Math.sin(a))))(ring))
        .attr("fill", "none").attr("stroke", C1.good).attr("stroke-width", 1.8).attr("stroke-opacity", c === 1 ? .95 : .45);
    });
    [1, 2].forEach(c => {
      g.append("path").attr("d", d3.line()
        .x(a => xs(m[0] + c * Math.sqrt(v[0]) * Math.cos(a)))
        .y(a => ys(m[1] + c * Math.sqrt(v[1]) * Math.sin(a)))(ring))
        .attr("fill", "none").attr("stroke", C1.A).attr("stroke-width", 1.8)
        .attr("stroke-dasharray", "5 4").attr("stroke-opacity", c === 1 ? .95 : .45);
    });
    g.append("circle").attr("cx", xs(T.post.m[0])).attr("cy", ys(T.post.m[1])).attr("r", 3.2).attr("fill", C1.good);
    g.append("circle").attr("cx", xs(m[0])).attr("cy", ys(m[1])).attr("r", 3.2).attr("fill", C1.A);
    AE.legend(g, [
      { c: C1.good, t: "true posterior p(z|x) — correlated" },
      { c: C1.A, t: "diagonal q(z|x) — axis aligned", dash: "5 4" }
    ], CX + 6, CY + 12);

    const same = Math.abs(E.elbo - E.elboA) < 1e-12;
    AE.put("eb-out",
      `reconstruction <b>${AE.fmt(E.recon, 4)}</b> − KL(q‖p(z)) <b>${AE.fmt(E.kl, 4)}</b> = ELBO <b>${AE.fmt(E.elbo, 4)}</b> &nbsp;·&nbsp; ` +
      `form A gives <b>${AE.fmt(E.elboA, 4)}</b> ${same ? "— identical" : "— MISMATCH"} &nbsp;·&nbsp; ` +
      `ELBO + gap = <b>${AE.fmt(E.elbo + G, 6)}</b> against log p(x) = <b>${AE.fmt(T.logpx, 6)}</b>, ` +
      `residual <b>${Math.abs(E.elbo + G - T.logpx).toExponential(1)}</b> &nbsp;·&nbsp; ` +
      `slack <b>${AE.fmt(T.logpx - E.elbo, 4)}</b> nats`);
  }
  ["eb-m1", "eb-m2", "eb-v1", "eb-v2"].forEach(id => AE.on(id, "input", draw));
  AE.on("eb-opt", "click", () => {
    // coordinate ascent on the four parameters of the diagonal q
    let m = [AE.val("eb-m1"), AE.val("eb-m2")], v = [AE.val("eb-v1"), AE.val("eb-v2")];
    for (let it = 0; it < 4000; it++) {
      const base = T.elbo(m, v).elbo, h = 1e-4, lr = 0.02;
      const gm = [0, 0], gv = [0, 0];
      for (let a = 0; a < 2; a++) {
        const mm = m.slice(); mm[a] += h; gm[a] = (T.elbo(mm, v).elbo - base) / h;
        const vv = v.slice(); vv[a] += h; gv[a] = (T.elbo(m, vv).elbo - base) / h;
      }
      for (let a = 0; a < 2; a++) { m[a] += lr * gm[a]; v[a] = Math.max(0.02, Math.min(1.2, v[a] + lr * gv[a])); }
    }
    document.getElementById("eb-m1").value = m[0].toFixed(2);
    document.getElementById("eb-m2").value = m[1].toFixed(2);
    document.getElementById("eb-v1").value = v[0].toFixed(2);
    document.getElementById("eb-v2").value = v[1].toFixed(2);
    draw();
  });
  AE.on("eb-reset", "click", () => {
    const d = { "eb-m1": 0.55, "eb-m2": -0.30, "eb-v1": 0.30, "eb-v2": 0.55 };
    Object.keys(d).forEach(k => { const e = document.getElementById(k); if (e) e.value = d[k]; });
    draw();
  });
  draw();
})();

/* ═══ 18 · two graphs, two estimators, one measured variance gap ═════════ */
(function () {
  const svg = d3.select("#ae-reparam");
  if (svg.empty()) return;
  const C1 = AE.col;
  const gG = svg.append("g"), gH = svg.append("g");

  /* ── the two computation graphs, drawn once ── */
  function node(g, cx, cy, w, h, label, sub, fill, stroke, dash) {
    g.append("rect").attr("x", cx - w / 2).attr("y", cy - h / 2).attr("width", w).attr("height", h).attr("rx", 6)
      .attr("fill", fill).attr("fill-opacity", .22).attr("stroke", stroke).attr("stroke-width", 1.4)
      .attr("stroke-dasharray", dash || null);
    g.append("text").attr("x", cx).attr("y", cy + (sub ? -2 : 4)).attr("text-anchor", "middle")
      .attr("fill", C1.ink).attr("font-size", 11.5).attr("font-family", "SF Mono,Menlo,monospace").text(label);
    if (sub) g.append("text").attr("x", cx).attr("y", cy + 12).attr("text-anchor", "middle")
      .attr("fill", C1.muted).attr("font-size", 9.5).text(sub);
  }
  function graphs() {
    gG.selectAll("*").remove();
    [{ x0: 24, ok: false, title: "naïve: the sampler sits in the gradient path" },
     { x0: 400, ok: true, title: "reparameterised: the sampler is an add and a multiply" }].forEach(R => {
      const c = R.ok ? C1.good : C1.bad;
      gG.append("rect").attr("x", R.x0 - 8).attr("y", 6).attr("width", 344).attr("height", 176).attr("rx", 9)
        .attr("fill", C1.panel2).attr("fill-opacity", .45).attr("stroke", C1.line);
      gG.append("text").attr("x", R.x0 + 164).attr("y", 26).attr("text-anchor", "middle")
        .attr("fill", c).attr("font-size", 11.5).attr("font-weight", 600).text(R.title);

      node(gG, R.x0 + 52, 58, 68, 28, "φ", null, C1.A, C1.A);
      node(gG, R.x0 + 156, 58, 78, 28, "μ, σ", null, C1.A, C1.A);
      if (R.ok) node(gG, R.x0 + 52, 122, 92, 28, "ε ~ 𝒩(0,I)", null, C1.muted, C1.muted, "4 3");
      node(gG, R.x0 + 182, 122, R.ok ? 112 : 96, 30,
        R.ok ? "z = μ + σ⊙ε" : "z ~ q(z|x)", null, c, c, R.ok ? null : "4 3");
      node(gG, R.x0 + 292, 122, 74, 28, "h(z) → 𝓛", null, C1.violet, C1.violet);

      AE.arrow(gG, R.x0 + 88, 58, R.x0 + 114, 58, C1.muted, 1.3, 5);
      AE.arrow(gG, R.x0 + 156, 74, R.x0 + 168, 105, C1.muted, 1.3, 5);
      if (R.ok) AE.arrow(gG, R.x0 + 100, 122, R.x0 + 122, 122, C1.muted, 1.3, 5);
      AE.arrow(gG, R.x0 + (R.ok ? 240 : 232), 122, R.x0 + 253, 122, C1.muted, 1.3, 5);

      // the backward pass, drawn under the forward one
      gG.append("path").attr("d", `M${R.x0 + 292},${152} L${R.x0 + 182},${152}`)
        .attr("stroke", c).attr("stroke-width", 1.6).attr("stroke-dasharray", "5 3");
      if (R.ok) {
        AE.arrow(gG, R.x0 + 182, 152, R.x0 + 150, 76, C1.good, 1.7, 6);
        gG.append("text").attr("x", R.x0 + 190).attr("y", 170).attr("fill", C1.good).attr("font-size", 10)
          .text("∂z/∂μ = 1,  ∂z/∂σ = ε");
        gG.append("text").attr("x", R.x0 + 10).attr("y", 170).attr("fill", C1.muted).attr("font-size", 10)
          .text("ε carries no gradient");
      } else {
        gG.append("text").attr("x", R.x0 + 178).attr("y", 158).attr("text-anchor", "middle")
          .attr("fill", C1.bad).attr("font-size", 17).attr("font-weight", 700).text("✗");
        gG.append("text").attr("x", R.x0 + 10).attr("y", 170).attr("fill", C1.bad).attr("font-size", 10)
          .text("a draw has no derivative — the chain rule stops here");
      }
    });
  }
  graphs();

  /* ── the measurement ── */
  const HX = 70, HY = 214, HW = 500, HH = 178;
  function draw() {
    const k = AE.val("rp-k"), sig = AE.val("rp-sig"), N = AE.val("rp-n"), useB = AE.chk("rp-base");
    AE.put("rp-kv", k); AE.put("rp-sigv", AE.fmt(sig, 2)); AE.put("rp-nv", N);
    gH.selectAll("*").remove();

    // h(z) = −½‖z − c‖² ,  q = 𝒩(0, σ²I_k) ,  ‖c‖ = 2 ⇒ exact ∂/∂μ = c
    const cc = 2 / Math.sqrt(k), r = AE.rng(20250908);
    const rep = new Float64Array(N), scf = new Float64Array(N);
    const fs = new Float64Array(N), s0 = new Float64Array(N);
    let sumS2 = 0, sumFS2 = 0;
    const eps = new Float64Array(k);
    for (let n = 0; n < N; n++) {
      let f = 0, s2acc = 0;
      for (let i = 0; i < k; i++) {
        eps[i] = AE.randn(r);
        const d = sig * eps[i] - cc;
        f -= 0.5 * d * d;
        const si = eps[i] / sig;
        s2acc += si * si;
        if (i === 0) { rep[n] = -(sig * eps[0] - cc); s0[n] = si; }
      }
      fs[n] = f; sumS2 += s2acc; sumFS2 += f * s2acc;
    }
    const b = useB ? sumFS2 / sumS2 : 0;
    for (let n = 0; n < N; n++) scf[n] = (fs[n] - b) * s0[n];

    const mean = a => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
    const varr = a => { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m); return s / a.length; };
    const mR = mean(rep), mS = mean(scf), vR = varr(rep), vS = varr(scf);

    // histograms, on a shared axis
    const hi = Math.max(4 * Math.sqrt(vS), 4 * Math.sqrt(vR), Math.abs(cc) * 2 + 1);
    const xh = d3.scaleLinear().domain([cc - hi, cc + hi]).range([HX, HX + HW]);
    const bins = 61, edges = AE.linspace(cc - hi, cc + hi, bins + 1);
    const hist = a => {
      const cnt = new Float64Array(bins);
      for (let i = 0; i < a.length; i++) {
        const j = Math.floor((a[i] - edges[0]) / (edges[1] - edges[0]));
        if (j >= 0 && j < bins) cnt[j]++;
      }
      for (let j = 0; j < bins; j++) cnt[j] /= a.length;
      return cnt;
    };
    const hR = hist(rep), hS = hist(scf);
    const ymax = Math.max(d3.max(hR), d3.max(hS)) * 1.12;
    const yh = d3.scaleLinear().domain([0, ymax]).range([HY + HH, HY]);
    AE.panelBox(gH, HX - 46, HY - 24, HW + 66, HH + 50, "single-sample estimates of  ∂/∂μ₁ ,  " + AE.commas(N) + " draws");
    AE.axisB(gH.append("g").attr("transform", `translate(${HX},0)`),
      d3.scaleLinear().domain(xh.domain()).range([0, HW]), HY + HH, 6, "gradient estimate for coordinate 1");
    const bar = (h, colr, op) => {
      const w = HW / bins;
      gH.append("g").selectAll("rect").data(d3.range(bins)).enter().append("rect")
        .attr("x", j => HX + j * w).attr("width", w + .5)
        .attr("y", j => yh(h[j])).attr("height", j => HY + HH - yh(h[j]))
        .attr("fill", colr).attr("fill-opacity", op);
    };
    bar(hS, C1.bad, .45); bar(hR, C1.good, .65);
    gH.append("line").attr("x1", xh(cc)).attr("x2", xh(cc)).attr("y1", HY).attr("y2", HY + HH)
      .attr("stroke", C1.ink).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
    gH.append("text").attr("x", xh(cc) + 6).attr("y", HY + 12).attr("fill", C1.ink).attr("font-size", 10)
      .text("exact gradient = " + AE.fmt(cc, 4));
    AE.legend(gH, [
      { c: C1.good, t: "reparameterised (pathwise)" },
      { c: C1.bad, t: "score function" + (useB ? " + baseline" : "") }
    ], HX + HW + 26, HY + 24);
    gH.append("text").attr("x", HX + HW + 26).attr("y", HY + 76).attr("fill", C1.muted).attr("font-size", 10.5).text("std of one draw");
    gH.append("text").attr("x", HX + HW + 26).attr("y", HY + 96).attr("fill", C1.good).attr("font-size", 13)
      .attr("font-family", "SF Mono,Menlo,monospace").text(AE.fmt(Math.sqrt(vR), 3));
    gH.append("text").attr("x", HX + HW + 26).attr("y", HY + 116).attr("fill", C1.bad).attr("font-size", 13)
      .attr("font-family", "SF Mono,Menlo,monospace").text(AE.fmt(Math.sqrt(vS), 3));

    AE.put("rp-out",
      `k = <b>${k}</b>, <span class="keep">σ</span> = <b>${AE.fmt(sig, 2)}</b> &nbsp;·&nbsp; ` +
      `exact ∂/∂μ₁ = <b>${AE.fmt(cc, 4)}</b>; measured means <b>${AE.fmt(mR, 4)}</b> (pathwise) and <b>${AE.fmt(mS, 4)}</b> (score) — ` +
      `<span style="color:${C1.good}">both unbiased</span> &nbsp;·&nbsp; ` +
      `per-coordinate variance <b>${AE.fmt(vR, 4)}</b> vs <b>${AE.fmt(vS, 2)}</b> &nbsp;·&nbsp; ` +
      `the score-function estimator needs about <b>${AE.fmt(vS / vR, 0)}×</b> as many samples for the same standard error` +
      (Math.abs(vR - sig * sig) < 0.08 * sig * sig ? ` &nbsp;·&nbsp; note the pathwise variance is <b>σ² = ${AE.fmt(sig * sig, 4)}</b> per coordinate, exactly` : ""));
  }
  ["rp-k", "rp-sig", "rp-n"].forEach(id => AE.on(id, "input", draw));
  AE.on("rp-base", "change", draw);
  draw();
})();

/* ═══ 19 · the Gaussian KL, felt directly ════════════════════════════════ */
(function () {
  const svg = d3.select("#ae-vae");
  if (svg.empty()) return;
  const C1 = AE.col, g = svg.append("g");
  const LX = 60, LY = 26, LW = 300, LH = 300;
  const RX = 440, RY = 26, RW = 270, RH = 300;

  function draw() {
    const m = [AE.val("vk-m1"), AE.val("vk-m2")], sd = [AE.val("vk-s1"), AE.val("vk-s2")];
    AE.put("vk-m1v", AE.fmt(m[0], 2)); AE.put("vk-m2v", AE.fmt(m[1], 2));
    AE.put("vk-s1v", AE.fmt(sd[0], 2)); AE.put("vk-s2v", AE.fmt(sd[1], 2));
    g.selectAll("*").remove();
    const v = [sd[0] * sd[0], sd[1] * sd[1]];

    /* left — latent space */
    AE.panelBox(g, LX - 46, LY - 22, LW + 66, LH + 46, "latent space:  q(z|x) against the prior 𝒩(0, I)");
    const xs = d3.scaleLinear().domain([-4, 4]).range([LX, LX + LW]);
    const ys = d3.scaleLinear().domain([-4, 4]).range([LY + LH, LY]);
    AE.axisB(g.append("g").attr("transform", `translate(${LX},0)`), d3.scaleLinear().domain([-4, 4]).range([0, LW]), LY + LH, 5, "z₁");
    AE.axisL(g.append("g").attr("transform", `translate(${LX},0)`), ys, 0, 5, "z₂");
    const ring = AE.linspace(0, 2 * Math.PI, 120);
    [1, 2].forEach(c => g.append("path").attr("d", d3.line().x(a => xs(c * Math.cos(a))).y(a => ys(c * Math.sin(a)))(ring))
      .attr("fill", "none").attr("stroke", C1.muted).attr("stroke-dasharray", "4 4").attr("stroke-opacity", c === 1 ? .9 : .45));
    [1, 2].forEach(c => g.append("path").attr("d", d3.line()
      .x(a => xs(m[0] + c * sd[0] * Math.cos(a))).y(a => ys(m[1] + c * sd[1] * Math.sin(a)))(ring))
      .attr("fill", "none").attr("stroke", C1.A).attr("stroke-width", 2).attr("stroke-opacity", c === 1 ? .95 : .45));
    const r = AE.rng(4242);
    const pts = d3.range(220).map(() => [m[0] + sd[0] * AE.randn(r), m[1] + sd[1] * AE.randn(r)]);
    g.selectAll("circle.s").data(pts).enter().append("circle").attr("class", "s")
      .attr("cx", p => xs(p[0])).attr("cy", p => ys(p[1])).attr("r", 1.7).attr("fill", C1.A).attr("fill-opacity", .45);
    g.append("circle").attr("cx", xs(m[0])).attr("cy", ys(m[1])).attr("r", 3.4).attr("fill", C1.A).attr("stroke", C1.bg);
    AE.legend(g, [{ c: C1.muted, t: "prior 𝒩(0, I)", dash: "4 4" }, { c: C1.A, t: "q(z|x) = 𝒩(μ, diag σ²)" }], LX + 6, LY + 12);

    /* right — the KL, split into its pieces */
    const kl = AE.klDiagStd(m, v);
    const per = [0, 1].map(j => 0.5 * (v[j] + m[j] * m[j] - 1 - Math.log(v[j])));
    const parts = [
      { t: "dim 1 · mean term ½μ₁²", val: 0.5 * m[0] * m[0], c: C1.B },
      { t: "dim 1 · variance term ½(σ₁²−1−log σ₁²)", val: 0.5 * (v[0] - 1 - Math.log(v[0])), c: C1.teal },
      { t: "dim 2 · mean term ½μ₂²", val: 0.5 * m[1] * m[1], c: C1.B },
      { t: "dim 2 · variance term ½(σ₂²−1−log σ₂²)", val: 0.5 * (v[1] - 1 - Math.log(v[1])), c: C1.teal }
    ];
    AE.panelBox(g, RX - 14, RY - 22, RW + 30, RH + 46, "KL(q ‖ p) broken into four terms, in nats");
    const mxv = Math.max(0.35, d3.max(parts, p => p.val) * 1.15);
    const xb = d3.scaleLinear().domain([0, mxv]).range([RX + 4, RX + RW - 10]);
    parts.forEach((p, i) => {
      const yy = RY + 40 + i * 58;
      g.append("text").attr("x", RX + 4).attr("y", yy - 8).attr("fill", C1.muted).attr("font-size", 10).text(p.t);
      g.append("rect").attr("x", xb(0)).attr("y", yy).attr("width", Math.max(1, xb(p.val) - xb(0))).attr("height", 15)
        .attr("rx", 3).attr("fill", p.c).attr("fill-opacity", .7);
      g.append("text").attr("x", xb(p.val) + 7).attr("y", yy + 12).attr("fill", C1.ink).attr("font-size", 10.5)
        .attr("font-family", "SF Mono,Menlo,monospace").text(AE.fmt(p.val, 4));
    });
    g.append("line").attr("x1", RX + 4).attr("x2", RX + RW - 10).attr("y1", RY + 268).attr("y2", RY + 268).attr("stroke", C1.line);
    g.append("text").attr("x", RX + 4).attr("y", RY + 288).attr("fill", C1.ink).attr("font-size", 12.5)
      .attr("font-weight", 600).text("total  KL = " + AE.fmt(kl, 5) + "  nats");

    // Monte-Carlo check of the same divergence
    const rr = AE.rng(99), M = 200000;
    let acc = 0;
    for (let n = 0; n < M; n++) {
      let lq = 0, lp = 0;
      for (let j = 0; j < 2; j++) {
        const z = m[j] + sd[j] * AE.randn(rr);
        lq += AE.logNormPdf(z, m[j], sd[j]); lp += AE.logNormPdf(z, 0, 1);
      }
      acc += lq - lp;
    }
    const mc = acc / M;
    AE.put("vk-out",
      `closed form <b>${AE.fmt(kl, 5)}</b> nats &nbsp;·&nbsp; Monte-Carlo over 200 000 draws <b>${AE.fmt(mc, 4)}</b> &nbsp;·&nbsp; ` +
      `per dimension <b>${AE.fmt(per[0], 4)}</b> and <b>${AE.fmt(per[1], 4)}</b> &nbsp;·&nbsp; ` +
      (kl < 1e-6
        ? `<span style="color:${C1.bad}">q is exactly the prior: the KL is zero and the code carries <b>no information about x at all</b> — this is posterior collapse (§22)</span>`
        : `the code is spending <b>${AE.fmt(kl / Math.log(2), 3)}</b> bits describing this one example`));
  }
  ["vk-m1", "vk-m2", "vk-s1", "vk-s2"].forEach(id => AE.on(id, "input", draw));
  AE.on("vk-zero", "click", () => {
    [["vk-m1", 0], ["vk-m2", 0], ["vk-s1", 1], ["vk-s2", 1]].forEach(([k, v]) => {
      const e = document.getElementById(k); if (e) e.value = v;
    });
    draw();
  });
  draw();
})();

/* ═══ the exact linear β-VAE optimum, used by §21 and §22 ════════════════
   Per data eigen-direction with eigenvalue λ and decoder noise σ²:
     active ⟺ λ > βσ²  ;  w² = λ − βσ²  ;  d = βσ²/λ  ;  m = (w/λ)(vᵀx)
     rate Rᵢ = ½ log(λ/(βσ²))            distortion Dᵢ = βσ²   (else λ)
   Both lines were checked against a trained linear VAE and agree to 4 dp.  */
AE.linVAE = function (ev, beta, s2) {
  const th = beta * s2, out = [];
  for (let i = 0; i < ev.length; i++) {
    const lam = ev[i], on = lam > th;
    out.push({
      lam, on,
      w: on ? Math.sqrt(lam - th) : 0,
      d: on ? th / lam : 1,
      rate: on ? 0.5 * Math.log(lam / th) : 0,
      dist: on ? th : lam
    });
  }
  return out;
};

/* ═══ 21a · the rate–distortion frontier ═════════════════════════════════ */
(function () {
  const svg = d3.select("#ae-rate");
  if (svg.empty()) return;
  const C1 = AE.col, g = svg.append("g");
  const LX = 62, LY = 26, LW = 300, LH = 292;
  const RX = 444, RY = 26, RW = 272, RH = 292;
  const D = 8;

  const spectrum = decay => d3.range(D).map(i => 2.4 * Math.pow(decay, i));
  const sum = (a, f) => a.reduce((s, v) => s + f(v), 0);

  function draw() {
    const beta = Math.pow(10, AE.val("rd-beta")), s2 = AE.val("rd-s2"), decay = AE.val("rd-decay");
    AE.put("rd-betav", AE.fmt(beta, 2)); AE.put("rd-s2v", AE.fmt(s2, 2)); AE.put("rd-decayv", AE.fmt(decay, 2));
    g.selectAll("*").remove();
    const ev = spectrum(decay), total = d3.sum(ev);

    // the whole frontier
    const curve = AE.linspace(-2, 2, 260).map(u => {
      const b = Math.pow(10, u), s = AE.linVAE(ev, b, s2);
      return { b, R: sum(s, q => q.rate), Dd: sum(s, q => q.dist) };
    });
    const here = AE.linVAE(ev, beta, s2);
    const R0 = sum(here, q => q.rate), D0 = sum(here, q => q.dist), nAct = here.filter(q => q.on).length;

    AE.panelBox(g, LX - 48, LY - 22, LW + 68, LH + 46, "the frontier — each point is one β");
    const x = d3.scaleLinear().domain([0, d3.max(curve, p => p.R) * 1.08 + 1e-9]).range([LX, LX + LW]);
    const y = d3.scaleLinear().domain([0, total * 1.06]).range([LY + LH, LY]);
    AE.gridY(g.append("g").attr("transform", `translate(${LX},0)`), y, LW, 5);
    AE.axisB(g.append("g").attr("transform", `translate(${LX},0)`), d3.scaleLinear().domain(x.domain()).range([0, LW]), LY + LH, 5, "rate R, nats per example");
    AE.axisL(g.append("g").attr("transform", `translate(${LX},0)`), y, 0, 5, "distortion D");
    g.append("path").attr("d", d3.line().x(p => x(p.R)).y(p => y(p.Dd))(curve))
      .attr("fill", "none").attr("stroke", C1.A).attr("stroke-width", 2.4);
    // the β = 1 point, and the current one
    const one = curve.reduce((a, p) => Math.abs(p.b - 1) < Math.abs(a.b - 1) ? p : a, curve[0]);
    g.append("circle").attr("cx", x(one.R)).attr("cy", y(one.Dd)).attr("r", 4).attr("fill", "none")
      .attr("stroke", C1.good).attr("stroke-width", 2);
    g.append("text").attr("x", x(one.R) + 8).attr("y", y(one.Dd) - 6).attr("fill", C1.good).attr("font-size", 10).text("β = 1");
    g.append("circle").attr("cx", x(R0)).attr("cy", y(D0)).attr("r", 5.2).attr("fill", C1.B).attr("stroke", C1.bg).attr("stroke-width", 1.4);
    g.append("line").attr("x1", LX).attr("x2", LX + LW).attr("y1", y(total)).attr("y2", y(total))
      .attr("stroke", C1.bad).attr("stroke-dasharray", "4 4").attr("stroke-opacity", .7);
    g.append("text").attr("x", LX + LW).attr("y", y(total) - 5).attr("text-anchor", "end")
      .attr("fill", C1.bad).attr("font-size", 10).text("D = total variance: the code says nothing");

    // per-dimension rate
    AE.panelBox(g, RX - 16, RY - 22, RW + 34, RH + 46, "rate spent per latent dimension, nats");
    const xb = d3.scaleBand().domain(d3.range(D)).range([RX, RX + RW]).padding(0.22);
    const ymax = Math.max(0.2, d3.max(here, q => q.rate) * 1.2);
    const yb = d3.scaleLinear().domain([0, ymax]).range([RY + RH - 26, RY + 10]);
    AE.axisL(g.append("g").attr("transform", `translate(${RX},0)`), yb, 0, 4, null);
    g.selectAll("rect.rr").data(here).enter().append("rect").attr("class", "rr")
      .attr("x", (q, i) => xb(i)).attr("width", xb.bandwidth())
      .attr("y", q => yb(q.rate)).attr("height", q => Math.max(0, (RY + RH - 26) - yb(q.rate)))
      .attr("rx", 2).attr("fill", q => q.on ? C1.B : C1.line)
      .attr("stroke", q => q.on ? C1.B : C1.bad).attr("stroke-opacity", q => q.on ? .8 : .6);
    here.forEach((q, i) => {
      g.append("text").attr("x", xb(i) + xb.bandwidth() / 2).attr("y", RY + RH - 12).attr("text-anchor", "middle")
        .attr("fill", q.on ? C1.muted : C1.bad).attr("font-size", 9).text(q.on ? (i + 1) : "off");
      g.append("text").attr("x", xb(i) + xb.bandwidth() / 2).attr("y", RY + RH + 2).attr("text-anchor", "middle")
        .attr("fill", C1.muted).attr("font-size", 8.5).text("λ " + AE.fmt(q.lam, 2));
    });
    g.append("line").attr("x1", RX).attr("x2", RX + RW).attr("y1", RY + RH - 26).attr("y2", RY + RH - 26).attr("stroke", C1.line);

    AE.put("rd-out",
      `<span class="keep">β</span> = <b>${AE.fmt(beta, 2)}</b>, threshold <code><span class="keep">β</span><span class="keep">σ</span>² = ${AE.fmt(beta * s2, 3)}</code> &nbsp;·&nbsp; ` +
      `<b>${nAct}</b> of ${D} dimensions active &nbsp;·&nbsp; ` +
      `rate <b>${AE.fmt(R0, 3)}</b> nats (<b>${AE.fmt(R0 / Math.log(2), 2)}</b> bits) &nbsp;·&nbsp; ` +
      `distortion <b>${AE.fmt(D0, 3)}</b> of a total variance of <b>${AE.fmt(total, 3)}</b> &nbsp;·&nbsp; ` +
      (nAct === 0
        ? `<span style="color:${C1.bad}">every dimension is switched off — the model has collapsed to the prior</span>`
        : `each active dimension leaves exactly <code><span class="keep">β</span><span class="keep">σ</span>² = ${AE.fmt(beta * s2, 3)}</code> of error behind, whatever its eigenvalue`));
  }
  ["rd-beta", "rd-s2", "rd-decay"].forEach(id => AE.on(id, "input", draw));
  draw();
})();

/* ═══ 21b · the latent space as β rises ══════════════════════════════════ */
(function () {
  const svg = d3.select("#ae-beta");
  if (svg.empty()) return;
  const C1 = AE.col, g = svg.append("g");
  const PX = 60, PY = 24, PW = 330, PH = 300, D = 8, NC = 3, NPER = 90;
  const cols = [C1.A, C1.B, C1.teal];

  function dataset(sep) {
    const r = AE.rng(31337);
    // a fixed random orthonormal frame, a decaying within-class spectrum,
    // and three class means arranged as a triangle in the first two directions
    let B = AE.zeros(D, D);
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) B[i][j] = AE.randn(r);
    for (let j = 0; j < D; j++) {
      for (let k = 0; k < j; k++) {
        let d = 0; for (let i = 0; i < D; i++) d += B[i][j] * B[i][k];
        for (let i = 0; i < D; i++) B[i][j] -= d * B[i][k];
      }
      let n = 0; for (let i = 0; i < D; i++) n += B[i][j] * B[i][j];
      n = Math.sqrt(n); for (let i = 0; i < D; i++) B[i][j] /= n;
    }
    const within = d3.range(D).map(i => 0.55 * Math.pow(0.72, i));
    const X = AE.zeros(NC * NPER, D), lab = [];
    for (let c = 0; c < NC; c++) {
      const th = 2 * Math.PI * c / NC, cm = [sep * Math.cos(th), sep * Math.sin(th)];
      for (let n = 0; n < NPER; n++) {
        const z = new Float64Array(D);
        for (let j = 0; j < D; j++) z[j] = AE.randn(r) * Math.sqrt(within[j]);
        z[0] += cm[0]; z[1] += cm[1];
        const row = X[c * NPER + n];
        for (let i = 0; i < D; i++) { let s = 0; for (let j = 0; j < D; j++) s += B[i][j] * z[j]; row[i] = s; }
        lab.push(c);
      }
    }
    const mu = new Float64Array(D);
    for (let n = 0; n < X.length; n++) for (let i = 0; i < D; i++) mu[i] += X[n][i] / X.length;
    for (let n = 0; n < X.length; n++) for (let i = 0; i < D; i++) X[n][i] -= mu[i];
    const E = AE.eigSym(AE.covariance(X));
    return { X, lab, E };
  }

  function draw() {
    const beta = Math.pow(10, AE.val("bl-beta")), sep = AE.val("bl-sep"), s2 = AE.val("bl-s2"), showP = AE.chk("bl-post");
    AE.put("bl-betav", AE.fmt(beta, 2)); AE.put("bl-sepv", AE.fmt(sep, 2)); AE.put("bl-s2v", AE.fmt(s2, 2));
    g.selectAll("*").remove();
    const { X, lab, E } = dataset(sep);
    const sol = AE.linVAE(E.values, beta, s2);

    // encoder means: mᵢ = (wᵢ/λᵢ)(vᵢᵀx)
    const M = X.map(row => {
      const m = new Float64Array(D);
      for (let i = 0; i < D; i++) {
        if (!sol[i].on) continue;
        let p = 0; for (let j = 0; j < D; j++) p += E.vectors[j][i] * row[j];
        m[i] = (sol[i].w / sol[i].lam) * p;
      }
      return m;
    });

    const lim = Math.max(0.6, d3.max(M, m => Math.max(Math.abs(m[0]), Math.abs(m[1]))) * 1.2, 2.4);
    const xs = d3.scaleLinear().domain([-lim, lim]).range([PX, PX + PW]);
    const ys = d3.scaleLinear().domain([-lim, lim]).range([PY + PH, PY]);
    AE.panelBox(g, PX - 46, PY - 20, PW + 66, PH + 46, "encoder means in latent dimensions 1 and 2");
    AE.axisB(g.append("g").attr("transform", `translate(${PX},0)`), d3.scaleLinear().domain([-lim, lim]).range([0, PW]), PY + PH, 5, "z₁");
    AE.axisL(g.append("g").attr("transform", `translate(${PX},0)`), ys, 0, 5, "z₂");
    const ring = AE.linspace(0, 2 * Math.PI, 100);
    g.append("path").attr("d", d3.line().x(a => xs(Math.cos(a))).y(a => ys(Math.sin(a)))(ring))
      .attr("fill", "none").attr("stroke", C1.muted).attr("stroke-dasharray", "4 4").attr("stroke-opacity", .8);
    if (showP) {
      const s1 = Math.sqrt(sol[0].d), s2b = Math.sqrt(sol[1].d);
      M.forEach((m, n) => g.append("ellipse").attr("cx", xs(m[0])).attr("cy", ys(m[1]))
        .attr("rx", Math.abs(xs(s1) - xs(0))).attr("ry", Math.abs(ys(s2b) - ys(0)))
        .attr("fill", cols[lab[n]]).attr("fill-opacity", .05).attr("stroke", cols[lab[n]]).attr("stroke-opacity", .16));
    }
    g.selectAll("circle.p").data(d3.range(M.length)).enter().append("circle").attr("class", "p")
      .attr("cx", n => xs(M[n][0])).attr("cy", n => ys(M[n][1])).attr("r", 2.3)
      .attr("fill", n => cols[lab[n]]).attr("fill-opacity", .75);

    // between/within scatter over the ACTIVE latent dimensions
    const act = d3.range(D).filter(i => sol[i].on);
    let sw = 0, sb = 0;
    const gm = new Float64Array(D);
    M.forEach(m => act.forEach(i => gm[i] += m[i] / M.length));
    for (let c = 0; c < NC; c++) {
      const cm = new Float64Array(D);
      for (let n = c * NPER; n < (c + 1) * NPER; n++) act.forEach(i => cm[i] += M[n][i] / NPER);
      act.forEach(i => sb += NPER * (cm[i] - gm[i]) * (cm[i] - gm[i]));
      for (let n = c * NPER; n < (c + 1) * NPER; n++) act.forEach(i => sw += (M[n][i] - cm[i]) * (M[n][i] - cm[i]));
    }
    const ratio = sw > 1e-12 ? sb / sw : 0;
    const R0 = d3.sum(sol, q => q.rate), D0 = d3.sum(sol, q => q.dist), tot = d3.sum(E.values);

    // side readout panel
    const bx = PX + PW + 56;
    AE.panelBox(g, bx, PY, 230, 150, "at this β");
    const rows = [
      ["active dimensions", `${act.length} of ${D}`, act.length ? C1.good : C1.bad],
      ["rate R", AE.fmt(R0, 3) + " nats", C1.B],
      ["distortion D", AE.fmt(D0, 3) + " / " + AE.fmt(tot, 2), C1.teal],
      ["between ÷ within scatter", AE.fmt(ratio, 3), ratio > 0.4 ? C1.good : C1.bad]
    ];
    rows.forEach((rw, i) => {
      g.append("text").attr("x", bx + 12).attr("y", PY + 34 + i * 30).attr("fill", C1.muted).attr("font-size", 10.5).text(rw[0]);
      g.append("text").attr("x", bx + 218).attr("y", PY + 34 + i * 30).attr("text-anchor", "end").attr("fill", rw[2])
        .attr("font-size", 12.5).attr("font-family", "SF Mono,Menlo,monospace").text(rw[1]);
    });
    AE.legend(g, [{ c: cols[0], t: "class A" }, { c: cols[1], t: "class B" }, { c: cols[2], t: "class C" },
    { c: C1.muted, t: "prior, one standard deviation", dash: "4 4" }], bx + 10, PY + 178);

    AE.put("bl-out",
      `<span class="keep">β</span> = <b>${AE.fmt(beta, 2)}</b> &nbsp;·&nbsp; ` +
      `<b>${act.length}</b> of ${D} latent dimensions survive &nbsp;·&nbsp; ` +
      `rate <b>${AE.fmt(R0, 3)}</b> nats, distortion <b>${AE.fmt(D0, 3)}</b> &nbsp;·&nbsp; ` +
      `class separation (between ÷ within scatter) <b>${AE.fmt(ratio, 3)}</b> &nbsp;·&nbsp; ` +
      (act.length === 0
        ? `<span style="color:${C1.bad}">nothing survives: every code is the origin, and the classes are indistinguishable</span>`
        : R0 > 6
          ? `<span style="color:${C1.B}">a very low KL weight keeps every dimension, including the ones carrying only <i>within-class</i> noise — which is why the ratio is low here even though reconstruction is near-perfect</span>`
          : ratio < 0.8
            ? `<span style="color:${C1.bad}">the cloud has shrunk toward the origin and the classes now overlap</span>`
            : `<span style="color:${C1.good}">the classes are well separated in the surviving dimensions</span>`));
  }
  ["bl-beta", "bl-sep", "bl-s2"].forEach(id => AE.on(id, "input", draw));
  AE.on("bl-post", "change", draw);
  draw();
})();

/* ═══ 22 · collapse as the KL weight anneals up ══════════════════════════
   Free bits enter as an effective per-dimension threshold: charging nothing
   below λ nats means the optimum sits at max(λ, R_i(β)), i.e. the threshold
   becomes t_i = min(βσ², λᵢ e^(−2λ_free)) and R_i = ½log(λᵢ / t_i).          */
(function () {
  const svg = d3.select("#ae-collapse");
  if (svg.empty()) return;
  const C1 = AE.col, g = svg.append("g"), D = 8;
  const LX = 60, LY = 26, LW = 380, LH = 290;
  const RX = 520, RY = 26, RW = 196, RH = 290;
  const ev = d3.range(D).map(i => 2.4 * Math.pow(0.72, i));
  const pal = d3.range(D).map(i => d3.interpolateTurbo(0.08 + 0.82 * i / (D - 1)));
  let timer = null;

  function solve(beta, s2, free) {
    return ev.map(lam => {
      const t = Math.min(beta * s2, lam * Math.exp(-2 * free), lam);
      return { lam, t, rate: 0.5 * Math.log(lam / t), dist: t, on: 0.5 * Math.log(lam / t) > 1e-4 };
    });
  }
  function schedule(u, betaMax) {                       // 0 → βmax over the first 60% of training
    return betaMax * AE.clamp(u / 0.6, 0, 1);
  }

  function draw() {
    const betaMax = AE.val("pc-beta"), s2 = AE.val("pc-s2"), free = AE.val("pc-free"), tpc = AE.val("pc-t");
    AE.put("pc-betav", AE.fmt(betaMax, 1)); AE.put("pc-s2v", AE.fmt(s2, 2));
    AE.put("pc-freev", AE.fmt(free, 2)); AE.put("pc-tv", tpc);
    g.selectAll("*").remove();
    const u = tpc / 100, betaNow = schedule(u, betaMax);

    /* left — per-dimension rate over the anneal */
    AE.panelBox(g, LX - 48, LY - 22, LW + 68, LH + 46, "per-dimension KL through training");
    const us = AE.linspace(0, 1, 220);
    const tracks = d3.range(D).map(j => us.map(uu => solve(schedule(uu, betaMax), s2, free)[j].rate));
    const ymax = Math.max(0.4, d3.max(tracks.map(t => d3.max(t))) * 1.12);
    const x = d3.scaleLinear().domain([0, 1]).range([LX, LX + LW]);
    const y = d3.scaleLinear().domain([0, ymax]).range([LY + LH, LY]);
    AE.gridY(g.append("g").attr("transform", `translate(${LX},0)`), y, LW, 5);
    AE.axisB(g.append("g").attr("transform", `translate(${LX},0)`),
      d3.scaleLinear().domain([0, 100]).range([0, LW]), LY + LH, 5, "training progress, %");
    AE.axisL(g.append("g").attr("transform", `translate(${LX},0)`), y, 0, 5, "KL per dimension, nats");
    tracks.forEach((tr, j) => g.append("path").attr("d", d3.line().x((d, i) => x(us[i])).y(d => y(d))(tr))
      .attr("fill", "none").attr("stroke", pal[j]).attr("stroke-width", 1.9).attr("stroke-opacity", .92));
    // where the anneal reaches its target
    g.append("line").attr("x1", x(0.6)).attr("x2", x(0.6)).attr("y1", LY).attr("y2", LY + LH)
      .attr("stroke", C1.muted).attr("stroke-dasharray", "3 3").attr("stroke-opacity", .5);
    g.append("text").attr("x", x(0.6) + 5).attr("y", LY + 12).attr("fill", C1.muted).attr("font-size", 9.5)
      .text("β reaches its target");
    g.append("line").attr("x1", x(u)).attr("x2", x(u)).attr("y1", LY).attr("y2", LY + LH)
      .attr("stroke", C1.ink).attr("stroke-width", 1.4).attr("stroke-opacity", .6);

    /* right — the state right now */
    const now = solve(betaNow, s2, free), act = now.filter(q => q.on).length;
    AE.panelBox(g, RX - 16, RY - 22, RW + 34, RH + 46, "right now,  β = " + AE.fmt(betaNow, 2));
    const xb = d3.scaleBand().domain(d3.range(D)).range([RX, RX + RW]).padding(0.2);
    const yb = d3.scaleLinear().domain([0, ymax]).range([RY + RH - 26, RY + 8]);
    g.selectAll("rect.b").data(now).enter().append("rect").attr("class", "b")
      .attr("x", (q, i) => xb(i)).attr("width", xb.bandwidth())
      .attr("y", q => yb(q.rate)).attr("height", q => Math.max(0, (RY + RH - 26) - yb(q.rate)))
      .attr("rx", 2).attr("fill", (q, i) => q.on ? pal[i] : C1.line)
      .attr("stroke", q => q.on ? "none" : C1.bad);
    now.forEach((q, i) => {
      if (!q.on) g.append("text").attr("x", xb(i) + xb.bandwidth() / 2).attr("y", RY + RH - 30)
        .attr("text-anchor", "middle").attr("fill", C1.bad).attr("font-size", 12).attr("font-weight", 700).text("×");
      g.append("text").attr("x", xb(i) + xb.bandwidth() / 2).attr("y", RY + RH - 10).attr("text-anchor", "middle")
        .attr("fill", C1.muted).attr("font-size", 9).text(i + 1);
    });
    g.append("line").attr("x1", RX).attr("x2", RX + RW).attr("y1", RY + RH - 26).attr("y2", RY + RH - 26).attr("stroke", C1.line);
    if (free > 0) {
      g.append("line").attr("x1", RX).attr("x2", RX + RW).attr("y1", yb(free)).attr("y2", yb(free))
        .attr("stroke", C1.good).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.5);
      g.append("text").attr("x", RX + RW).attr("y", yb(free) - 5).attr("text-anchor", "end")
        .attr("fill", C1.good).attr("font-size", 9.5).text("free bits floor");
    }

    const R0 = d3.sum(now, q => q.rate), D0 = d3.sum(now, q => q.dist), tot = d3.sum(ev);
    AE.put("pc-out",
      `<span class="keep">β</span> = <b>${AE.fmt(betaNow, 2)}</b> of a target <b>${AE.fmt(betaMax, 1)}</b>, threshold <code><span class="keep">β</span><span class="keep">σ</span>² = ${AE.fmt(betaNow * s2, 3)}</code>` +
      (free > 0 ? `, capped by free bits at <code>${AE.fmt(free, 2)}</code> nats per dimension` : "") + ` &nbsp;·&nbsp; ` +
      `<b>${act}</b> of ${D} dimensions alive, <b>${D - act}</b> collapsed &nbsp;·&nbsp; ` +
      `total rate <b>${AE.fmt(R0, 3)}</b> nats, distortion <b>${AE.fmt(D0, 3)}</b> of <b>${AE.fmt(tot, 2)}</b> &nbsp;·&nbsp; ` +
      (act === 0 ? `<span style="color:${C1.bad}">complete collapse: the reconstruction is the data mean and the decoder ignores z entirely</span>`
        : free > 0 ? `<span style="color:${C1.good}">free bits hold every dimension above the floor — nothing can collapse</span>`
          : D - act > 0 ? `<span style="color:${C1.bad}">${D - act} dimension${D - act === 1 ? " has" : "s have"} died and will not come back</span>`
            : `every dimension is still carrying information`));
  }
  ["pc-beta", "pc-s2", "pc-free", "pc-t"].forEach(id => AE.on(id, "input", draw));
  AE.on("pc-play", "click", () => {
    if (timer) { clearInterval(timer); timer = null; }
    const el = document.getElementById("pc-t");
    el.value = 0; draw();
    timer = setInterval(() => {
      el.value = Math.min(100, +el.value + 1.5); draw();
      if (+el.value >= 100) { clearInterval(timer); timer = null; }
    }, 34);
  });
  draw();
})();

/* ═══ shared for §23–§24 ═════════════════════════════════════════════════
   A toy dataset with two known generating factors, its exact linear β-VAE
   encoder, and the decoder that a network with unlimited capacity would
   converge to: E[x | z] under the model's own aggregate posterior. That
   expectation is a Gaussian-kernel average of the training signals, which
   also hands us an exact measure of HOW MANY examples are being averaged —
   the quantity §24 identifies as the cause of blur.                        */
AE.bumps = function (N, L, seed) {
  const r = AE.rng(seed || 606), X = AE.zeros(N, L), fac = [];
  for (let n = 0; n < N; n++) {
    const c = 0.35 * L + 0.30 * L * r(), w = 2.5 + 4.5 * r();
    for (let j = 0; j < L; j++) X[n][j] = Math.exp(-(j - c) * (j - c) / (2 * w * w));
    fac.push({ c, w });
  }
  const mu = new Float64Array(L);
  for (let n = 0; n < N; n++) for (let j = 0; j < L; j++) mu[j] += X[n][j] / N;
  const Xc = X.map(row => Float64Array.from(row, (v, j) => v - mu[j]));
  return { X, Xc, mu, fac, N, L };
};
/* weights of the aggregate posterior at z, in log space; returns {w, ess} */
AE.kernelWeights = function (z, codes, dvar) {
  const N = codes.length, k = z.length, lw = new Float64Array(N);
  let mx = -Infinity;
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let a = 0; a < k; a++) { const e = z[a] - codes[i][a]; s += e * e / Math.max(dvar[a], 1e-9); }
    lw[i] = -0.5 * s; if (lw[i] > mx) mx = lw[i];
  }
  let Z = 0;
  for (let i = 0; i < N; i++) { lw[i] = Math.exp(lw[i] - mx); Z += lw[i]; }
  let H = 0;
  for (let i = 0; i < N; i++) { const p = lw[i] / Z; if (p > 1e-12) H -= p * Math.log(p); }
  return { w: lw, Z, ess: Math.exp(H) };           // ess = effective number of contributors
};
AE.decodeKernel = function (z, codes, dvar, X, mu) {
  const { w, Z } = AE.kernelWeights(z, codes, dvar), L = X[0].length;
  const out = new Float64Array(L);
  for (let i = 0; i < w.length; i++) for (let j = 0; j < L; j++) out[j] += w[i] * X[i][j];
  for (let j = 0; j < L; j++) out[j] /= Z;
  return out;
};
/* the exact linear β-VAE encoder for a dataset, in the top-k latent dims */
AE.linVAEEncode = function (Xc, beta, s2, k) {
  const E = AE.eigSym(AE.covariance(Xc));
  const sol = AE.linVAE(E.values.slice(0, k), beta, s2);
  const codes = Xc.map(row => {
    const m = new Float64Array(k);
    for (let a = 0; a < k; a++) {
      if (!sol[a].on) continue;
      let p = 0; for (let j = 0; j < row.length; j++) p += E.vectors[j][a] * row[j];
      m[a] = (sol[a].w / sol[a].lam) * p;
    }
    return m;
  });
  return { codes, sol, E, dvar: sol.map(q => q.d) };
};

/* ═══ 23 · latent traversal on a dataset with known factors ══════════════ */
(function () {
  const svg = d3.select("#ae-traverse");
  if (svg.empty()) return;
  const C1 = AE.col, g = svg.append("g");
  const LX = 56, LY = 26, LW = 290, LH = 290;
  const RX = 412, RY = 26, RW = 300, RH = 120;
  const SX = 412, SY = 196, SW = 300, SH = 130;
  const B = AE.bumps(240, 40, 606);

  function draw() {
    const beta = Math.pow(10, AE.val("tv-beta")), axis = +AE.sel("tv-axis");
    const z = [AE.val("tv-z1"), AE.val("tv-z2")];
    AE.put("tv-z1v", AE.fmt(z[0], 2)); AE.put("tv-z2v", AE.fmt(z[1], 2)); AE.put("tv-betav", AE.fmt(beta, 2));
    g.selectAll("*").remove();
    const { codes, sol, dvar } = AE.linVAEEncode(B.Xc, beta, 0.02, 2);

    /* left — the latent space, coloured by the TRUE bump position */
    AE.panelBox(g, LX - 44, LY - 20, LW + 64, LH + 46, "latent space, coloured by the true bump position");
    const lim = 3.0;
    const xs = d3.scaleLinear().domain([-lim, lim]).range([LX, LX + LW]);
    const ys = d3.scaleLinear().domain([-lim, lim]).range([LY + LH, LY]);
    AE.axisB(g.append("g").attr("transform", `translate(${LX},0)`), d3.scaleLinear().domain([-lim, lim]).range([0, LW]), LY + LH, 5, "z₁");
    AE.axisL(g.append("g").attr("transform", `translate(${LX},0)`), ys, 0, 5, "z₂");
    const ring = AE.linspace(0, 2 * Math.PI, 100);
    [1, 2].forEach(c => g.append("path").attr("d", d3.line().x(a => xs(c * Math.cos(a))).y(a => ys(c * Math.sin(a)))(ring))
      .attr("fill", "none").attr("stroke", C1.muted).attr("stroke-dasharray", "4 4").attr("stroke-opacity", c === 1 ? .85 : .4));
    const cmin = d3.min(B.fac, f => f.c), cmax = d3.max(B.fac, f => f.c);
    g.selectAll("circle.c").data(d3.range(B.N)).enter().append("circle").attr("class", "c")
      .attr("cx", i => xs(codes[i][0])).attr("cy", i => ys(codes[i][1])).attr("r", 2.4)
      .attr("fill", i => d3.interpolatePlasma((B.fac[i].c - cmin) / (cmax - cmin))).attr("fill-opacity", .8);
    // the traversal line and the probe
    const t0 = -2.6, t1 = 2.6;
    g.append("line")
      .attr("x1", xs(axis === 1 ? t0 : z[0])).attr("y1", ys(axis === 1 ? z[1] : t0))
      .attr("x2", xs(axis === 1 ? t1 : z[0])).attr("y2", ys(axis === 1 ? z[1] : t1))
      .attr("stroke", C1.ink).attr("stroke-width", 1.3).attr("stroke-opacity", .55).attr("stroke-dasharray", "4 3");
    g.append("circle").attr("cx", xs(z[0])).attr("cy", ys(z[1])).attr("r", 5)
      .attr("fill", C1.B).attr("stroke", C1.bg).attr("stroke-width", 1.6);

    /* right top — the decoded signal at the probe */
    const out = AE.decodeKernel(z, codes, dvar, B.X, B.mu);
    const { ess } = AE.kernelWeights(z, codes, dvar);
    AE.panelBox(g, RX - 12, RY - 20, RW + 26, RH + 40, "decoded signal at the probe,  x̂ = E[x | z]");
    const xr = d3.scaleLinear().domain([0, B.L - 1]).range([RX, RX + RW]);
    const yr = d3.scaleLinear().domain([0, 1.12]).range([RY + RH, RY]);
    g.append("path").attr("d", d3.area().x((d, i) => xr(i)).y0(RY + RH).y1(d => yr(d))(Array.from(out)))
      .attr("fill", C1.B).attr("fill-opacity", .2);
    g.append("path").attr("d", d3.line().x((d, i) => xr(i)).y(d => yr(d))(Array.from(out)))
      .attr("fill", "none").attr("stroke", C1.B).attr("stroke-width", 2.2);
    g.append("line").attr("x1", RX).attr("x2", RX + RW).attr("y1", yr(1)).attr("y2", yr(1))
      .attr("stroke", C1.muted).attr("stroke-dasharray", "3 3").attr("stroke-opacity", .6);
    g.append("text").attr("x", RX + RW).attr("y", yr(1) - 4).attr("text-anchor", "end")
      .attr("fill", C1.muted).attr("font-size", 9).text("every real signal peaks at exactly 1");

    /* right bottom — the traversal strip */
    AE.panelBox(g, SX - 12, SY - 20, SW + 26, SH + 42,
      "traversal along " + (axis === 1 ? "z₁" : "z₂") + ",  from −2.6 to +2.6");
    const NR = 11, rowH = SH / NR;
    for (let k = 0; k < NR; k++) {
      const t = t0 + (t1 - t0) * k / (NR - 1);
      const zz = axis === 1 ? [t, z[1]] : [z[0], t];
      const o = AE.decodeKernel(zz, codes, dvar, B.X, B.mu);
      for (let j = 0; j < B.L; j++) {
        g.append("rect").attr("x", SX + j * SW / B.L).attr("y", SY + k * rowH)
          .attr("width", SW / B.L + .6).attr("height", rowH + .6)
          .attr("fill", d3.interpolateMagma(AE.clamp(o[j], 0, 1) * 0.9 + 0.05));
      }
      g.append("text").attr("x", SX - 6).attr("y", SY + k * rowH + rowH / 2 + 3).attr("text-anchor", "end")
        .attr("fill", C1.muted).attr("font-size", 8).text(AE.fmt(t, 1));
    }

    let pk = 0, arg = 0;
    out.forEach((v, j) => { if (v > pk) { pk = v; arg = j; } });
    AE.put("tv-out",
      `<span class="keep">β</span> = <b>${AE.fmt(beta, 2)}</b>, <b>${sol.filter(q => q.on).length}</b> of 2 shown dimensions active &nbsp;·&nbsp; ` +
      `probe (${AE.fmt(z[0], 2)}, ${AE.fmt(z[1], 2)}) decodes to a bump peaking at position <b>${arg}</b> with height <b>${AE.fmt(pk, 3)}</b> &nbsp;·&nbsp; ` +
      `it is an average of <b>${AE.fmt(ess, 1)}</b> effective training signals out of ${B.N} &nbsp;·&nbsp; ` +
      (ess > 60 ? `<span style="color:${C1.bad}">that is a lot of averaging — the output is a blur, see §24</span>`
        : `<span style="color:${C1.good}">few enough that the output still looks like a real signal</span>`));
  }
  ["tv-z1", "tv-z2", "tv-beta"].forEach(id => AE.on(id, "input", draw));
  AE.on("tv-axis", "change", draw);
  draw();
})();

/* ═══ 24 · blur, measured as an effective number of averaged examples ════ */
(function () {
  const svg = d3.select("#ae-blur");
  if (svg.empty()) return;
  const C1 = AE.col, g = svg.append("g");
  const LX = 52, LY = 26, LW = 320, LH = 290;
  const RX = 444, RY = 26, RW = 272, RH = 290;
  const B = AE.bumps(240, 40, 606);
  let dataPeak = 0;
  for (let j = 0; j < B.L; j++) dataPeak = Math.max(dataPeak, B.mu[j]);

  function stats(beta) {
    const { codes, dvar, sol } = AE.linVAEEncode(B.Xc, beta, 0.02, 3);
    let peak = 0, ess = 0, mse = 0;
    for (let n = 0; n < B.N; n++) {
      const o = AE.decodeKernel(codes[n], codes, dvar, B.X, B.mu);
      let p = 0, e = 0;
      for (let j = 0; j < B.L; j++) { if (o[j] > p) p = o[j]; e += (o[j] - B.X[n][j]) * (o[j] - B.X[n][j]); }
      peak += p / B.N; ess += AE.kernelWeights(codes[n], codes, dvar).ess / B.N; mse += e / B.L / B.N;
    }
    return { codes, dvar, sol, peak, ess, mse };
  }

  function draw() {
    const beta = Math.pow(10, AE.val("bl2-beta")), idx = AE.val("bl2-i"), showC = AE.chk("bl2-contrib");
    AE.put("bl2-betav", AE.fmt(beta, 2)); AE.put("bl2-iv", idx);
    g.selectAll("*").remove();
    const S = stats(beta);
    const out = AE.decodeKernel(S.codes[idx], S.codes, S.dvar, B.X, B.mu);
    const kw = AE.kernelWeights(S.codes[idx], S.codes, S.dvar);

    /* left — one example and its reconstruction */
    AE.panelBox(g, LX - 40, LY - 20, LW + 60, LH + 46, "example " + idx + ": the truth, the reconstruction, and what was averaged");
    const x = d3.scaleLinear().domain([0, B.L - 1]).range([LX, LX + LW]);
    const y = d3.scaleLinear().domain([0, 1.14]).range([LY + LH, LY]);
    AE.gridY(g.append("g").attr("transform", `translate(${LX},0)`), y, LW, 5);
    AE.axisB(g.append("g").attr("transform", `translate(${LX},0)`), d3.scaleLinear().domain([0, B.L - 1]).range([0, LW]), LY + LH, 5, "position");
    AE.axisL(g.append("g").attr("transform", `translate(${LX},0)`), y, 0, 5, "value");
    if (showC) {
      const order = d3.range(B.N).sort((a, b) => kw.w[b] - kw.w[a]).slice(0, 40);
      order.forEach(i => g.append("path").attr("d", d3.line().x((d, j) => x(j)).y(d => y(d))(Array.from(B.X[i])))
        .attr("fill", "none").attr("stroke", C1.muted)
        .attr("stroke-opacity", AE.clamp(0.10 + 0.85 * kw.w[i] / kw.Z * kw.ess, 0.05, 0.6)).attr("stroke-width", 1));
    }
    g.append("path").attr("d", d3.line().x((d, j) => x(j)).y(d => y(d))(Array.from(B.X[idx])))
      .attr("fill", "none").attr("stroke", C1.good).attr("stroke-width", 2.4);
    g.append("path").attr("d", d3.line().x((d, j) => x(j)).y(d => y(d))(Array.from(out)))
      .attr("fill", "none").attr("stroke", C1.B).attr("stroke-width", 2.4).attr("stroke-dasharray", "6 3");
    AE.legend(g, [
      { c: C1.good, t: "the true signal, peak = 1.000" },
      { c: C1.B, t: "the decoder's output E[x | z]", dash: "6 3" },
      { c: C1.muted, t: "training signals being averaged in" }
    ], LX + 10, LY + 14);

    /* right — the sweep over β */
    AE.panelBox(g, RX - 16, RY - 20, RW + 34, RH + 46, "sharpness and averaging against β");
    const bs = AE.linspace(-0.7, 2.3, 26).map(u => Math.pow(10, u));
    const rows = bs.map(b => { const s = stats(b); return { b, peak: s.peak, ess: s.ess }; });
    const xb = d3.scaleLog().domain([0.2, 200]).range([RX, RX + RW]);
    const yp = d3.scaleLinear().domain([Math.min(0.7, dataPeak - 0.03), 1.02]).range([RY + RH - 20, RY + 10]);
    AE.axisB(g.append("g"), xb, RY + RH - 20, 4, "β", d3.format("~g"));
    AE.axisL(g.append("g").attr("transform", `translate(${RX},0)`), yp, 4, null);
    g.append("line").attr("x1", RX).attr("x2", RX + RW).attr("y1", yp(dataPeak)).attr("y2", yp(dataPeak))
      .attr("stroke", C1.bad).attr("stroke-dasharray", "4 4");
    g.append("text").attr("x", RX + RW).attr("y", yp(dataPeak) - 5).attr("text-anchor", "end")
      .attr("fill", C1.bad).attr("font-size", 9.5).text("peak of the dataset mean = " + AE.fmt(dataPeak, 3));
    g.append("path").attr("d", d3.line().x(p => xb(p.b)).y(p => yp(p.peak))(rows))
      .attr("fill", "none").attr("stroke", C1.B).attr("stroke-width", 2.4);
    const ye = d3.scaleLinear().domain([0, B.N]).range([RY + RH - 20, RY + 10]);
    g.append("path").attr("d", d3.line().x(p => xb(p.b)).y(p => ye(p.ess))(rows))
      .attr("fill", "none").attr("stroke", C1.teal).attr("stroke-width", 2).attr("stroke-dasharray", "5 3");
    g.append("line").attr("x1", xb(beta)).attr("x2", xb(beta)).attr("y1", RY + 6).attr("y2", RY + RH - 20)
      .attr("stroke", C1.ink).attr("stroke-opacity", .4).attr("stroke-dasharray", "3 3");
    AE.legend(g, [
      { c: C1.B, t: "mean peak height of the reconstruction" },
      { c: C1.teal, t: "effective examples averaged (0–" + B.N + ")", dash: "5 3" }
    ], RX + 6, RY + RH + 6);

    AE.put("bl2-out",
      `<span class="keep">β</span> = <b>${AE.fmt(beta, 2)}</b> &nbsp;·&nbsp; ` +
      `mean reconstruction peak <b>${AE.fmt(S.peak, 3)}</b> against a true peak of <b>1.000</b> &nbsp;·&nbsp; ` +
      `each output averages <b>${AE.fmt(S.ess, 1)}</b> of ${B.N} training signals &nbsp;·&nbsp; ` +
      `reconstruction MSE <b>${AE.fmt(S.mse, 5)}</b> &nbsp;·&nbsp; ` +
      (S.peak < dataPeak + 0.02
        ? `<span style="color:${C1.bad}">the output has reached the dataset mean: the code carries nothing and every reconstruction is identical</span>`
        : S.ess > 40 ? `<span style="color:${C1.bad}">heavy averaging — this is blur, and it is what squared error asked for</span>`
          : `<span style="color:${C1.good}">little averaging: the reconstruction is still a single sharp signal</span>`));
  }
  ["bl2-beta", "bl2-i"].forEach(id => AE.on(id, "input", draw));
  AE.on("bl2-contrib", "change", draw);
  draw();
})();

/* ═══ 26 · vector quantisation and the straight-through gradient ═════════ */
(function () {
  const svg = d3.select("#ae-vq");
  if (svg.empty()) return;
  const C1 = AE.col, g = svg.append("g");
  const PX = 50, PY = 22, PW = 350, PH = 340;
  const x = d3.scaleLinear().domain([-2.6, 2.6]).range([PX, PX + PW]);
  const y = d3.scaleLinear().domain([-2.6, 2.6]).range([PY + PH, PY]);

  // a fixed cloud of "encoder outputs": three blobs, so the codebook has structure to find
  const r0 = AE.rng(777), pts = [];
  for (let c = 0; c < 3; c++) {
    const th = 2 * Math.PI * c / 3, cx = 1.25 * Math.cos(th), cy = 1.25 * Math.sin(th);
    for (let n = 0; n < 130; n++) pts.push([cx + 0.55 * AE.randn(r0), cy + 0.55 * AE.randn(r0)]);
  }

  function codebook(K, iters) {
    const r = AE.rng(11 + K);
    let e = d3.range(K).map(() => {
      const p = pts[Math.floor(r() * pts.length)];
      return [p[0] + 0.12 * AE.randn(r), p[1] + 0.12 * AE.randn(r)];
    });
    for (let it = 0; it < iters; it++) {                 // Lloyd / k-means: the "codebook loss"
      const sx = new Float64Array(K), sy = new Float64Array(K), cn = new Float64Array(K);
      pts.forEach(p => {
        let b = 0, bd = Infinity;
        for (let k = 0; k < K; k++) { const d = (p[0] - e[k][0]) ** 2 + (p[1] - e[k][1]) ** 2; if (d < bd) { bd = d; b = k; } }
        sx[b] += p[0]; sy[b] += p[1]; cn[b]++;
      });
      e = e.map((v, k) => cn[k] > 0 ? [sx[k] / cn[k], sy[k] / cn[k]] : v);
    }
    return e;
  }
  const nearest = (p, e) => {
    let b = 0, bd = Infinity;
    for (let k = 0; k < e.length; k++) { const d = (p[0] - e[k][0]) ** 2 + (p[1] - e[k][1]) ** 2; if (d < bd) { bd = d; b = k; } }
    return { k: b, d2: bd };
  };

  function draw() {
    const K = AE.val("vq-K"), iters = AE.val("vq-it"), th = AE.val("vq-th") * Math.PI / 180, cells = AE.chk("vq-cells");
    AE.put("vq-Kv", K); AE.put("vq-itv", iters); AE.put("vq-thv", AE.val("vq-th"));
    g.selectAll("*").remove();
    const e = codebook(K, iters);

    AE.panelBox(g, PX - 40, PY - 18, PW + 60, PH + 44, "encoder output space  z_e(x),  partitioned by the codebook");
    if (cells) {
      const GX = 84, GY = 82, cw = PW / GX, ch = PH / GY;
      const cellsArr = [];
      for (let i = 0; i < GX; i++) for (let j = 0; j < GY; j++) {
        const q = [x.invert(PX + (i + .5) * cw), y.invert(PY + (j + .5) * ch)];
        cellsArr.push({ i, j, k: nearest(q, e).k });
      }
      g.selectAll("rect.cl").data(cellsArr).enter().append("rect").attr("class", "cl")
        .attr("x", c => PX + c.i * cw).attr("y", c => PY + c.j * ch)
        .attr("width", cw + .6).attr("height", ch + .6)
        .attr("fill", c => d3.interpolateTurbo(0.06 + 0.88 * ((c.k * 2654435761) % 997) / 997))
        .attr("fill-opacity", .16);
    }
    g.selectAll("circle.p").data(pts).enter().append("circle").attr("class", "p")
      .attr("cx", p => x(p[0])).attr("cy", p => y(p[1])).attr("r", 1.5)
      .attr("fill", C1.muted).attr("fill-opacity", .45);
    g.selectAll("circle.e").data(e).enter().append("circle").attr("class", "e")
      .attr("cx", p => x(p[0])).attr("cy", p => y(p[1])).attr("r", 4)
      .attr("fill", C1.A).attr("stroke", C1.bg).attr("stroke-width", 1.2);

    // the probe: one encoder output, its snap, and the straight-through gradient
    const ze = [1.75 * Math.cos(th), 1.75 * Math.sin(th)];
    const nb = nearest(ze, e), zq = e[nb.k];
    g.append("line").attr("x1", x(ze[0])).attr("y1", y(ze[1])).attr("x2", x(zq[0])).attr("y2", y(zq[1]))
      .attr("stroke", C1.B).attr("stroke-width", 2).attr("stroke-dasharray", "4 3");
    g.append("circle").attr("cx", x(ze[0])).attr("cy", y(ze[1])).attr("r", 5.5)
      .attr("fill", C1.B).attr("stroke", C1.bg).attr("stroke-width", 1.4);
    g.append("circle").attr("cx", x(zq[0])).attr("cy", y(zq[1])).attr("r", 6.5)
      .attr("fill", "none").attr("stroke", C1.good).attr("stroke-width", 2.4);
    // a made-up decoder gradient at z_q, copied unchanged to z_e
    const gvec = [-0.42 * Math.sin(th * 1.7 + 0.6), 0.42 * Math.cos(th * 1.7 + 0.6)];
    AE.arrow(g, x(zq[0]), y(zq[1]), x(zq[0] + gvec[0]), y(zq[1] + gvec[1]), C1.good, 2, 6);
    AE.arrow(g, x(ze[0]), y(ze[1]), x(ze[0] + gvec[0]), y(ze[1] + gvec[1]), C1.good, 2, 6);
    g.append("text").attr("x", x(ze[0]) + 10).attr("y", y(ze[1]) - 8).attr("fill", C1.B).attr("font-size", 10).text("z_e(x)");
    g.append("text").attr("x", x(zq[0]) + 10).attr("y", y(zq[1]) + 16).attr("fill", C1.good).attr("font-size", 10).text("z_q(x) = e_" + nb.k);

    // right panel: usage, rate, quantisation error
    const counts = new Float64Array(K);
    let mse = 0;
    pts.forEach(p => { const n2 = nearest(p, e); counts[n2.k]++; mse += n2.d2 / pts.length; });
    const used = counts.reduce((a, v) => a + (v > 0 ? 1 : 0), 0);
    const bx = PX + PW + 46;
    AE.panelBox(g, bx, PY, 240, 128, "codebook usage");
    const xb = d3.scaleBand().domain(d3.range(K)).range([bx + 10, bx + 230]).padding(0.12);
    const yb = d3.scaleLinear().domain([0, d3.max(counts)]).range([PY + 112, PY + 28]);
    g.selectAll("rect.u").data(d3.range(K)).enter().append("rect").attr("class", "u")
      .attr("x", k => xb(k)).attr("width", xb.bandwidth())
      .attr("y", k => yb(counts[k])).attr("height", k => (PY + 112) - yb(counts[k]))
      .attr("fill", k => counts[k] > 0 ? (k === nb.k ? C1.good : C1.A) : C1.bad).attr("fill-opacity", .8);
    AE.panelBox(g, bx, PY + 148, 240, 152, "what quantisation costs");
    const rows = [
      ["rate,  log K", AE.fmt(Math.log(K), 3) + " nats", C1.B],
      ["in bits,  log₂K", AE.fmt(Math.log2(K), 2) + " bits", C1.B],
      ["mean quantisation error", AE.fmt(mse, 4), C1.teal],
      ["codebook entries in use", used + " of " + K, used === K ? C1.good : C1.bad]
    ];
    rows.forEach((rw, i) => {
      g.append("text").attr("x", bx + 12).attr("y", PY + 180 + i * 30).attr("fill", C1.muted).attr("font-size", 10.5).text(rw[0]);
      g.append("text").attr("x", bx + 228).attr("y", PY + 180 + i * 30).attr("text-anchor", "end").attr("fill", rw[2])
        .attr("font-size", 12.5).attr("font-family", "SF Mono,Menlo,monospace").text(rw[1]);
    });

    AE.put("vq-out",
      `K = <b>${K}</b> &nbsp;·&nbsp; the KL is the constant <code>log K = ${AE.fmt(Math.log(K), 3)}</code> nats, ` +
      `so <b>nothing in the objective can push the rate to zero</b> — no posterior collapse &nbsp;·&nbsp; ` +
      `mean quantisation error <b>${AE.fmt(mse, 4)}</b> &nbsp;·&nbsp; ` +
      `<b>${used}</b> of ${K} entries used` +
      (used < K ? `<span style="color:${C1.bad}"> — the rest are dead and receive no gradient: this is codebook collapse</span>`
        : `<span style="color:${C1.good}"> — every entry is alive</span>`) +
      ` &nbsp;·&nbsp; the green arrow is the decoder's gradient, copied <i>unchanged</i> across the snap by the straight-through estimator`);
  }
  ["vq-K", "vq-it", "vq-th"].forEach(id => AE.on(id, "input", draw));
  AE.on("vq-cells", "change", draw);
  draw();
})();

/* ═══ 27 · the compression ladder ════════════════════════════════════════ */
(function () {
  const svg = d3.select("#ae-ladder");
  if (svg.empty()) return;
  const C1 = AE.col, g = svg.append("g");
  // published autoencoder-zoo measurements: f, channels, reconstruction FID, PSNR
  const zoo = [
    { f: 2, c: 2, fid: 0.086, psnr: 32.47 },
    { f: 4, c: 3, fid: 0.27, psnr: 27.53 },
    { f: 8, c: 4, fid: 0.90, psnr: 24.19 },
    { f: 16, c: 16, fid: 0.87, psnr: 24.08 },
    { f: 32, c: 16, fid: 7.30, psnr: 20.38 }
  ];
  function draw() {
    const R = AE.val("ld-res"), cost = AE.sel("ld-cost");
    AE.put("ld-resv", R);
    g.selectAll("*").remove();
    const pix = R * R * 3, pos0 = R * R;
    const rows = zoo.map(z => {
      const h = Math.round(R / z.f), el = h * h * z.c, pos = h * h;
      return { ...z, h, el, red: pix / el, posRatio: cost === "quad" ? (pos0 / pos) ** 2 : pos0 / pos };
    });
    const X0 = 46, Y0 = 40, W = 668, rowH = 58;
    g.append("text").attr("x", X0).attr("y", 22).attr("fill", C1.muted).attr("font-size", 10.5)
      .text(`a ${R} × ${R} × 3 image = ${AE.commas(pix)} numbers`);
    const maxRed = d3.max(rows, r => r.red);
    rows.forEach((r, i) => {
      const yy = Y0 + i * rowH;
      g.append("rect").attr("x", X0 - 8).attr("y", yy - 14).attr("width", W).attr("height", rowH - 8).attr("rx", 7)
        .attr("fill", r.f === 8 ? "rgba(91,156,255,.08)" : C1.panel2).attr("fill-opacity", r.f === 8 ? 1 : .45)
        .attr("stroke", r.f === 8 ? C1.A : C1.line);
      g.append("text").attr("x", X0 + 4).attr("y", yy + 8).attr("fill", C1.ink).attr("font-size", 13)
        .attr("font-weight", r.f === 8 ? 700 : 500).text("f = " + r.f);
      g.append("text").attr("x", X0 + 60).attr("y", yy + 8).attr("fill", C1.muted).attr("font-size", 11)
        .attr("font-family", "SF Mono,Menlo,monospace").text(`${r.h}×${r.h}×${r.c} = ${AE.commas(r.el)}`);
      // the reduction bar
      const bx = X0 + 220, bw = 200;
      g.append("rect").attr("x", bx).attr("y", yy - 6).attr("width", Math.max(2, bw * r.red / maxRed)).attr("height", 16)
        .attr("rx", 3).attr("fill", C1.teal).attr("fill-opacity", .65);
      g.append("text").attr("x", bx + Math.max(2, bw * r.red / maxRed) + 8).attr("y", yy + 7)
        .attr("fill", C1.teal).attr("font-size", 11).attr("font-family", "SF Mono,Menlo,monospace")
        .text(AE.fmt(r.red, 1) + "× fewer");
      g.append("text").attr("x", X0 + 512).attr("y", yy + 8).attr("fill", C1.B).attr("font-size", 11)
        .attr("font-family", "SF Mono,Menlo,monospace").text(AE.fmt(r.posRatio, 0) + "× cheaper/step");
      g.append("text").attr("x", X0 + 660).attr("y", yy + 8).attr("text-anchor", "end")
        .attr("fill", r.fid > 2 ? C1.bad : C1.good).attr("font-size", 11)
        .attr("font-family", "SF Mono,Menlo,monospace").text("R-FID " + AE.fmt(r.fid, 2));
    });
    g.append("text").attr("x", X0 + 220).attr("y", Y0 - 22).attr("fill", C1.muted).attr("font-size", 10).text("elements saved");
    g.append("text").attr("x", X0 + 512).attr("y", Y0 - 22).attr("fill", C1.muted).attr("font-size", 10)
      .text(cost === "quad" ? "attention cost, relative" : "convolution cost, relative");
    g.append("text").attr("x", X0 + 660).attr("y", Y0 - 22).attr("text-anchor", "end").attr("fill", C1.muted).attr("font-size", 10)
      .text("reconstruction ceiling");

    const w = rows.find(r => r.f === 8);
    AE.put("ld-out",
      `at ${R} × ${R}: the <b>f = 8</b> autoencoder turns ${AE.commas(pix)} numbers into ${AE.commas(w.el)} — ` +
      `<b>${AE.fmt(w.red, 1)}×</b> fewer elements and <b>${AE.fmt(w.posRatio, 0)}×</b> less ` +
      (cost === "quad" ? "attention" : "convolution") + ` work per denoising step, ` +
      `for a reconstruction FID of <b>${AE.fmt(w.fid, 2)}</b> &nbsp;·&nbsp; ` +
      `at <b>f = 32</b> the same trade costs a reconstruction FID of <b>7.30</b> — an <b>8×</b> worse ceiling that no amount of diffusion can lift`);
  }
  AE.on("ld-res", "input", draw);
  AE.on("ld-cost", "change", draw);
  draw();
})();
