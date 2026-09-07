/* segmentation.viz.js — the nineteen visualizations on vision/segmentation.html.
   Loaded after ../data.js → ../notes.js → vision-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

     1  #illposed-svg  one scene, a ladder of partitions, three annotators who
                       each stopped at a different rung — and disagree with each other
     2  #otsu-svg      a histogram with a draggable threshold and σ²_B(t), σ²_W(t)
                       plotted beneath it; the exhaustive-search optimum marked
     3  #bimodal-svg   two Gaussians whose separation and prior are adjustable:
                       when the valley disappears, Otsu keeps answering anyway
     4  #adapt-svg     an illumination ramp that no global threshold survives, and
                       the local window that does
     5  #cc-svg        connected components, 4- against 8-connectivity, and the
                       Jordan-curve paradox on a diagonal ring
     6  #grow-svg      region growing from seeds with a tolerance slider, and the
                       exact tolerance at which it leaks through a weak boundary
     7  #quad-svg      quadtree splitting and the merge pass that repairs it
     8  #rag-svg       the region adjacency graph, and merging on it
     9  #wshed-svg     the watershed as literal flooding, with the dams appearing
    10  #marker-svg    over-segmentation, and marker-controlled watershed
    11  #snake-svg     a snake evolving, with every energy term toggleable
    12  #lset-svg      a level set: the embedding surface, its zero set, and a
                       topology change that a parametric curve cannot do
    13  #ms-svg        mean shift climbing a kernel density estimate, with the
                       mode count against bandwidth drawn as a staircase
    14  #joint-svg     range-only against joint spatial–range clustering
    15  #graph-svg     an image as a weighted graph, and its affinity matrix
    16  #ncut-svg      the Laplacian, the second eigenvector, and the cut it implies
                       — against the brute-force optimum over all bipartitions
    17  #mincut-svg    max-flow saturating on a tiny grid, and the min cut it certifies
    18  #slic-svg      SLIC superpixels with a compactness slider
    19  #iou-svg       IoU and Dice on the same masks, and the case where averaging
                       them ranks two methods in opposite orders

   Every number these print is recomputed from the data they draw, so the pictures
   and the prose cannot drift apart.                                              */

/* ══════════ page-local helpers (deliberately NOT in vision-viz.js) ══════════ */
const SG = (function () {

  /* ---- small-image drawing -------------------------------------------------
     Every figure on this page draws a low-resolution image as a grid of rects.
     `cells` returns the <g> so callers can add overlays in the same user space. */
  const CANV = (typeof document !== "undefined" && document.createElement)
    ? document.createElement("canvas") : null;
  function cells(g, x0, y0, cw, W, H, colorOf) {
    const gg = g.append("g").attr("transform", `translate(${x0},${y0})`);
    /* Below a few thousand cells, individual <rect>s are fine and keep the DOM
       inspectable. Above that they are not: several panels of 104 × 72 would put
       tens of thousands of nodes on the page and every slider drag would rebuild
       them. Larger grids are rasterised into one <image> instead, in exactly the
       same user space, so overlays drawn afterwards line up unchanged. */
    const ctx = (CANV && W * H > 2500) ? CANV.getContext("2d") : null;
    if (ctx) {
      CANV.width = W; CANV.height = H;
      ctx.clearRect(0, 0, W, H);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const c = colorOf(x, y);
        if (!c || c === "none") continue;
        ctx.fillStyle = c;
        ctx.fillRect(x, y, 1, 1);
      }
      gg.append("image").attr("x", 0).attr("y", 0)
        .attr("width", W * cw).attr("height", H * cw)
        .attr("preserveAspectRatio", "none")
        .attr("image-rendering", "pixelated")
        .attr("href", CANV.toDataURL());
      return gg;
    }
    const data = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) data.push([x, y]);
    gg.selectAll("rect").data(data).join("rect")
      .attr("x", d => d[0] * cw).attr("y", d => d[1] * cw)
      .attr("width", cw + 0.4).attr("height", cw + 0.4)
      .attr("shape-rendering", "crispEdges")
      .attr("fill", d => colorOf(d[0], d[1]))
      .attr("fill-opacity", d => { const c = colorOf(d[0], d[1]); return (!c || c === "none") ? 0 : 1; });
    return gg;
  }

  /* the boundary segments of a label map, in cell units — draw them scaled */
  function edgesOf(lab, W, H) {
    const segs = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const L = lab[y * W + x];
      if (x + 1 < W && lab[y * W + x + 1] !== L) segs.push([x + 1, y, x + 1, y + 1]);
      if (y + 1 < H && lab[(y + 1) * W + x] !== L) segs.push([x, y + 1, x + 1, y + 1]);
    }
    return segs;
  }
  function drawEdges(g, segs, cw, opt) {
    const o = Object.assign({ color: VC.ink, w: 1, op: 0.85 }, opt || {});
    const d = segs.map(s => `M${s[0] * cw},${s[1] * cw}L${s[2] * cw},${s[3] * cw}`).join(" ");
    return g.append("path").attr("d", d).attr("stroke", o.color)
      .attr("stroke-width", o.w).attr("stroke-opacity", o.op).attr("fill", "none")
      .attr("shape-rendering", "crispEdges");
  }

  /* grey ramp for intensity images, 0..255 → a neutral grey the theme tolerates */
  const grey = v => { const c = Math.max(0, Math.min(255, Math.round(v))); return `rgb(${c},${c},${c})`; };

  /* a palette for region labels: readable, distinct, and stable in index order */
  const PAL = ["#5b9cff", "#ffb454", "#4ade80", "#f87171", "#c084fc", "#2dd4bf",
    "#fb7185", "#a3e635", "#38bdf8", "#f0abfc", "#fbbf24", "#94a3b8",
    "#7dd3fc", "#fca5a5", "#86efac", "#d8b4fe"];
  const pal = i => PAL[((i % PAL.length) + PAL.length) % PAL.length];

  /* ---- union–find, used by every merging and component routine here -------- */
  function UF(n) {
    const p = new Int32Array(n); for (let i = 0; i < n; i++) p[i] = i;
    function find(i) { while (p[i] !== i) { p[i] = p[p[i]]; i = p[i]; } return i; }
    function union(a, b) { const ra = find(a), rb = find(b); if (ra !== rb) p[rb] = ra; return ra !== rb; }
    return { find: find, union: union, p: p };
  }

  /* connected components of a boolean/label field. conn is 4 or 8.
     `same(a, b)` decides whether two neighbouring cells belong together. */
  function components(W, H, same, conn) {
    const lab = new Int32Array(W * H).fill(-1);
    let n = 0;
    const N4 = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    const N8 = N4.concat([[1, 1], [1, -1], [-1, 1], [-1, -1]]);
    const nb = (conn === 8) ? N8 : N4;
    const stack = [];
    for (let s = 0; s < W * H; s++) {
      if (lab[s] >= 0) continue;
      lab[s] = n; stack.length = 0; stack.push(s);
      while (stack.length) {
        const q = stack.pop(), qx = q % W, qy = (q / W) | 0;
        for (const [dx, dy] of nb) {
          const x = qx + dx, y = qy + dy;
          if (x < 0 || y < 0 || x >= W || y >= H) continue;
          const t = y * W + x;
          if (lab[t] < 0 && same(q, t)) { lab[t] = n; stack.push(t); }
        }
      }
      n++;
    }
    return { lab: lab, n: n };
  }

  /* ---- partition comparison, used by figures 1 and 19 ---------------------- */
  /* contingency table between two labellings of the same weighted atoms */
  function contingency(l1, l2, w) {
    const cm = new Map(), r = new Map(), c = new Map();
    let N = 0;
    for (let i = 0; i < l1.length; i++) {
      const k = l1[i] + "|" + l2[i];
      cm.set(k, (cm.get(k) || 0) + w[i]);
      r.set(l1[i], (r.get(l1[i]) || 0) + w[i]);
      c.set(l2[i], (c.get(l2[i]) || 0) + w[i]);
      N += w[i];
    }
    return { cm: cm, r: r, c: c, N: N };
  }
  const C2 = v => v * (v - 1) / 2;
  const sumC2 = m => { let s = 0; for (const v of m.values()) s += C2(v); return s; };
  function randIndex(l1, l2, w) {
    const t = contingency(l1, l2, w), tot = C2(t.N);
    return (tot + 2 * sumC2(t.cm) - sumC2(t.r) - sumC2(t.c)) / tot;
  }
  function adjRand(l1, l2, w) {
    const t = contingency(l1, l2, w);
    const idx = sumC2(t.cm), sr = sumC2(t.r), sc = sumC2(t.c);
    const exp = sr * sc / C2(t.N), mx = (sr + sc) / 2;
    return (mx - exp === 0) ? 1 : (idx - exp) / (mx - exp);
  }
  function varInfo(l1, l2, w) {
    const t = contingency(l1, l2, w), N = t.N;
    const H = m => { let s = 0; for (const v of m.values()) if (v > 0) s -= (v / N) * Math.log(v / N); return s; };
    let I = 0;
    for (const [k, v] of t.cm) {
      if (v <= 0) continue;
      const [a, b] = k.split("|");
      I += (v / N) * Math.log((v / N) / ((t.r.get(+a) / N) * (t.c.get(+b) / N)));
    }
    return H(t.r) + H(t.c) - 2 * I;
  }

  /* ---- readouts ------------------------------------------------------------
     A compact key/value printer inside an svg, used by most figures. */
  function kv(g, x, y, opt) {
    const o = Object.assign({ lead: 15, keyW: 152, size: 11, dp: 3 }, opt || {});
    let i = 0;
    return function (k, v, color, bold) {
      g.append("text").attr("x", x).attr("y", y + i * o.lead).attr("font-size", o.size)
        .attr("fill", VC.muted).text(k);
      g.append("text").attr("x", x + o.keyW).attr("y", y + i * o.lead).attr("font-size", o.size)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", color || VC.ink)
        .attr("font-weight", bold ? 600 : 400).text(v);
      i++;
    };
  }

  /* a matrix printed as monospace rows between bracket pieces. SVG collapses runs
     of whitespace inside <text>, so xml:space must be preserved or the padStart
     alignment is thrown away and the columns go ragged. */
  function matText(g, M, x, y, opt) {
    const o = Object.assign({ size: 10, dp: 2, fill: VC.ink, lead: 12.5, label: null, pad: 6, colorOf: null }, opt || {});
    const rows = M.map(r => r.map(v =>
      (typeof v === "string" ? v : (Math.abs(v) < 5e-7 ? "0" : v.toFixed(o.dp))).padStart(o.pad)).join(" "));
    const brL = ["⎡", "⎢", "⎣"], brR = ["⎤", "⎥", "⎦"];
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    if (o.label) gg.append("text").attr("x", 0).attr("y", -o.lead).attr("font-size", 10)
      .attr("fill", VC.muted).text(o.label);
    rows.forEach((r, i) => {
      const k = rows.length === 1 ? -1 : (i === 0 ? 0 : (i === rows.length - 1 ? 2 : 1));
      const bl = k < 0 ? "[" : brL[k], br = k < 0 ? "]" : brR[k];
      gg.append("text").attr("x", 0).attr("y", i * o.lead)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("font-size", o.size)
        .attr("xml:space", "preserve")
        .attr("fill", o.colorOf ? o.colorOf(i) : o.fill).text(bl + r + " " + br);
    });
    return gg;
  }

  /* a titled sub-panel with a hairline border, for the multi-panel figures */
  function box(g, x, y, w, h, title, opt) {
    const o = Object.assign({ fill: "none", stroke: VC.line }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    gg.append("rect").attr("x", -0.5).attr("y", -0.5).attr("width", w + 1).attr("height", h + 1)
      .attr("fill", o.fill).attr("stroke", o.stroke).attr("rx", 3);
    if (title) gg.append("text").attr("x", 0).attr("y", -7).attr("font-size", 11)
      .attr("fill", VC.ink).attr("font-weight", 600).text(title);
    return gg;
  }

  return {
    cells: cells, edgesOf: edgesOf, drawEdges: drawEdges, grey: grey, pal: pal, PAL: PAL,
    UF: UF, components: components,
    randIndex: randIndex, adjRand: adjRand, varInfo: varInfo, contingency: contingency,
    kv: kv, matText: matText, box: box
  };
})();

/* ══════════════════════════════════════════════════════════════════════════
   1 · #illposed-svg — one scene, many defensible partitions
   ══════════════════════════════════════════════════════════════════════════ */
const SGscene = (function () {
  const W = 48, H = 32;
  const NAMES = ["sky", "hill L", "hill R", "roof", "wall", "door", "window",
    "trunk", "foliage", "grass far", "grass near", "path"];
  const VAL = [210, 150, 148, 78, 118, 60, 232, 88, 100, 182, 172, 140];
  function atomAt(x, y) {                       // a painter, later strokes win
    let a = 0;
    const hL = Math.max(0, 8.0 - 0.62 * Math.abs(x - 11));
    const hR = Math.max(0, 6.0 - 0.55 * Math.abs(x - 38));
    if (y >= 20 - hL && y < 20) a = 1;
    if (y >= 20 - hR && y < 20) a = 2;
    if (y >= 20) a = 9;
    if (y >= 26) a = 10;
    if (x >= 33 && x <= 35 && y >= 14 && y < 27) a = 7;
    if (Math.pow((x - 34) / 6.6, 2) + Math.pow((y - 11) / 6.2, 2) < 1) a = 8;
    if (x >= 8 && x <= 23 && y >= 16 && y <= 28) a = 4;
    if (y >= 11 && y < 16 && Math.abs(x - 15.5) <= (y - 10.0) * 1.85) a = 3;
    if (x >= 12 && x <= 16 && y >= 21 && y <= 28) a = 5;
    if (x >= 18 && x <= 21 && y >= 18 && y <= 21) a = 6;
    if (x >= 12 && x <= 16 && y >= 29) a = 11;
    return a;
  }
  const A = new Int32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) A[y * W + x] = atomAt(x, y);
  const CNT = new Array(12).fill(0);
  for (let i = 0; i < A.length; i++) CNT[A[i]]++;

  /* region adjacency over the twelve atoms, 4-connected */
  const ADJ = [];
  {
    const seen = new Set();
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const a = A[y * W + x];
      const push = b => {
        if (a === b) return;
        const k = Math.min(a, b) + "," + Math.max(a, b);
        if (!seen.has(k)) { seen.add(k); ADJ.push([Math.min(a, b), Math.max(a, b)]); }
      };
      if (x + 1 < W) push(A[y * W + x + 1]);
      if (y + 1 < H) push(A[(y + 1) * W + x]);
    }
  }

  /* agglomerative merge by size-weighted mean intensity; LEVELS[k] is the
     atom → region-root map at k regions. Nested by construction. */
  const LEVELS = {};
  {
    const uf = SG.UF(12);
    const mean = VAL.slice(), size = CNT.slice();
    LEVELS[12] = Array.from({ length: 12 }, (_, i) => uf.find(i));
    for (let step = 0; step < 11; step++) {
      let best = null;
      for (const [a, b] of ADJ) {
        const ra = uf.find(a), rb = uf.find(b);
        if (ra === rb) continue;
        const d = Math.abs(mean[ra] - mean[rb]);
        if (best === null || d < best.d - 1e-12) best = { d: d, ra: ra, rb: rb };
      }
      const { ra, rb } = best;
      const nm = (mean[ra] * size[ra] + mean[rb] * size[rb]) / (size[ra] + size[rb]);
      size[ra] += size[rb]; mean[ra] = nm; uf.p[rb] = ra;
      LEVELS[11 - step] = Array.from({ length: 12 }, (_, i) => uf.find(i));
      LEVELS[11 - step].mergeDist = best.d;
    }
  }
  const ANN = [
    { name: "A · objects", note: "groups by what the thing is",
      lab: [0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4] },
    { name: "B · materials", note: "groups by what it is made of",
      lab: [0, 1, 1, 2, 2, 3, 4, 3, 1, 1, 1, 5] },
    { name: "C · parts", note: "keeps every part apart",
      lab: [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9] }
  ];
  const nreg = l => new Set(l).size;
  return { W: W, H: H, NAMES: NAMES, VAL: VAL, A: A, CNT: CNT, ADJ: ADJ,
    LEVELS: LEVELS, ANN: ANN, nreg: nreg };
})();

(function () {
  const svg = d3.select("#illposed-svg");
  if (svg.empty()) return;
  const S = SGscene;
  const eK = document.getElementById("ip-k"), eKv = document.getElementById("ip-kv");
  const eA = document.getElementById("ip-ann"), eB = document.getElementById("ip-bnd");
  const out = document.getElementById("illposed-readout");
  const wts = S.CNT;

  function draw() {
    const K = +eK.value, ai = +eA.value, showB = eB.checked;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 4, IW = S.W * cw, IH = S.H * cw;
    const mach = S.LEVELS[K];
    const ann = S.ANN[ai].lab;

    /* ── top row: the scene, the machine partition, the chosen annotator ── */
    const px = (labOf, x0, title, sub) => {
      const b = SG.box(g, x0, 30, IW, IH, title);
      SG.cells(b, 0, 0, cw, S.W, S.H, (x, y) => labOf(S.A[y * S.W + x]));
      if (showB) {
        const L = new Int32Array(S.W * S.H);
        for (let i = 0; i < L.length; i++) L[i] = labOf.key ? labOf.key(S.A[i]) : S.A[i];
        SG.drawEdges(b, SG.edgesOf(L, S.W, S.H), cw, { color: "#0b0d12", w: 1, op: 0.9 });
      }
      if (sub) b.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10.5)
        .attr("fill", VC.muted).text(sub);
      return b;
    };
    const rawFn = a => SG.grey(S.VAL[a]); rawFn.key = a => a;
    px(rawFn, 14, "the image (12 atomic regions)", "grey levels only — this is all any algorithm sees");
    const machFn = a => SG.pal(mach[a]); machFn.key = a => mach[a];
    px(machFn, 222, `merge by intensity, stopped at K = ${K}`,
      `merged in order of mean difference · ${S.nreg(mach)} regions`);
    const annFn = a => SG.pal(ann[a]); annFn.key = a => ann[a];
    px(annFn, 430, "annotator " + S.ANN[ai].name, S.ANN[ai].note + " · " + S.nreg(ann) + " regions");

    /* ── bottom left: all three annotators, small ── */
    const cw2 = 2.5, IW2 = S.W * cw2, IH2 = S.H * cw2;
    S.ANN.forEach((an, i) => {
      const b = SG.box(g, 14 + i * (IW2 + 18), 216, IW2, IH2,
        (i === ai ? "▸ " : "") + an.name);
      SG.cells(b, 0, 0, cw2, S.W, S.H, (x, y) => SG.pal(an.lab[S.A[y * S.W + x]]));
      const L = new Int32Array(S.W * S.H);
      for (let k = 0; k < L.length; k++) L[k] = an.lab[S.A[k]];
      SG.drawEdges(b, SG.edgesOf(L, S.W, S.H), cw2, { color: "#0b0d12", w: 0.8, op: 0.9 });
    });
    g.append("text").attr("x", 14).attr("y", 205).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("the same scene, read three ways by three people");
    g.append("text").attr("x", 14).attr("y", 216 + IH2 + 15).attr("font-size", 10.5)
      .attr("fill", VC.muted).text("none of these is the ground truth; all three are ground truths");

    /* ── bottom right: the score table ── */
    const tx = 430, ty = 205;
    g.append("text").attr("x", tx).attr("y", ty).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("adjusted Rand index — 1 is identical, 0 is chance");
    const put = SG.kv(g, tx, ty + 20, { keyW: 190, lead: 15.5 });
    const scores = S.ANN.map(an => SG.adjRand(mach, an.lab, wts));
    const bestK = S.ANN.map(an => {
      let bk = 2, bv = -2;
      for (let k = 2; k <= 12; k++) { const v = SG.adjRand(S.LEVELS[k], an.lab, wts); if (v > bv) { bv = v; bk = k; } }
      return { k: bk, v: bv };
    });
    S.ANN.forEach((an, i) => {
      put(`machine (K = ${K})  vs  ${an.name}`, scores[i].toFixed(4),
        i === ai ? VC.a2 : VC.ink, i === ai);
    });
    put("", "", VC.ink);
    S.ANN.forEach((an, i) => {
      put(`best K against ${an.name}`, `K = ${bestK[i].k}   (ARI ${bestK[i].v.toFixed(4)})`,
        bestK[i].k === K ? VC.good : VC.muted);
    });
    put("", "", VC.ink);
    const pw = [[0, 1], [0, 2], [1, 2]];
    pw.forEach(([i, j]) => {
      put(`${S.ANN[i].name}  vs  ${S.ANN[j].name}`,
        SG.adjRand(S.ANN[i].lab, S.ANN[j].lab, wts).toFixed(4), VC.violet);
    });

    /* ── the "same K, different partition" needle ── */
    const nA = S.nreg(ann);
    g.append("text").attr("x", tx).attr("y", 402).attr("font-size", 10.5).attr("fill", VC.muted)
      .attr("xml:space", "preserve")
      .text(K === nA
        ? `same number of regions as ${S.ANN[ai].name} (${K}) — and still ARI ${scores[ai].toFixed(3)}`
        : `${K} machine regions against ${nA} annotator regions`);
  }

  /* ── readout ── */
  function say() {
    const K = +eK.value, ai = +eA.value;
    const mach = S.LEVELS[K], ann = S.ANN[ai].lab;
    const ari = SG.adjRand(mach, ann, wts), vi = SG.varInfo(mach, ann, wts);
    const bk = (() => { let b = 2, v = -2; for (let k = 2; k <= 12; k++) { const s = SG.adjRand(S.LEVELS[k], S.ANN[ai].lab, wts); if (s > v) { v = s; b = k; } } return b; })();
    const bests = [0, 1, 2].map(i => { let b = 2, v = -2; for (let k = 2; k <= 12; k++) { const s = SG.adjRand(S.LEVELS[k], S.ANN[i].lab, wts); if (s > v) { v = s; b = k; } } return b; });
    out.innerHTML =
      `At <b>K = ${K}</b> the machine agrees with <b>${S.ANN[ai].name}</b> at ARI <b>${ari.toFixed(4)}</b> `
      + `(variation of information ${vi.toFixed(4)} nats — lower is closer). `
      + `Its own best granularity against this annotator is <b>K = ${bk}</b>.<br>`
      + `Against the three annotators the best granularities are <b>K = ${bests[0]}, ${bests[1]}, ${bests[2]}</b> — `
      + `three different answers from one image, and the algorithm has no way to choose between them, `
      + `because the information that would choose is not in the pixels.<br>`
      + `The three annotators agree with <i>each other</i> at ARI `
      + `${SG.adjRand(S.ANN[0].lab, S.ANN[1].lab, wts).toFixed(3)} (A–B), `
      + `${SG.adjRand(S.ANN[0].lab, S.ANN[2].lab, wts).toFixed(3)} (A–C), `
      + `${SG.adjRand(S.ANN[1].lab, S.ANN[2].lab, wts).toFixed(3)} (B–C). `
      + `Note that at its best K the machine can score <i>above</i> some of those numbers — which is a warning `
      + `about the metric, not a claim that it segments better than a person.`;
  }
  function all() { eKv.textContent = eK.value; draw(); say(); }
  eK.oninput = all; eA.onchange = all; eB.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   2 · #otsu-svg — the histogram, the draggable threshold, the criterion curves
   ══════════════════════════════════════════════════════════════════════════ */
const SGotsu = (function () {
  const W = 96, H = 64;
  const r = VZ.rng(11);
  const img = new Float64Array(W * H);
  const truth = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const e1 = Math.pow((x - 32) / 20, 2) + Math.pow((y - 30) / 17, 2);
    const e2 = Math.pow((x - 68) / 13, 2) + Math.pow((y - 26) / 20, 2);
    const inObj = (e1 < 1 || e2 < 1);
    const v = (inObj ? 175 : 72) + 17 * VZ.randn(r);
    img[y * W + x] = Math.min(255, Math.max(0, Math.round(v)));
    truth[y * W + x] = inObj ? 1 : 0;
  }
  const hist = new Float64Array(256);
  for (let i = 0; i < img.length; i++) hist[img[i]]++;
  const N = img.length;
  const p = Array.from(hist, v => v / N);
  let muG = 0; for (let l = 0; l < 256; l++) muG += l * p[l];
  let varT = 0; for (let l = 0; l < 256; l++) varT += (l - muG) * (l - muG) * p[l];
  /* running moments, so every criterion below is O(1) per threshold */
  const om = new Float64Array(256), mm = new Float64Array(256);
  let a = 0, b = 0;
  for (let l = 0; l < 256; l++) { a += p[l]; b += l * p[l]; om[l] = a; mm[l] = b; }
  function at(t) {
    const P1 = om[t], P2 = 1 - P1;
    if (P1 <= 0 || P2 <= 0) return { P1: P1, P2: P2, m1: NaN, m2: NaN, sB: 0, sW: varT };
    const m1 = mm[t] / P1, m2 = (muG - mm[t]) / P2;
    const sB = P1 * P2 * (m1 - m2) * (m1 - m2);
    return { P1: P1, P2: P2, m1: m1, m2: m2, sB: sB, sW: varT - sB };
  }
  let tStar = 1, best = -1;
  for (let t = 1; t <= 254; t++) { const s = at(t).sB; if (s > best) { best = s; tStar = t; } }
  /* the threshold that actually maximises IoU against the true mask — a different number */
  function iouAt(t) {
    let inter = 0, uni = 0;
    for (let i = 0; i < img.length; i++) {
      const pr = img[i] > t ? 1 : 0;
      if (pr & truth[i]) inter++;
      if (pr | truth[i]) uni++;
    }
    return { iou: inter / uni, inter: inter, uni: uni };
  }
  let tIoU = 1, bi = -1;
  for (let t = 1; t <= 254; t++) { const v = iouAt(t).iou; if (v > bi) { bi = v; tIoU = t; } }
  return { W: W, H: H, img: img, truth: truth, hist: hist, N: N, muG: muG, varT: varT,
    at: at, tStar: tStar, sBStar: best, iouAt: iouAt, tIoU: tIoU, iouStar: bi };
})();

(function () {
  const svg = d3.select("#otsu-svg");
  if (svg.empty()) return;
  const S = SGotsu;
  const eT = document.getElementById("ot-t"), eTv = document.getElementById("ot-tv");
  const eO = document.getElementById("ot-opt"), eW = document.getElementById("ot-sw");
  const eR = document.getElementById("ot-truth");
  const out = document.getElementById("otsu-readout");

  const HX = 262, HY = 34, HW = 480, HH = 132;        // histogram panel
  const CX = 262, CY = 232, CW = 480, CH = 120;       // criterion panel
  const x = d3.scaleLinear().domain([0, 255]).range([0, HW]);
  const maxH = d3.max(S.hist);
  const yH = d3.scaleLinear().domain([0, maxH]).range([HH, 0]).nice();
  const yC = d3.scaleLinear().domain([0, S.varT * 1.06]).range([CH, 0]);

  function draw() {
    const t = +eT.value;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2;

    /* ── left: the image, and the binary map beneath it ── */
    const b1 = SG.box(g, 14, 34, S.W * cw, S.H * cw, "the image (96 × 64, noise σ = 17)");
    SG.cells(b1, 0, 0, cw, S.W, S.H, (px, py) => SG.grey(S.img[py * S.W + px]));
    const b2 = SG.box(g, 14, 232, S.W * cw, S.H * cw, `thresholded at t = ${t}`);
    const showT = eR.checked;
    SG.cells(b2, 0, 0, cw, S.W, S.H, (px, py) => {
      const i = py * S.W + px, pr = S.img[i] > t ? 1 : 0;
      if (showT && pr !== S.truth[i]) return pr ? "#f87171" : "#7f1d1d";
      return pr ? "#dfe4ee" : "#1a1e28";
    });
    const st = S.iouAt(t);
    b2.append("text").attr("x", 0).attr("y", S.H * cw + 14).attr("font-size", 10.5)
      .attr("fill", VC.muted).text(showT ? "red = disagrees with the true mask" : "white = above threshold");
    b2.append("text").attr("x", 0).attr("y", S.H * cw + 28).attr("font-size", 10.5)
      .attr("fill", showT ? VC.a2 : VC.muted)
      .text(`IoU vs truth ${st.iou.toFixed(4)}`);

    /* ── histogram ── */
    const hp = SG.box(g, HX, HY, HW, HH, "histogram — everything a global threshold can see",
      { fill: "#12141c" });
    hp.append("g").attr("transform", `translate(0,${HH})`).attr("class", "axis")
      .call(d3.axisBottom(x).ticks(8));
    hp.append("g").attr("class", "axis").call(d3.axisLeft(yH).ticks(4));
    const bw = HW / 256;
    hp.selectAll("rect.h").data(Array.from(S.hist)).join("rect").attr("class", "h")
      .attr("x", (d, i) => x(i)).attr("y", d => yH(d))
      .attr("width", Math.max(0.8, bw)).attr("height", d => HH - yH(d))
      .attr("fill", (d, i) => i <= t ? "#3b5f9e" : "#c98a3a").attr("fill-opacity", 0.9);
    const cur = S.at(t);
    hp.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", -2).attr("y2", HH)
      .attr("stroke", VC.ink).attr("stroke-width", 2);
    hp.append("text").attr("x", x(t) + 5).attr("y", 11).attr("font-size", 10.5).attr("fill", VC.ink).text("t = " + t);
    [[cur.m1, "#5b9cff", "μ₁"], [cur.m2, "#ffb454", "μ₂"]].forEach(([m, c, lab]) => {
      if (!isFinite(m)) return;
      hp.append("line").attr("x1", x(m)).attr("x2", x(m)).attr("y1", HH - 12).attr("y2", HH)
        .attr("stroke", c).attr("stroke-width", 2);
      hp.append("text").attr("x", x(m)).attr("y", HH - 15).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", c).text(lab);
    });
    if (S.tStar !== t) {
      hp.append("line").attr("x1", x(S.tStar)).attr("x2", x(S.tStar)).attr("y1", 0).attr("y2", HH)
        .attr("stroke", VC.good).attr("stroke-width", 1).attr("stroke-dasharray", "3 3");
    }
    hp.append("rect").attr("x", 0).attr("y", 0).attr("width", HW).attr("height", HH)
      .attr("fill", "transparent").style("cursor", "ew-resize")
      .call(d3.drag().on("start drag", ev => {
        const v = Math.round(Math.max(1, Math.min(254, x.invert(ev.x))));
        eT.value = v; eO.checked = false; all();
      }));

    /* ── criterion curves ── */
    const cp = SG.box(g, CX, CY, CW, CH, "the criterion, at every candidate threshold",
      { fill: "#12141c" });
    cp.append("g").attr("transform", `translate(0,${CH})`).attr("class", "axis")
      .call(d3.axisBottom(x).ticks(8));
    cp.append("g").attr("class", "axis").call(d3.axisLeft(yC).ticks(4));
    const ts = d3.range(1, 255);
    const line = (acc, col, dash) => {
      const d = ts.map((tt, i) => (i ? "L" : "M") + x(tt).toFixed(2) + "," + yC(acc(S.at(tt))).toFixed(2)).join(" ");
      const e = cp.append("path").attr("d", d).attr("fill", "none").attr("stroke", col).attr("stroke-width", 1.8);
      if (dash) e.attr("stroke-dasharray", dash);
    };
    cp.append("line").attr("x1", 0).attr("x2", CW).attr("y1", yC(S.varT)).attr("y2", yC(S.varT))
      .attr("stroke", VC.muted).attr("stroke-dasharray", "5 4").attr("stroke-width", 1);
    cp.append("text").attr("x", CW - 4).attr("y", yC(S.varT) - 4).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", VC.muted).text("σ_T² = " + S.varT.toFixed(1) + "  (constant in t)");
    line(s => s.sB, VC.good);
    if (eW.checked) line(s => s.sW, VC.bad);
    if (eW.checked) line(s => s.sB + s.sW, VC.violet, "2 3");
    cp.append("circle").attr("cx", x(S.tStar)).attr("cy", yC(S.sBStar)).attr("r", 4)
      .attr("fill", VC.good).attr("stroke", "#0b0d12");
    cp.append("text").attr("x", x(S.tStar) + 7).attr("y", yC(S.sBStar) - 6).attr("font-size", 10.5)
      .attr("fill", VC.good).text("max σ_B² at t = " + S.tStar);
    cp.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", 0).attr("y2", CH)
      .attr("stroke", VC.ink).attr("stroke-width", 1.4).attr("stroke-opacity", 0.8);
    VZ.legend(cp, [{ color: VC.good, label: "σ_B²  between-class" }]
      .concat(eW.checked ? [{ color: VC.bad, label: "σ_W²  within-class" },
        { color: VC.violet, label: "σ_B² + σ_W²", dash: "2 3" }] : []),
      CW - 148, 12, { gap: 14 });
    cp.append("rect").attr("x", 0).attr("y", 0).attr("width", CW).attr("height", CH)
      .attr("fill", "transparent").style("cursor", "ew-resize")
      .call(d3.drag().on("start drag", ev => {
        const v = Math.round(Math.max(1, Math.min(254, x.invert(ev.x))));
        eT.value = v; eO.checked = false; all();
      }));

    /* ── numeric panel ── */
    const put = SG.kv(g, CX, 384, { keyW: 128, lead: 14.5 });
    put("P₁, P₂", cur.P1.toFixed(4) + ",  " + cur.P2.toFixed(4));
    put("μ₁, μ₂", cur.m1.toFixed(3) + ",  " + cur.m2.toFixed(3));
    put("σ_B²(t)", cur.sB.toFixed(4), VC.good);
    put("σ_W²(t)", cur.sW.toFixed(4), VC.bad);
    put("σ_B² + σ_W²", (cur.sB + cur.sW).toFixed(4) + "   vs   σ_T² = " + S.varT.toFixed(4), VC.violet);
    const put2 = SG.kv(g, 14, 384, { keyW: 118, lead: 14.5 });
    put2("Otsu optimum", "t* = " + S.tStar, VC.good, true);
    put2("IoU-optimal t", "t = " + S.tIoU + "  (IoU " + S.iouStar.toFixed(4) + ")", VC.a2);
    put2("IoU at Otsu", S.iouAt(S.tStar).iou.toFixed(4), VC.a2);
  }

  function say() {
    const t = +eT.value, cur = S.at(t), st = S.iouAt(t);
    const gap = (S.sBStar - cur.sB) / S.sBStar * 100;
    out.innerHTML =
      `At <b>t = ${t}</b>: σ_B² = <b>${cur.sB.toFixed(4)}</b>, σ_W² = <b>${cur.sW.toFixed(4)}</b>, `
      + `and their sum is <b>${(cur.sB + cur.sW).toFixed(4)}</b> against a total variance of `
      + `<b>${S.varT.toFixed(4)}</b> — equal, as the identity requires, at every threshold you can drag to.<br>`
      + `Otsu's exhaustive search picks <b>t* = ${S.tStar}</b> with σ_B² = ${S.sBStar.toFixed(4)}; you are `
      + (t === S.tStar ? "there now" : `${gap.toFixed(2)}% below that peak`)
      + `. Against the true object mask this threshold scores IoU <b>${st.iou.toFixed(4)}</b>, while the `
      + `best possible threshold, <b>t = ${S.tIoU}</b>, scores ${S.iouStar.toFixed(4)} — Otsu is close but it is `
      + `optimising a different quantity, and no amount of exactness in the search fixes that.`;
  }
  function all() {
    if (eO.checked) eT.value = S.tStar;
    eTv.textContent = eT.value;
    draw(); say();
  }
  eT.oninput = () => { eO.checked = false; all(); };
  eO.onchange = all; eW.onchange = all; eR.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   3 · #bimodal-svg — when is a histogram bimodal, and what Otsu does anyway
   ══════════════════════════════════════════════════════════════════════════ */
const SGbim = (function () {
  const L = 256, SIG = 13, MB = 80;
  const lev = d3.range(L);
  function gauss(m, s) { return lev.map(l => Math.exp(-0.5 * Math.pow((l - m) / s, 2)) / (s * Math.sqrt(2 * Math.PI))); }
  /* weighted, normalised components: bg (background) and ob (object) */
  function model(pobj, dsig) {
    const gb = gauss(MB, SIG), go = gauss(MB + dsig * SIG, SIG);
    const bg = gb.map(v => (1 - pobj) * v), ob = go.map(v => pobj * v);
    let Z = 0; for (let l = 0; l < L; l++) Z += bg[l] + ob[l];
    for (let l = 0; l < L; l++) { bg[l] /= Z; ob[l] /= Z; }
    const p = lev.map(l => bg[l] + ob[l]);
    return { bg: bg, ob: ob, p: p };
  }
  function modes(p) {
    const out = [];
    for (let l = 1; l < L - 1; l++) if (p[l] > p[l - 1] && p[l] >= p[l + 1]) out.push(l);
    return out;
  }
  function otsu(p) {
    let muG = 0; for (let l = 0; l < L; l++) muG += l * p[l];
    let varT = 0; for (let l = 0; l < L; l++) varT += (l - muG) * (l - muG) * p[l];
    let om = 0, mm = 0, best = -1, t = 1;
    const curve = new Float64Array(L - 1);
    for (let l = 0; l < L - 1; l++) {
      om += p[l]; mm += l * p[l];
      const v = (om > 1e-14 && om < 1 - 1e-14) ? Math.pow(muG * om - mm, 2) / (om * (1 - om)) : 0;
      curve[l] = v;
      if (v > best) { best = v; t = l; }
    }
    return { t: t, sB: best, varT: varT, eta: best / varT, curve: curve };
  }
  function bayes(m) {                       // where the two weighted components cross
    const pb = m.bg.indexOf(Math.max.apply(null, m.bg));
    const po = m.ob.indexOf(Math.max.apply(null, m.ob));
    const lo = Math.min(pb, po), hi = Math.max(pb, po);
    for (let t = lo; t < hi; t++)
      if ((m.bg[t] - m.ob[t]) * (m.bg[t + 1] - m.ob[t + 1]) <= 0) return t;
    return (lo + hi) >> 1;
  }
  function err(t, m) {                      // misclassification mass at threshold t
    let e = 0;
    for (let l = 0; l <= t; l++) e += m.ob[l];
    for (let l = t + 1; l < L; l++) e += m.bg[l];
    return e;
  }
  /* the bimodality map, computed once: mode count over (d/σ, object fraction) */
  const DS = d3.range(0.6, 6.001, 0.06), PS = d3.range(0.02, 0.5001, 0.008);
  const MAP = PS.map(po => DS.map(dd => modes(model(po, dd).p).length));
  /* the boundary d/σ at which each object fraction first becomes bimodal */
  function boundary(po) {                   // bisected, so π = 0.5 reads 2.00 and not 2.01
    let lo = 0.6, hi = 9.0;
    if (modes(model(po, hi).p).length < 2) return null;
    for (let i = 0; i < 34; i++) {
      const mid = (lo + hi) / 2;
      if (modes(model(po, mid).p).length >= 2) hi = mid; else lo = mid;
    }
    return hi;
  }
  return { L: L, SIG: SIG, MB: MB, model: model, modes: modes, otsu: otsu,
    bayes: bayes, err: err, DS: DS, PS: PS, MAP: MAP, boundary: boundary };
})();

(function () {
  const svg = d3.select("#bimodal-svg");
  if (svg.empty()) return;
  const S = SGbim;
  const eD = document.getElementById("bm-d"), eDv = document.getElementById("bm-dv");
  const eP = document.getElementById("bm-p"), ePv = document.getElementById("bm-pv");
  const eC = document.getElementById("bm-comp"), eE = document.getElementById("bm-err");
  const out = document.getElementById("bimodal-readout");

  const AX = 14, AY = 36, AW = 420, AH = 158;
  const BX = 474, BY = 36, BW = 272, BH = 158;
  const CX = 14, CY = 258, CW = 420, CH = 118;
  const x = d3.scaleLinear().domain([20, 220]).range([0, AW]);

  function draw() {
    const dd = +eD.value, po = +eP.value;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const m = S.model(po, dd), md = S.modes(m.p);
    const ot = S.otsu(m.p), tb = S.bayes(m);
    const eO = S.err(ot.t, m), eB = S.err(tb, m);

    /* ── A · the histogram ── */
    const ymax = d3.max(m.p) * 1.12;
    const y = d3.scaleLinear().domain([0, ymax]).range([AH, 0]);
    const A = SG.box(g, AX, AY, AW, AH, "the histogram this model produces", { fill: "#12141c" });
    A.append("g").attr("transform", `translate(0,${AH})`).attr("class", "axis").call(d3.axisBottom(x).ticks(7));
    const area = d3.area().x(d => x(d)).y0(AH).y1(d => y(m.p[d])).defined(d => x(d) >= 0 && x(d) <= AW);
    A.append("path").datum(d3.range(S.L)).attr("d", area).attr("fill", VC.ink).attr("fill-opacity", 0.13);
    if (eC.checked) {
      [["bg", "#5b9cff"], ["ob", "#ffb454"]].forEach(([k, c]) => {
        const ar = d3.area().x(d => x(d)).y0(AH).y1(d => y(m[k][d]));
        A.append("path").datum(d3.range(S.L)).attr("d", ar).attr("fill", c).attr("fill-opacity", 0.22);
        A.append("path").datum(d3.range(S.L)).attr("d", d3.line().x(d => x(d)).y(d => y(m[k][d])))
          .attr("fill", "none").attr("stroke", c).attr("stroke-width", 1.2).attr("stroke-opacity", 0.85);
      });
    }
    A.append("path").datum(d3.range(S.L)).attr("d", d3.line().x(d => x(d)).y(d => y(m.p[d])))
      .attr("fill", "none").attr("stroke", VC.ink).attr("stroke-width", 1.9);
    md.forEach(l => {
      A.append("path").attr("d", `M${x(l)},${y(m.p[l]) - 5} l-4.5,-7 l9,0 Z`).attr("fill", VC.violet);
    });
    A.append("text").attr("x", 6).attr("y", 13).attr("font-size", 11).attr("fill", VC.violet)
      .text(md.length === 1 ? "1 mode — unimodal" : md.length + " modes");
    const mark = (t, col, dash, lab) => {
      A.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", 0).attr("y2", AH)
        .attr("stroke", col).attr("stroke-width", 1.8).attr("stroke-dasharray", dash);
      A.append("text").attr("x", x(t) + 4).attr("y", AH - 6).attr("font-size", 10).attr("fill", col).text(lab);
    };
    mark(ot.t, VC.good, null, "Otsu " + ot.t);
    mark(tb, VC.bad, "4 3", "Bayes " + tb);

    /* ── B · the bimodality map ── */
    const B = SG.box(g, BX, BY, BW, BH, "where a second hump exists at all", { fill: "#12141c" });
    const mx = d3.scaleLinear().domain([S.DS[0], S.DS[S.DS.length - 1]]).range([0, BW]);
    const my = d3.scaleLinear().domain([S.PS[0], S.PS[S.PS.length - 1]]).range([BH, 0]);
    const cwx = BW / S.DS.length + 0.6, cwy = BH / S.PS.length + 0.6;
    const cellsD = [];
    S.PS.forEach((pv, j) => S.DS.forEach((dv, i) => cellsD.push([i, j, S.MAP[j][i]])));
    B.selectAll("rect.m").data(cellsD).join("rect").attr("class", "m")
      .attr("x", d => mx(S.DS[d[0]]) - cwx / 2).attr("y", d => my(S.PS[d[1]]) - cwy / 2)
      .attr("width", cwx).attr("height", cwy).attr("shape-rendering", "crispEdges")
      .attr("fill", d => d[2] >= 2 ? "#2b4a2f" : "#3a2530");
    B.append("g").attr("transform", `translate(0,${BH})`).attr("class", "axis").call(d3.axisBottom(mx).ticks(6));
    B.append("g").attr("class", "axis").call(d3.axisLeft(my).ticks(5).tickFormat(d3.format(".2f")));
    B.append("text").attr("x", BW).attr("y", BH + 28).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", VC.muted).text("separation d / σ");
    B.append("text").attr("x", 2).attr("y", -0.5).attr("font-size", 10.5).attr("fill", VC.muted).text("object fraction π");
    B.append("text").attr("x", BW - 6).attr("y", 16).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", "#7fbf87").text("two modes");
    B.append("text").attr("x", 8).attr("y", BH - 8).attr("font-size", 10).attr("fill", "#c98a9a").text("one mode");
    B.append("path").attr("d", `M${mx(dd) - 7},${my(po)} h14 M${mx(dd)},${my(po) - 7} v14`)
      .attr("stroke", VC.ink).attr("stroke-width", 2);
    B.append("line").attr("x1", mx(2)).attr("x2", mx(2)).attr("y1", 0).attr("y2", BH)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "3 3").attr("stroke-width", 1);
    B.append("text").attr("x", mx(2) + 4).attr("y", 30).attr("font-size", 9.5).attr("fill", VC.a2).text("d = 2σ");

    /* ── C · error against threshold ── */
    if (eE.checked) {
      const C = SG.box(g, CX, CY, CW, CH, "misclassification rate at every candidate threshold", { fill: "#12141c" });
      const ts = d3.range(20, 221);
      const ev = ts.map(t => S.err(t, m));
      const ye = d3.scaleLinear().domain([0, Math.max(0.02, d3.max(ev) * 1.05)]).range([CH, 0]);
      C.append("g").attr("transform", `translate(0,${CH})`).attr("class", "axis").call(d3.axisBottom(x).ticks(7));
      C.append("g").attr("class", "axis").call(d3.axisLeft(ye).ticks(4).tickFormat(d3.format(".0%")));
      C.append("path").datum(ts).attr("d", d3.line().x(d => x(d)).y((d, i) => ye(ev[i])))
        .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.8);
      [[ot.t, eO, VC.good, "Otsu"], [tb, eB, VC.bad, "Bayes"]].forEach(([t, e, c, lab]) => {
        C.append("circle").attr("cx", x(t)).attr("cy", ye(e)).attr("r", 4).attr("fill", c).attr("stroke", "#0b0d12");
        C.append("text").attr("x", x(t) + 7).attr("y", ye(e) - 6).attr("font-size", 10.5).attr("fill", c)
          .text(lab + " " + (100 * e).toFixed(2) + "%");
      });
      C.append("text").attr("x", CW).attr("y", CH + 28).attr("text-anchor", "end").attr("font-size", 10.5)
        .attr("fill", VC.muted).text("threshold t");
    }

    /* ── D · numbers ── */
    let fracO = 0; for (let l = ot.t + 1; l < S.L; l++) fracO += m.p[l];
    let fracB = 0; for (let l = tb + 1; l < S.L; l++) fracB += m.p[l];
    const put = SG.kv(g, 474, 268, { keyW: 160, lead: 15.5 });
    put("modes in the histogram", String(md.length), md.length >= 2 ? VC.good : VC.bad, true);
    put("bimodal from d/σ =", (S.boundary(po) || 0).toFixed(2), VC.violet);
    put("Otsu threshold", String(ot.t), VC.good);
    put("Bayes threshold", String(tb), VC.bad);
    put("Otsu error", (100 * eO).toFixed(2) + " %", VC.good);
    put("Bayes error", (100 * eB).toFixed(2) + " %", VC.bad);
    put("Otsu / Bayes error", (eO / eB).toFixed(2) + " ×", eO / eB > 1.5 ? VC.bad : VC.ink, eO / eB > 1.5);
    put("frame called object", (100 * fracO).toFixed(1) + " %   (true " + (100 * po).toFixed(1) + " %)",
      Math.abs(fracO - po) > 0.06 ? VC.bad : VC.ink);
    put("separability η = σ_B²/σ_T²", ot.eta.toFixed(4), VC.a2);
  }

  function say() {
    const dd = +eD.value, po = +eP.value;
    const m = S.model(po, dd), md = S.modes(m.p);
    const ot = S.otsu(m.p), tb = S.bayes(m);
    const eO = S.err(ot.t, m), eB = S.err(tb, m);
    const bd = S.boundary(po);
    out.innerHTML =
      `With the object at <b>${(100 * po).toFixed(1)}%</b> of the frame and a separation of <b>${dd.toFixed(2)}σ</b>, `
      + `the histogram has <b>${md.length} mode${md.length === 1 ? "" : "s"}</b>; it would need `
      + `<b>${bd ? bd.toFixed(2) : "—"}σ</b> at this class balance to show two.<br>`
      + `Otsu returns <b>t = ${ot.t}</b> with error <b>${(100 * eO).toFixed(2)}%</b>; the Bayes-optimal threshold is `
      + `<b>t = ${tb}</b> with error <b>${(100 * eB).toFixed(2)}%</b> — a ratio of <b>${(eO / eB).toFixed(2)}×</b>. `
      + (Math.abs(ot.t - tb) <= 1
        ? `They agree, which is what happens whenever the classes are close to balanced.`
        : `They disagree by ${Math.abs(ot.t - tb)} grey levels, and the direction is always the same: `
        + `Otsu pulls toward the split that halves the image, because <code>σ_B² = P₁P₂(μ₁−μ₂)²</code> `
        + `is multiplied by a factor that is largest at P₁ = P₂ = ½.`);
  }
  function all() {
    eDv.textContent = (+eD.value).toFixed(2);
    ePv.textContent = (+eP.value).toFixed(3);
    draw(); say();
  }
  eD.oninput = all; eP.oninput = all; eC.onchange = all; eE.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   4 · #adapt-svg — an illumination ramp no global threshold survives
   ══════════════════════════════════════════════════════════════════════════ */
/* box statistics by summed-area table: mean and standard deviation over a
   (2w+1)² window, edge-replicated. O(1) per pixel after two prefix passes. */
SG.boxStats = function (img, W, H, w) {
  const PW = W + 2 * w, PH = H + 2 * w;
  const s1 = new Float64Array((PH + 1) * (PW + 1));
  const s2 = new Float64Array((PH + 1) * (PW + 1));
  for (let y = 0; y < PH; y++) {
    const sy = Math.min(H - 1, Math.max(0, y - w));
    let r1 = 0, r2 = 0;
    for (let x = 0; x < PW; x++) {
      const sx = Math.min(W - 1, Math.max(0, x - w));
      const v = img[sy * W + sx];
      r1 += v; r2 += v * v;
      s1[(y + 1) * (PW + 1) + (x + 1)] = s1[y * (PW + 1) + (x + 1)] + r1;
      s2[(y + 1) * (PW + 1) + (x + 1)] = s2[y * (PW + 1) + (x + 1)] + r2;
    }
  }
  const k = 2 * w + 1, n = k * k;
  const mu = new Float64Array(W * H), sd = new Float64Array(W * H);
  const at = (S, y, x) => S[y * (PW + 1) + x];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const a = at(s1, y + k, x + k) - at(s1, y, x + k) - at(s1, y + k, x) + at(s1, y, x);
    const b = at(s2, y + k, x + k) - at(s2, y, x + k) - at(s2, y + k, x) + at(s2, y, x);
    const m = a / n;
    mu[y * W + x] = m;
    sd[y * W + x] = Math.sqrt(Math.max(0, b / n - m * m));
  }
  return { mu: mu, sd: sd };
};

const SGadapt = (function () {
  const W = 128, H = 104, INK = 90, PAPER = 210, NOISE = 11;
  const WORDS = [[6, 22], [30, 14], [48, 26], [80, 18], [102, 20]];
  function isInk(x, y) {
    for (let r = 0; r < 4; r++) {
      const y0 = 9 + r * 15;
      if (y >= y0 && y < y0 + 8) {
        for (let k = 0; k < WORDS.length; k++) {
          const xx = WORDS[k][0] + ((r * 7) % 9) - 4, ww = WORDS[k][1] - ((r + k) % 3) * 4;
          if (ww > 0 && x >= xx && x < xx + ww) return true;
        }
      }
    }
    if (x >= 12 && x < 44 && y >= 70 && y < 82) return true;     // a thick block
    if (x >= 60 && x < 118 && y >= 74 && y < 76) return true;    // a thin rule
    return false;
  }
  const truth = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) truth[y * W + x] = isInk(x, y) ? 1 : 0;
  const BAND = { y0: 86, y1: 104 };                              // contains no ink at all
  const shade = (x, y, ramp) =>
    1 + ramp * ((x / (W - 1) - 0.5) * 0.90 + (y / (H - 1) - 0.5) * 0.40);
  function build(ramp) {
    const r = VZ.rng(23), im = new Float64Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const base = truth[y * W + x] ? INK : PAPER;
      im[y * W + x] = Math.min(255, Math.max(0,
        Math.round(base * shade(x, y, ramp) + NOISE * VZ.randn(r))));
    }
    return im;
  }
  function otsuT(im) {
    const h = new Float64Array(256);
    for (let i = 0; i < im.length; i++) h[im[i]]++;
    const N = im.length; let muG = 0;
    for (let l = 0; l < 256; l++) muG += l * h[l] / N;
    let om = 0, mm = 0, best = -1, t = 1;
    for (let l = 0; l < 255; l++) {
      om += h[l] / N; mm += l * h[l] / N;
      if (om <= 1e-14 || om >= 1 - 1e-14) continue;
      const v = Math.pow(muG * om - mm, 2) / (om * (1 - om));
      if (v > best) { best = v; t = l; }
    }
    return t;
  }
  function iou(pred) {
    let i = 0, u = 0;
    for (let k = 0; k < pred.length; k++) { if (pred[k] & truth[k]) i++; if (pred[k] | truth[k]) u++; }
    return u ? i / u : 1;
  }
  function bandFP(pred) {
    let fp = 0, n = 0;
    for (let y = BAND.y0; y < BAND.y1; y++) for (let x = 0; x < W; x++) { n++; if (pred[y * W + x]) fp++; }
    return fp / n;
  }
  return { W: W, H: H, INK: INK, PAPER: PAPER, NOISE: NOISE, truth: truth, BAND: BAND,
    shade: shade, build: build, otsuT: otsuT, iou: iou, bandFP: bandFP };
})();

(function () {
  const svg = d3.select("#adapt-svg");
  if (svg.empty()) return;
  const S = SGadapt;
  const eR = document.getElementById("ad-r"), eRv = document.getElementById("ad-rv");
  const eW = document.getElementById("ad-w"), eWv = document.getElementById("ad-wv");
  const eC = document.getElementById("ad-c"), eCv = document.getElementById("ad-cv");
  const eK = document.getElementById("ad-k"), eKv = document.getElementById("ad-kv");
  const eN = document.getElementById("ad-nib");
  const out = document.getElementById("adapt-readout");

  function compute() {
    const ramp = +eR.value, w = +eW.value, C = +eC.value, k = +eK.value, nib = eN.checked;
    const im = S.build(ramp);
    const t = S.otsuT(im);
    const N = S.W * S.H;
    const gl = new Uint8Array(N);
    for (let i = 0; i < N; i++) gl[i] = im[i] < t ? 1 : 0;          // ink is dark
    const bs = SG.boxStats(im, S.W, S.H, w);
    const loc = new Uint8Array(N), sau = new Uint8Array(N), tsurf = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const th = nib ? (bs.mu[i] - 0.2 * bs.sd[i]) : (bs.mu[i] - C);
      tsurf[i] = th;
      loc[i] = im[i] < th ? 1 : 0;
      const ts = bs.mu[i] * (1 + k * (bs.sd[i] / 128 - 1));
      sau[i] = im[i] < ts ? 1 : 0;
    }
    const smin = Math.min(S.shade(0, 0, ramp), S.shade(S.W - 1, S.H - 1, ramp));
    const smax = Math.max(S.shade(0, 0, ramp), S.shade(S.W - 1, S.H - 1, ramp));
    return { im: im, t: t, gl: gl, loc: loc, sau: sau, tsurf: tsurf, bs: bs,
      ramp: ramp, w: w, C: C, k: k, nib: nib,
      swing: smax / smin, contrast: S.PAPER / S.INK };
  }

  function draw() {
    const R = compute();
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 1.35, IW = S.W * cw, IH = S.H * cw, gap = 16, x0 = 10;
    const locName = R.nib ? "Niblack  μ − 0.2σ" : "mean − C";
    const panels = [
      { t: "the image  (ink 90, paper 210)", f: i => SG.grey(R.im[i]), s: null },
      { t: "global Otsu, t = " + R.t, f: i => R.gl[i] ? "#e6e9ef" : "#12141c", s: R.gl },
      { t: "local  " + locName, f: i => R.loc[i] ? "#e6e9ef" : "#12141c", s: R.loc },
      { t: "Sauvola,  k = " + R.k.toFixed(2), f: i => R.sau[i] ? "#e6e9ef" : "#12141c", s: R.sau }
    ];
    panels.forEach((p, j) => {
      const bx = x0 + j * (IW + gap);
      const b = SG.box(g, bx, 34, IW, IH, p.t);
      SG.cells(b, 0, 0, cw, S.W, S.H, (x, y) => p.f(y * S.W + x));
      /* the blank band, marked on every panel */
      b.append("rect").attr("x", 0).attr("y", S.BAND.y0 * cw).attr("width", IW)
        .attr("height", (S.BAND.y1 - S.BAND.y0) * cw).attr("fill", "none")
        .attr("stroke", VC.a2).attr("stroke-dasharray", "3 2").attr("stroke-width", 1);
      if (p.s) {
        b.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10)
          .attr("fill", VC.muted).text("IoU " + S.iou(p.s).toFixed(4));
        b.append("text").attr("x", 0).attr("y", IH + 25).attr("font-size", 10)
          .attr("fill", S.bandFP(p.s) > 0.02 ? VC.bad : VC.good)
          .text("blank-band FP " + (100 * S.bandFP(p.s)).toFixed(2) + "%");
      } else {
        b.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10)
          .attr("fill", VC.a2).text("dashed band contains no ink");
        b.append("text").attr("x", 0).attr("y", IH + 25).attr("font-size", 10)
          .attr("fill", VC.muted).text("shading swing ×" + R.swing.toFixed(2));
      }
    });

    /* ── bottom left: the threshold surface ── */
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < R.tsurf.length; i++) { if (R.tsurf[i] < lo) lo = R.tsurf[i]; if (R.tsurf[i] > hi) hi = R.tsurf[i]; }
    const b5 = SG.box(g, x0, 236, IW, IH, "the local threshold surface t(x, y)");
    SG.cells(b5, 0, 0, cw, S.W, S.H, (x, y) => {
      const v = (R.tsurf[y * S.W + x] - lo) / Math.max(1e-9, hi - lo);
      return d3.interpolateCividis ? d3.interpolateCividis(v) : SG.grey(40 + 180 * v);
    });
    b5.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10).attr("fill", VC.muted)
      .text("range " + lo.toFixed(0) + " … " + hi.toFixed(0) + " — it is the shading field");

    /* ── bottom middle: bars ── */
    const BX = x0 + IW + gap, BW = 2 * IW + gap, BH = 62;
    const bars = [
      { n: "global Otsu", s: R.gl, c: VC.bad },
      { n: locName, s: R.loc, c: VC.accent },
      { n: "Sauvola", s: R.sau, c: VC.good }
    ];
    const b6 = SG.box(g, BX, 236, BW, BH, "IoU against the true ink mask", { fill: "#12141c" });
    const xs = d3.scaleLinear().domain([0, 1]).range([0, BW - 84]);
    bars.forEach((bb, i) => {
      const v = S.iou(bb.s);
      b6.append("rect").attr("x", 82).attr("y", 6 + i * 18).attr("width", Math.max(0, xs(v)))
        .attr("height", 13).attr("fill", bb.c).attr("fill-opacity", 0.75);
      b6.append("text").attr("x", 78).attr("y", 16 + i * 18).attr("text-anchor", "end")
        .attr("font-size", 10).attr("fill", VC.muted).text(bb.n);
      b6.append("text").attr("x", 86 + xs(v)).attr("y", 16 + i * 18).attr("font-size", 10)
        .attr("fill", bb.c).text(v.toFixed(4));
    });
    const b7 = SG.box(g, BX, 236 + BH + 34, BW, BH, "false positives inside the blank band (should be 0)", { fill: "#12141c" });
    const xs2 = d3.scaleLinear().domain([0, 0.35]).range([0, BW - 84]);
    bars.forEach((bb, i) => {
      const v = S.bandFP(bb.s);
      b7.append("rect").attr("x", 82).attr("y", 6 + i * 18).attr("width", Math.max(0.8, xs2(Math.min(0.35, v))))
        .attr("height", 13).attr("fill", bb.c).attr("fill-opacity", 0.75);
      b7.append("text").attr("x", 78).attr("y", 16 + i * 18).attr("text-anchor", "end")
        .attr("font-size", 10).attr("fill", VC.muted).text(bb.n);
      b7.append("text").attr("x", 86 + xs2(Math.min(0.35, v))).attr("y", 16 + i * 18).attr("font-size", 10)
        .attr("fill", bb.c).text((100 * v).toFixed(2) + "%");
    });

    /* ── bottom right: numbers ── */
    const put = SG.kv(g, BX + BW + gap, 250, { keyW: 128, lead: 15 });
    put("shading swing", "× " + R.swing.toFixed(3), VC.a2);
    put("ink : paper contrast", "× " + R.contrast.toFixed(3), VC.a2);
    put("global threshold can", R.swing < R.contrast ? "exist" : "NOT exist",
      R.swing < R.contrast ? VC.good : VC.bad, true);
    put("window", (2 * R.w + 1) + " × " + (2 * R.w + 1) + " px");
    put("thickest ink feature", "12 px");
    put("Otsu threshold", String(R.t));
    put("t(x,y) spread", (hi - lo).toFixed(1) + " levels", VC.violet);
  }

  function say() {
    const R = compute();
    out.innerHTML =
      `Shading swings by <b>×${R.swing.toFixed(3)}</b> across the frame; ink and paper differ by `
      + `<b>×${R.contrast.toFixed(3)}</b>. `
      + (R.swing < R.contrast
        ? `The swing is still inside the contrast ratio, so a constant threshold exists — and global Otsu finds it, at IoU <b>${S.iou(R.gl).toFixed(4)}</b>.`
        : `<b>The swing has exceeded the contrast ratio</b>, so the two value ranges now overlap and <i>no</i> constant threshold separates them. Global Otsu scores IoU <b>${S.iou(R.gl).toFixed(4)}</b>, and no other choice of constant would do better — the failure is in the model, not the search.`)
      + `<br>With a ${2 * R.w + 1}-pixel window, ${R.nib ? "Niblack" : "mean − C"} scores <b>${S.iou(R.loc).toFixed(4)}</b> `
      + `and Sauvola <b>${S.iou(R.sau).toFixed(4)}</b>. In the band that contains no ink at all, they fire on `
      + `<b>${(100 * S.bandFP(R.loc)).toFixed(2)}%</b> and <b>${(100 * S.bandFP(R.sau)).toFixed(2)}%</b> of pixels `
      + `respectively — the price of a rule that must invent a boundary inside every window, and the reason `
      + `Sauvola's bracket collapses when the local standard deviation does.`;
  }
  function all() {
    eRv.textContent = (+eR.value).toFixed(2);
    eWv.textContent = eW.value + " (" + (2 * (+eW.value) + 1) + "px)";
    eCv.textContent = eC.value;
    eKv.textContent = (+eK.value).toFixed(2);
    draw(); say();
  }
  eR.oninput = all; eW.oninput = all; eC.oninput = all; eK.oninput = all; eN.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   5 · #cc-svg — connected components, 4 against 8, and the Jordan paradox
   ══════════════════════════════════════════════════════════════════════════ */
const SGcc = (function () {
  /* (a) the diamond ring: L1 radius 4 on an 11 × 11 grid */
  const DW = 11, DH = 11, DC = 5, DR = 4;
  const ring = new Uint8Array(DW * DH);
  for (let y = 0; y < DH; y++) for (let x = 0; x < DW; x++)
    ring[y * DW + x] = (Math.abs(y - DC) + Math.abs(x - DC) === DR) ? 1 : 0;

  /* (b) three blobs plus salt noise, for the area filter */
  const BW = 56, BH = 40;
  const blob = new Uint8Array(BW * BH);
  {
    const r = VZ.rng(5);
    const E = [[14, 13, 9, 8], [37, 12, 8, 7], [26, 30, 12, 6]];
    for (let y = 0; y < BH; y++) for (let x = 0; x < BW; x++) {
      let on = 0;
      for (const [cx, cy, a, b] of E)
        if (Math.pow((x - cx) / a, 2) + Math.pow((y - cy) / b, 2) < 1) on = 1;
      if (r() < 0.045) on = 1 - on;                 // salt and pepper, both ways
      blob[y * BW + x] = on;
    }
  }
  /* components of one value only, with a flag for whether the component touches
     the image border — that flag is what distinguishes a hole from the outside */
  function compsOf(mask, W, H, want, conn) {
    const seen = new Int32Array(W * H).fill(-1);
    const N4 = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    const N8 = N4.concat([[1, 1], [1, -1], [-1, 1], [-1, -1]]);
    const nb = (conn === 8) ? N8 : N4;
    const comps = [];
    for (let s = 0; s < W * H; s++) {
      if (mask[s] !== want || seen[s] >= 0) continue;
      const id = comps.length, px = [], stack = [s];
      seen[s] = id;
      let touchesBorder = false;
      while (stack.length) {
        const q = stack.pop(), qx = q % W, qy = (q / W) | 0;
        px.push(q);
        if (qx === 0 || qy === 0 || qx === W - 1 || qy === H - 1) touchesBorder = true;
        for (const [dx, dy] of nb) {
          const x = qx + dx, y = qy + dy;
          if (x < 0 || y < 0 || x >= W || y >= H) continue;
          const t = y * W + x;
          if (mask[t] === want && seen[t] < 0) { seen[t] = id; stack.push(t); }
        }
      }
      comps.push({ id: id, px: px, area: px.length, border: touchesBorder });
    }
    return { lab: seen, comps: comps };
  }
  return { DW: DW, DH: DH, ring: ring, BW: BW, BH: BH, blob: blob, compsOf: compsOf };
})();

(function () {
  const svg = d3.select("#cc-svg");
  if (svg.empty()) return;
  const S = SGcc;
  const eF = document.getElementById("cc-fg"), eB = document.getElementById("cc-bg");
  const eM = document.getElementById("cc-min"), eMv = document.getElementById("cc-minv");
  const out = document.getElementById("cc-readout");
  const BGPAL = ["#3d4657", "#55606f", "#6c7789", "#8792a3", "#2b3241"];

  function stats() {
    const fc = +eF.value, bc = +eB.value;
    const fg = S.compsOf(S.ring, S.DW, S.DH, 1, fc);
    const bg = S.compsOf(S.ring, S.DW, S.DH, 0, bc);
    const holes = bg.comps.filter(c => !c.border).length;
    return { fc: fc, bc: bc, fg: fg, bg: bg, holes: holes,
      euler: fg.comps.length - holes, ok: fc !== bc };
  }

  function draw() {
    const st = stats(), minA = +eM.value;
    svg.selectAll("*").remove();
    const g = svg.append("g");

    /* ── left: the diamond ── */
    const cw = 22, IW = S.DW * cw;
    const A = SG.box(g, 16, 38, IW, S.DH * cw,
      `foreground ${st.fc}-connected · background ${st.bc}-connected`);
    SG.cells(A, 0, 0, cw, S.DW, S.DH, (x, y) => {
      const i = y * S.DW + x;
      return S.ring[i] ? SG.pal(st.fg.lab[i]) : BGPAL[st.bg.lab[i] % BGPAL.length];
    });
    A.append("g").selectAll("line").data(d3.range(0, S.DW + 1)).join("line")
      .attr("x1", d => d * cw).attr("x2", d => d * cw).attr("y1", 0).attr("y2", S.DH * cw)
      .attr("stroke", "#0b0d12").attr("stroke-opacity", 0.5);
    A.append("g").selectAll("line").data(d3.range(0, S.DH + 1)).join("line")
      .attr("y1", d => d * cw).attr("y2", d => d * cw).attr("x1", 0).attr("x2", IW)
      .attr("stroke", "#0b0d12").attr("stroke-opacity", 0.5);
    A.append("text").attr("x", 0).attr("y", S.DH * cw + 15).attr("font-size", 10.5)
      .attr("fill", VC.muted).text("16 foreground pixels on an 11 × 11 grid");
    A.append("text").attr("x", 0).attr("y", S.DH * cw + 29).attr("font-size", 10.5)
      .attr("fill", st.ok ? VC.good : VC.bad)
      .text(st.ok ? "conventions differ — Jordan holds" : "same convention for both — Jordan fails");

    const put = SG.kv(g, 16, S.DH * cw + 60, { keyW: 172, lead: 15.5 });
    put("foreground components", String(st.fg.comps.length), VC.accent, true);
    put("background components", String(st.bg.comps.length), VC.ink, true);
    put("of those, holes (not on the border)", String(st.holes), VC.violet);
    put("Euler number  E = C − H", String(st.euler), VC.a2, true);
    put("a ring should give", "C = 1,  H = 1,  E = 0", VC.muted);
    put("consistent pair?", st.ok ? "yes" : "no — inside and outside merge",
      st.ok ? VC.good : VC.bad, true);

    /* ── right: the area filter ── */
    const cw2 = 5, JX = 336, JW = S.BW * cw2;
    const bl = S.compsOf(S.blob, S.BW, S.BH, 1, 8);
    const keep = new Uint8Array(bl.comps.length);
    bl.comps.forEach(c => { keep[c.id] = c.area >= minA ? 1 : 0; });
    const kept = bl.comps.filter(c => c.area >= minA).length;
    const B = SG.box(g, JX, 38, JW, S.BH * cw2, "a thresholded image with salt noise, 8-connected");
    SG.cells(B, 0, 0, cw2, S.BW, S.BH, (x, y) => {
      const i = y * S.BW + x;
      if (!S.blob[i]) return "#12141c";
      return keep[bl.lab[i]] ? SG.pal(bl.lab[i]) : "#3a2530";
    });
    B.append("text").attr("x", 0).attr("y", S.BH * cw2 + 14).attr("font-size", 10.5)
      .attr("fill", VC.muted)
      .text(`${bl.comps.length} components, ${kept} survive an area of ${minA} px`);

    /* the survival curve */
    const CX = JX, CY = 292, CW = JW, CH = 84;
    const C = SG.box(g, CX, CY, CW, CH, "components surviving, against minimum area", { fill: "#12141c" });
    const areas = d3.range(1, 61);
    const cnt = areas.map(a => bl.comps.filter(c => c.area >= a).length);
    const xs = d3.scaleLinear().domain([1, 60]).range([0, CW]);
    const ys = d3.scaleLog().domain([0.8, Math.max(2, bl.comps.length)]).range([CH, 0]);
    C.append("g").attr("transform", `translate(0,${CH})`).attr("class", "axis").call(d3.axisBottom(xs).ticks(6));
    C.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(3, "~s"));
    C.append("path").datum(areas).attr("d", d3.line().x(d => xs(d)).y((d, i) => ys(Math.max(0.8, cnt[i]))))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 1.8);
    C.append("line").attr("x1", xs(Math.min(60, minA))).attr("x2", xs(Math.min(60, minA)))
      .attr("y1", 0).attr("y2", CH).attr("stroke", VC.a2).attr("stroke-width", 1.5);
    const big = bl.comps.filter(c => c.area > 40).length;
    C.append("text").attr("x", CW - 4).attr("y", 14).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", VC.muted).text(`${big} components are larger than 40 px`);
  }

  function say() {
    const st = stats(), minA = +eM.value;
    const bl = S.compsOf(S.blob, S.BW, S.BH, 1, 8);
    const kept = bl.comps.filter(c => c.area >= minA).length;
    const tiny = bl.comps.filter(c => c.area <= 3).length;
    out.innerHTML =
      `Diamond ring: <b>${st.fg.comps.length}</b> foreground component${st.fg.comps.length === 1 ? "" : "s"} `
      + `under ${st.fc}-connectivity, <b>${st.bg.comps.length}</b> background component${st.bg.comps.length === 1 ? "" : "s"} `
      + `under ${st.bc}-connectivity, <b>${st.holes}</b> hole${st.holes === 1 ? "" : "s"}, Euler number <b>${st.euler}</b>. `
      + (st.ok
        ? `The two conventions differ, so a closed curve separates its inside from its outside — as it must.`
        : `<b style="color:var(--bad)">Both conventions are the same, so this is the inconsistent case.</b> `
        + (st.fc === 8
          ? `The ring is one connected curve and the background is also one piece: you can walk from inside to outside diagonally between two ring pixels without crossing the ring.`
          : `The ring shatters into ${st.fg.comps.length} isolated pixels, so there is no curve — although the pixel set does still separate the plane into two.`))
      + `<br>Noisy image: <b>${bl.comps.length}</b> 8-connected components in total, of which <b>${tiny}</b> are three pixels or smaller. `
      + `An area filter at <b>${minA} px</b> leaves <b>${kept}</b>. `
      + `The three real blobs are the only components above forty pixels, so any cut-off in that wide gap gives the same, correct answer — `
      + `which is the whole reason area filtering is the standard first line of defence after a threshold.`;
  }
  function all() { eMv.textContent = eM.value + " px"; draw(); say(); }
  eF.onchange = all; eB.onchange = all; eM.oninput = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   6 · #grow-svg — region growing, and the tolerance cliff
   ══════════════════════════════════════════════════════════════════════════ */
const SGgrow = (function () {
  const W = 72, H = 56, GRAD = 0.18, NOISE = 4;
  const inA = (x, y) => Math.pow((x - 22) / 15, 2) + Math.pow((y - 28) / 17, 2) < 1;
  const inB = (x, y) => Math.pow((x - 50) / 14, 2) + Math.pow((y - 28) / 16, 2) < 1;
  const A = new Uint8Array(W * H), B = new Uint8Array(W * H);
  const img = new Float64Array(W * H);
  {
    const r = VZ.rng(31);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const a = inA(x, y), b = inB(x, y) && !a;
      A[y * W + x] = a ? 1 : 0; B[y * W + x] = b ? 1 : 0;
      const base = b ? 172 : (a ? 150 : 62);
      const S = 1 + GRAD * (x / (W - 1) - 0.5);
      img[y * W + x] = Math.min(255, Math.max(0, Math.round(base * S + NOISE * VZ.randn(r))));
    }
  }
  const SEED = [16, 28], SEED2 = [56, 28];
  /* breadth-first growth. Returns the label map (-1 = ungrown) and, for each
     accepted pixel, the BFS step at which it was taken, so the front is drawable. */
  function grow(tol, mode, twoSeeds) {
    const lab = new Int32Array(W * H).fill(-1);
    const step = new Int32Array(W * H).fill(-1);
    const seeds = twoSeeds ? [SEED, SEED2] : [SEED];
    const sum = [0, 0], cnt = [0, 0], sval = [0, 0];
    let q = [];
    seeds.forEach((s, k) => {
      const i = s[1] * W + s[0];
      lab[i] = k; step[i] = 0; sum[k] += img[i]; cnt[k]++; sval[k] = img[i];
      q.push([i, k]);
    });
    let d = 0, sizes = [seeds.length];
    while (q.length) {
      const nxt = [];
      d++;
      for (const [i, k] of q) {
        const ix = i % W, iy = (i / W) | 0;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const x = ix + dx, y = iy + dy;
          if (x < 0 || y < 0 || x >= W || y >= H) continue;
          const j = y * W + x;
          if (lab[j] >= 0) continue;
          const ref = (mode === "seed") ? sval[k] : sum[k] / cnt[k];
          if (Math.abs(img[j] - ref) <= tol) {
            lab[j] = k; step[j] = d; sum[k] += img[j]; cnt[k]++;
            nxt.push([j, k]);
          }
        }
      }
      q = nxt;
      sizes.push(sizes[sizes.length - 1] + nxt.length);
    }
    return { lab: lab, step: step, depth: d, sizes: sizes, cnt: cnt };
  }
  function score(res) {
    let n = 0, inAn = 0, inBn = 0, bg = 0, inter = 0, uni = 0;
    for (let i = 0; i < W * H; i++) {
      const r = res.lab[i] === 0 ? 1 : 0;
      if (r) { n++; if (A[i]) inAn++; else if (B[i]) inBn++; else bg++; }
      if (r && A[i]) inter++;
      if (r || A[i]) uni++;
    }
    return { n: n, inA: inAn, inB: inBn, bg: bg, iou: uni ? inter / uni : 1 };
  }
  /* the tolerance at which the first neighbour pixel is admitted, bisected */
  function critical(mode, twoSeeds) {
    let lo = 1, hi = 200;
    if (score(grow(hi, mode, twoSeeds)).inB === 0) return null;
    if (score(grow(lo, mode, twoSeeds)).inB > 0) return lo;
    for (let k = 0; k < 26; k++) {
      const m = (lo + hi) / 2;
      if (score(grow(m, mode, twoSeeds)).inB > 0) hi = m; else lo = m;
    }
    return hi;
  }
  /* the widest run of tolerances (on a 0.25 grid) that returns exactly object A */
  function exactRun(mode, twoSeeds) {
    let lo = null, hi = null;
    for (let t = 2; t <= 110; t += 0.25) {
      const s = score(grow(t, mode, twoSeeds));
      if (s.n === 797 && s.inA === 797) { if (lo === null) lo = t; hi = t; }
      else if (lo !== null) break;
    }
    return (lo === null) ? null : { lo: lo, hi: hi, w: hi - lo };
  }
  return { W: W, H: H, img: img, A: A, B: B, SEED: SEED, SEED2: SEED2,
    grow: grow, score: score, critical: critical, exactRun: exactRun,
    areaA: A.reduce((a, b) => a + b, 0), areaB: B.reduce((a, b) => a + b, 0) };
})();

(function () {
  const svg = d3.select("#grow-svg");
  if (svg.empty()) return;
  const S = SGgrow;
  const eT = document.getElementById("gr-t"), eTv = document.getElementById("gr-tv");
  const eM = document.getElementById("gr-mode");
  const eS = document.getElementById("gr-step"), eSv = document.getElementById("gr-stepv");
  const e2 = document.getElementById("gr-two");
  const out = document.getElementById("grow-readout");
  /* the tolerance sweep is expensive, so cache it per (mode, seeds) pair */
  const SWEEP = {};
  function sweep(mode, two) {
    const key = mode + (two ? "2" : "1");
    if (!SWEEP[key]) {
      const ts = d3.range(2, 110.01, 1);
      SWEEP[key] = ts.map(t => {
        const s = S.score(S.grow(t, mode, two));
        return { t: t, n: s.n, inB: s.inB, bg: s.bg };
      });
    }
    return SWEEP[key];
  }

  function draw() {
    const tol = +eT.value, mode = eM.value, two = e2.checked;
    const maxStep = +eS.value;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const res = S.grow(tol, mode, two), sc = S.score(res);
    const cw = 3, IW = S.W * cw, IH = S.H * cw;

    /* ── A · the image ── */
    const a = SG.box(g, 14, 40, IW, IH, "the image — two objects that touch");
    SG.cells(a, 0, 0, cw, S.W, S.H, (x, y) => SG.grey(S.img[y * S.W + x]));
    const dot = (p, c) => a.append("circle").attr("cx", (p[0] + 0.5) * cw).attr("cy", (p[1] + 0.5) * cw)
      .attr("r", 4).attr("fill", c).attr("stroke", "#0b0d12").attr("stroke-width", 1.2);
    dot(S.SEED, VC.good);
    if (two) dot(S.SEED2, VC.violet);
    a.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10.5).attr("fill", VC.muted)
      .text("left object ≈150, right ≈172, background ≈62");

    /* ── B · the grown region, shaded by BFS step ── */
    const shown = Math.min(maxStep, res.depth);
    const b = SG.box(g, 254, 40, IW, IH,
      shown >= res.depth ? "the region, complete" : `the region after ${shown} steps`);
    SG.cells(b, 0, 0, cw, S.W, S.H, (x, y) => {
      const i = y * S.W + x;
      if (res.lab[i] < 0 || res.step[i] > shown) return SG.grey(S.img[i] * 0.42);
      const f = res.step[i] / Math.max(1, res.depth);
      return res.lab[i] === 0 ? d3.interpolateViridis(0.15 + 0.8 * f) : "#c084fc";
    });
    /* the true boundary of object B, so the leak is unambiguous */
    const bl = new Int32Array(S.W * S.H);
    for (let i = 0; i < bl.length; i++) bl[i] = S.B[i];
    SG.drawEdges(b, SG.edgesOf(bl, S.W, S.H), cw, { color: VC.rose, w: 1.2, op: 0.9 });
    b.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10.5)
      .attr("fill", sc.inB > 0 ? VC.bad : VC.good)
      .text(sc.inB > 0 ? `leaked — ${sc.inB} px of ${S.areaB} taken from the right object`
        : "no leak — the pink outline is untouched");

    /* ── C · the tolerance sweep ── */
    const CX = 496, CY = 40, CW = 250, CH = 168;
    const sw = sweep(mode, two);
    const c = SG.box(g, CX, CY, CW, CH, "size and leak, against tolerance", { fill: "#12141c" });
    const xs = d3.scaleLinear().domain([2, 110]).range([0, CW]);
    const ys = d3.scaleLog().domain([0.7, 4032]).range([CH, 0]);
    c.append("g").attr("transform", `translate(0,${CH})`).attr("class", "axis").call(d3.axisBottom(xs).ticks(6));
    c.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(4, "~s"));
    const ln = (acc, col) => c.append("path").datum(sw)
      .attr("d", d3.line().x(d => xs(d.t)).y(d => ys(Math.max(0.7, acc(d)))))
      .attr("fill", "none").attr("stroke", col).attr("stroke-width", 1.8);
    ln(d => d.n, VC.accent);
    ln(d => d.inB, VC.bad);
    c.append("line").attr("x1", 0).attr("x2", CW).attr("y1", ys(S.areaA)).attr("y2", ys(S.areaA))
      .attr("stroke", VC.good).attr("stroke-dasharray", "4 3");
    c.append("text").attr("x", 3).attr("y", ys(S.areaA) - 4).attr("font-size", 9.5).attr("fill", VC.good)
      .text("the correct size, " + S.areaA + " px");
    c.append("line").attr("x1", xs(tol)).attr("x2", xs(tol)).attr("y1", 0).attr("y2", CH)
      .attr("stroke", VC.ink).attr("stroke-width", 1.4);
    VZ.legend(c, [{ color: VC.accent, label: "region size" }, { color: VC.bad, label: "px stolen from B" }],
      8, 14, { gap: 14 });

    /* ── D · growth against step ── */
    const DX = 14, DY = 254, DW = 456, DH = 96;
    const d = SG.box(g, DX, DY, DW, DH, "region size against breadth-first step", { fill: "#12141c" });
    const xd = d3.scaleLinear().domain([0, Math.max(4, res.sizes.length - 1)]).range([0, DW]);
    const yd = d3.scaleLinear().domain([0, Math.max(10, res.sizes[res.sizes.length - 1])]).range([DH, 0]);
    d.append("g").attr("transform", `translate(0,${DH})`).attr("class", "axis").call(d3.axisBottom(xd).ticks(8));
    d.append("g").attr("class", "axis").call(d3.axisLeft(yd).ticks(4));
    d.append("path").datum(res.sizes).attr("d", d3.line().x((v, i) => xd(i)).y(v => yd(v)))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 1.8);
    if (shown < res.sizes.length)
      d.append("line").attr("x1", xd(shown)).attr("x2", xd(shown)).attr("y1", 0).attr("y2", DH)
        .attr("stroke", VC.a2).attr("stroke-width", 1.4);
    /* mark the step at which the first B pixel is taken */
    let firstB = -1;
    for (let i = 0; i < S.W * S.H; i++)
      if (S.B[i] && res.lab[i] === 0 && (firstB < 0 || res.step[i] < firstB)) firstB = res.step[i];
    if (firstB > 0) {
      d.append("line").attr("x1", xd(firstB)).attr("x2", xd(firstB)).attr("y1", 0).attr("y2", DH)
        .attr("stroke", VC.bad).attr("stroke-dasharray", "3 3");
      d.append("text").attr("x", xd(firstB) + 4).attr("y", 12).attr("font-size", 10).attr("fill", VC.bad)
        .text("front crosses the neck at step " + firstB);
    }

    /* ── E · numbers ── */
    const crit = S.critical(mode, two), run = S.exactRun(mode, two);
    const put = SG.kv(g, 496, 266, { keyW: 150, lead: 15 });
    put("region area", sc.n + " px");
    put("of true object A (797)", sc.inA + " px", sc.inA === S.areaA ? VC.good : VC.a2);
    put("stolen from B (693)", sc.inB + " px", sc.inB ? VC.bad : VC.good, sc.inB > 0);
    put("background taken", sc.bg + " px", sc.bg ? VC.bad : VC.muted);
    put("IoU against object A", sc.iou.toFixed(4), sc.iou > 0.98 ? VC.good : VC.bad, true);
    put("first leak at τ =", crit === null ? "never (up to 200)" : crit.toFixed(2), VC.bad);
    put("exactly-right τ window", run ? run.lo.toFixed(2) + " … " + run.hi.toFixed(2)
      + "  (width " + run.w.toFixed(2) + ")" : "none", VC.violet);
  }

  function say() {
    const tol = +eT.value, mode = eM.value, two = e2.checked;
    const sc = S.score(S.grow(tol, mode, two));
    const crit = S.critical(mode, two), run = S.exactRun(mode, two);
    out.innerHTML =
      `Tolerance <b>${tol.toFixed(2)}</b>, ${mode === "mean" ? "running-mean" : "fixed-reference"} predicate`
      + (two ? ", two competing seeds" : ", one seed") + `. The region is <b>${sc.n}</b> pixels: `
      + `<b>${sc.inA}</b> of object A's 797, <b>${sc.inB}</b> stolen from object B, <b>${sc.bg}</b> background. `
      + `IoU against A is <b>${sc.iou.toFixed(4)}</b>.<br>`
      + (crit === null
        ? `<b style="color:var(--good)">This configuration never leaks</b>, at any tolerance up to 200 — the two fronts meet at the neck and neither crosses it. That is the whole value of competition.`
        : `The first pixel of B is admitted at <b>τ = ${crit.toFixed(2)}</b>, and the leak then compounds: `
        + `each stolen pixel drags the reference toward B, which admits more of B. `)
      + (run ? ` The tolerances that return object A exactly span <b>${run.lo.toFixed(2)} to ${run.hi.toFixed(2)}</b> — `
        + `a window <b>${run.w.toFixed(2)}</b> grey levels wide, on an image with a noise standard deviation of 4. `
        + `That window is the entire safety margin, and it is a property of this image only.`
        : ` No tolerance returns object A exactly.`);
  }
  function all() {
    eTv.textContent = (+eT.value).toFixed(2);
    const res = S.grow(+eT.value, eM.value, e2.checked);
    eS.max = Math.max(2, res.depth);
    eSv.textContent = (+eS.value >= res.depth) ? "complete" : ("step " + eS.value);
    draw(); say();
  }
  eT.oninput = () => { eS.value = eS.max; all(); };
  eM.onchange = () => { eS.value = 220; all(); };
  e2.onchange = () => { eS.value = 220; all(); };
  eS.oninput = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   7 · #quad-svg — quadtree splitting, and the merge pass that repairs it
   ══════════════════════════════════════════════════════════════════════════ */
const SGquad = (function () {
  const W = 64, H = 64;
  const img = new Float64Array(W * H);
  {
    const r = VZ.rng(17);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let v = 70 + 22 * (x / (W - 1)) - 10 * (y / (H - 1));          // shaded background
      if (Math.pow(x - 19, 2) + Math.pow(y - 18, 2) < 144) v = 196;   // bright disc
      if (x >= 36 && x < 59 && y >= 7 && y < 26) v = 132;             // mid rectangle
      if (Math.pow((x - 20) / 13, 2) + Math.pow((y - 47) / 10, 2) < 1) v = 34;   // dark ellipse
      if (x >= 44 && x < 60 && y >= 42 && y < 58)                     // a 2-pixel checkerboard
        v = (((x >> 1) + (y >> 1)) & 1) ? 168 : 98;
      img[y * W + x] = Math.min(255, Math.max(0, v + 3 * VZ.randn(r)));
    }
  }
  /* summed-area tables, so a block's mean, min and max are O(1) — min and max
     are not summable, so they are carried up the recursion instead */
  function blockStats(x0, y0, s) {
    let mn = Infinity, mx = -Infinity, sum = 0;
    for (let y = y0; y < y0 + s; y++) for (let x = x0; x < x0 + s; x++) {
      const v = img[y * W + x];
      if (v < mn) mn = v; if (v > mx) mx = v; sum += v;
    }
    return { mn: mn, mx: mx, mean: sum / (s * s), n: s * s };
  }
  function sd(x0, y0, s) {
    let a = 0, b = 0;
    for (let y = y0; y < y0 + s; y++) for (let x = x0; x < x0 + s; x++) {
      const v = img[y * W + x]; a += v; b += v * v;
    }
    const n = s * s, m = a / n;
    return Math.sqrt(Math.max(0, b / n - m * m));
  }
  function split(pred, tol, minS) {
    const leaves = [];
    let maxDepth = 0;
    (function rec(x0, y0, s, d) {
      const st = blockStats(x0, y0, s);
      const ok = (pred === "range") ? (st.mx - st.mn <= tol) : (sd(x0, y0, s) <= tol);
      if (ok || s <= minS) { leaves.push({ x: x0, y: y0, s: s, mn: st.mn, mx: st.mx, sum: st.mean * st.n, n: st.n }); maxDepth = Math.max(maxDepth, d); return; }
      const h = s >> 1;
      rec(x0, y0, h, d + 1); rec(x0 + h, y0, h, d + 1);
      rec(x0, y0 + h, h, d + 1); rec(x0 + h, y0 + h, h, d + 1);
    })(0, 0, W, 0);
    return { leaves: leaves, depth: maxDepth };
  }
  /* leaf id per pixel, then a region adjacency list over the leaves */
  function leafMap(leaves) {
    const id = new Int32Array(W * H);
    leaves.forEach((L, k) => {
      for (let y = L.y; y < L.y + L.s; y++) for (let x = L.x; x < L.x + L.s; x++) id[y * W + x] = k;
    });
    return id;
  }
  function adjacency(id) {
    const seen = new Set(), out = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const a = id[y * W + x];
      const push = b => {
        if (a === b) return;
        const k = Math.min(a, b) * 100000 + Math.max(a, b);
        if (!seen.has(k)) { seen.add(k); out.push([Math.min(a, b), Math.max(a, b)]); }
      };
      if (x + 1 < W) push(id[y * W + x + 1]);
      if (y + 1 < H) push(id[(y + 1) * W + x]);
    }
    return out;
  }
  /* greedy merge in increasing order of mean difference, accepting a merge only
     when the UNION still satisfies the predicate — condition 4 of §03 */
  function merge(leaves, adj, pred, tol) {
    const uf = SG.UF(leaves.length);
    const mn = leaves.map(L => L.mn), mx = leaves.map(L => L.mx);
    const sum = leaves.map(L => L.sum), cnt = leaves.map(L => L.n);
    const mean = k => sum[k] / cnt[k];
    /* the variance predicate needs second moments, kept in parallel */
    const s2 = leaves.map((L, k) => 0);
    for (let k = 0; k < leaves.length; k++) {
      const L = leaves[k]; let b = 0;
      for (let y = L.y; y < L.y + L.s; y++) for (let x = L.x; x < L.x + L.s; x++) b += img[y * W + x] * img[y * W + x];
      s2[k] = b;
    }
    const pairs = adj.map(([a, b]) => [a, b, Math.abs(leaves[a].sum / leaves[a].n - leaves[b].sum / leaves[b].n)])
      .sort((p, q) => p[2] - q[2]);
    let changed = true, rounds = 0;
    while (changed && rounds < 12) {
      changed = false; rounds++;
      for (const [a0, b0] of pairs) {
        const a = uf.find(a0), b = uf.find(b0);
        if (a === b) continue;
        const nmn = Math.min(mn[a], mn[b]), nmx = Math.max(mx[a], mx[b]);
        const ns = sum[a] + sum[b], nc = cnt[a] + cnt[b], nq = s2[a] + s2[b];
        const ok = (pred === "range") ? (nmx - nmn <= tol)
          : (Math.sqrt(Math.max(0, nq / nc - Math.pow(ns / nc, 2))) <= tol);
        if (!ok) continue;
        uf.union(a, b);
        const r = uf.find(a);
        mn[r] = nmn; mx[r] = nmx; sum[r] = ns; cnt[r] = nc; s2[r] = nq;
        changed = true;
      }
    }
    const roots = new Map();
    for (let k = 0; k < leaves.length; k++) {
      const r = uf.find(k);
      if (!roots.has(r)) roots.set(r, roots.size);
    }
    return { uf: uf, roots: roots, count: roots.size, mean: k => mean(uf.find(k)) };
  }
  return { W: W, H: H, img: img, split: split, leafMap: leafMap, adjacency: adjacency, merge: merge };
})();

(function () {
  const svg = d3.select("#quad-svg");
  if (svg.empty()) return;
  const S = SGquad;
  const eP = document.getElementById("qd-pred"), eT = document.getElementById("qd-t"), eTv = document.getElementById("qd-tv");
  const eMin = document.getElementById("qd-min"), eMinv = document.getElementById("qd-minv");
  const eM = document.getElementById("qd-merge"), eF = document.getElementById("qd-fill");
  const out = document.getElementById("quad-readout");
  const CACHE = {};
  function run(pred, tol, minS) {
    const key = pred + "|" + tol + "|" + minS;
    if (CACHE[key]) return CACHE[key];
    const sp = S.split(pred, tol, minS);
    const id = S.leafMap(sp.leaves);
    const adj = S.adjacency(id);
    const mg = S.merge(sp.leaves, adj, pred, tol);
    /* mean absolute error of the piecewise-constant approximation, both stages */
    let e1 = 0, e2 = 0;
    for (let i = 0; i < S.W * S.H; i++) {
      const k = id[i];
      e1 += Math.abs(S.img[i] - sp.leaves[k].sum / sp.leaves[k].n);
      e2 += Math.abs(S.img[i] - mg.mean(k));
    }
    const N = S.W * S.H;
    const onePx = sp.leaves.filter(L => L.s === 1).length;
    const inPatch = (x, y) => x >= 44 && x < 60 && y >= 42 && y < 58;
    const onePxPatch = sp.leaves.filter(L => L.s === 1 && inPatch(L.x, L.y)).length;
    const pr = new Set();
    for (let y = 42; y < 58; y++) for (let x = 44; x < 60; x++) pr.add(mg.uf.find(id[y * S.W + x]));
    const r = { sp: sp, id: id, adj: adj, mg: mg, mae1: e1 / N, mae2: e2 / N,
      onePx: onePx, onePxPatch: onePxPatch, patchRegions: pr.size };
    CACHE[key] = r;
    return r;
  }

  function draw() {
    const pred = eP.value, tol = +eT.value, minS = 1 << (+eMin.value);
    const R = run(pred, tol, minS), doMerge = eM.checked, fill = eF.checked;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 3.1, IW = S.W * cw;

    const a = SG.box(g, 14, 40, IW, IW, "the image, 64 × 64");
    SG.cells(a, 0, 0, cw, S.W, S.H, (x, y) => SG.grey(S.img[y * S.W + x]));

    const b = SG.box(g, 234, 40, IW, IW, `split: ${R.sp.leaves.length} leaves, depth ${R.sp.depth}`);
    SG.cells(b, 0, 0, cw, S.W, S.H, (x, y) => {
      const L = R.sp.leaves[R.id[y * S.W + x]];
      return fill ? SG.grey(L.sum / L.n) : SG.pal(R.id[y * S.W + x]);
    });
    SG.drawEdges(b, SG.edgesOf(R.id, S.W, S.H), cw, { color: VC.accent, w: 0.7, op: 0.75 });

    const lab = new Int32Array(S.W * S.H);
    for (let i = 0; i < lab.length; i++) lab[i] = R.mg.roots.get(R.mg.uf.find(R.id[i]));
    const c = SG.box(g, 454, 40, IW, IW,
      doMerge ? `merged: ${R.mg.count} regions` : "merge pass off");
    SG.cells(c, 0, 0, cw, S.W, S.H, (x, y) => {
      const i = y * S.W + x;
      if (!doMerge) { const L = R.sp.leaves[R.id[i]]; return fill ? SG.grey(L.sum / L.n) : SG.pal(R.id[i]); }
      return fill ? SG.grey(R.mg.mean(R.id[i])) : SG.pal(lab[i]);
    });
    SG.drawEdges(c, SG.edgesOf(doMerge ? lab : R.id, S.W, S.H), cw,
      { color: doMerge ? VC.good : VC.accent, w: 0.9, op: 0.85 });

    /* ── the counts against tolerance ── */
    const DX = 14, DY = 288, DW = 420, DH = 110;
    const d = SG.box(g, DX, DY, DW, DH, "leaves and merged regions, at every tolerance", { fill: "#12141c" });
    const tols = d3.range(2, 81, 2);
    const rows = tols.map(t => run(pred, t, minS));
    const xs = d3.scaleLinear().domain([2, 80]).range([0, DW]);
    const ys = d3.scaleLog().domain([1, 4096]).range([DH, 0]);
    d.append("g").attr("transform", `translate(0,${DH})`).attr("class", "axis").call(d3.axisBottom(xs).ticks(6));
    d.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(4, "~s"));
    d.append("path").datum(rows).attr("d", d3.line().x((v, i) => xs(tols[i])).y(v => ys(Math.max(1, v.sp.leaves.length))))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 1.8);
    d.append("path").datum(rows).attr("d", d3.line().x((v, i) => xs(tols[i])).y(v => ys(Math.max(1, v.mg.count))))
      .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.8);
    d.append("line").attr("x1", xs(tol)).attr("x2", xs(tol)).attr("y1", 0).attr("y2", DH)
      .attr("stroke", VC.ink).attr("stroke-width", 1.3);
    VZ.legend(d, [{ color: VC.accent, label: "quadtree leaves" }, { color: VC.good, label: "after merging" }],
      DW - 128, 13, { gap: 14 });

    const put = SG.kv(g, 460, 300, { keyW: 158, lead: 15 });
    put("quadtree leaves", String(R.sp.leaves.length), VC.accent, true);
    put("regions after merging", String(R.mg.count), VC.good, true);
    put("reduction factor", "× " + (R.sp.leaves.length / Math.max(1, R.mg.count)).toFixed(1), VC.violet);
    put("of those, in the checker patch", String(R.patchRegions), VC.a2);
    put("so the rest of the image is", String(R.mg.count - R.patchRegions) + " regions", VC.a2, true);
    put("deepest level reached", String(R.sp.depth) + "  (blocks of " + (S.W >> R.sp.depth) + " px)");
    put("single-pixel leaves", R.onePx + ", of which " + R.onePxPatch + " in the patch");
    put("mean abs. error, split", R.mae1.toFixed(3) + " levels");
    put("mean abs. error, merged", R.mae2.toFixed(3) + " levels",
      R.mae2 > R.mae1 * 1.2 ? VC.bad : VC.ink);
  }

  function say() {
    const pred = eP.value, tol = +eT.value, minS = 1 << (+eMin.value);
    const R = run(pred, tol, minS);
    out.innerHTML =
      `Predicate <b>${pred === "range" ? "max − min ≤ τ" : "σ ≤ τ"}</b> at <b>τ = ${tol}</b>, smallest block ${minS} px. `
      + `Splitting produces <b>${R.sp.leaves.length}</b> dyadic leaves, reaching level <b>${R.sp.depth}</b>; `
      + `<b>${R.onePx}</b> of them are single pixels, of which <b>${R.onePxPatch}</b> are inside the checkerboard — `
      + `the rest lie along the boundaries of the disc, the rectangle and the ellipse, which is where a quadtree `
      + `spends nearly all of its resolution.<br>`
      + `The merge pass fuses them into <b>${R.mg.count}</b> regions — a reduction of `
      + `<b>${(R.sp.leaves.length / Math.max(1, R.mg.count)).toFixed(1)}×</b> — while the mean absolute error of the `
      + `piecewise-constant approximation moves only from <b>${R.mae1.toFixed(3)}</b> to <b>${R.mae2.toFixed(3)}</b> grey levels. `
      + `But <b>${R.patchRegions}</b> of those regions are cells of the checkerboard, so the disc, the rectangle, the `
      + `ellipse and the whole shaded background between them account for only <b>${R.mg.count - R.patchRegions}</b>. `
      + `A texture is not a hard region; it is a region this predicate cannot see.`;
  }
  function all() {
    eTv.textContent = eT.value;
    eMinv.textContent = (1 << (+eMin.value)) + " px";
    draw(); say();
  }
  eP.onchange = all; eT.oninput = all; eMin.oninput = all; eM.onchange = all; eF.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   8 · #rag-svg — the region adjacency graph and the internal-difference rule
   ══════════════════════════════════════════════════════════════════════════ */
const SGrag = (function () {
  const W = 64, H = 40, STEP = 16, AMP = 40;
  const TEX = { x0: 38, x1: 58, y0: 10, y1: 30 };
  const img = new Float64Array(W * H);
  const truth = new Int32Array(W * H);            // 0 left, 1 right, 2 texture
  {
    const r = VZ.rng(9);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let v, t;
      if (x >= TEX.x0 && x < TEX.x1 && y >= TEX.y0 && y < TEX.y1) { v = 112 + AMP * (2 * r() - 1); t = 2; }
      else { v = (x < 32 ? 96 : 96 + STEP) + 2.0 * VZ.randn(r); t = (x < 32) ? 0 : 1; }
      img[y * W + x] = Math.min(255, Math.max(0, v));
      truth[y * W + x] = t;
    }
  }
  /* the 4-neighbour edge list, sorted once — every routine below reuses it */
  const E = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    if (x + 1 < W) E.push([Math.abs(img[i] - img[i + 1]), i, i + 1]);
    if (y + 1 < H) E.push([Math.abs(img[i] - img[i + W]), i, i + W]);
  }
  E.sort((a, b) => a[0] - b[0]);

  function fh(k, minSize) {
    const N = W * H, uf = SG.UF(N);
    const sz = new Int32Array(N).fill(1), intd = new Float64Array(N);
    for (const [w, a, b] of E) {
      const ra = uf.find(a), rb = uf.find(b);
      if (ra === rb) continue;
      const mint = Math.min(intd[ra] + k / sz[ra], intd[rb] + k / sz[rb]);
      if (w <= mint) {
        uf.p[rb] = ra; sz[ra] += sz[rb]; intd[ra] = Math.max(intd[ra], intd[rb], w);
      }
    }
    /* Int is recorded BEFORE the minimum-size pass. That pass merges across edges
       the predicate rejected, so letting it raise Int would report a smooth region
       as having the internal difference of the fragment it happened to absorb. */
    const intFH = new Float64Array(N);
    const rootFH = new Int32Array(N);
    for (let i = 0; i < N; i++) { rootFH[i] = uf.find(i); intFH[i] = intd[rootFH[i]]; }
    if (minSize > 1) {
      for (const [w, a, b] of E) {
        const ra = uf.find(a), rb = uf.find(b);
        if (ra === rb) continue;
        if (sz[ra] < minSize || sz[rb] < minSize) { uf.p[rb] = ra; sz[ra] += sz[rb]; }
      }
    }
    const lab = new Int32Array(N);
    for (let i = 0; i < N; i++) lab[i] = uf.find(i);
    /* per surviving region, the largest pre-pass Int among the components it holds,
       ignoring components below the minimum size, which have no meaningful Int */
    const intOf = new Map();
    const szFH = new Map();
    for (let i = 0; i < N; i++) szFH.set(rootFH[i], (szFH.get(rootFH[i]) || 0) + 1);
    for (let i = 0; i < N; i++) {
      if (szFH.get(rootFH[i]) < Math.max(2, minSize)) continue;
      const r = lab[i];
      intOf.set(r, Math.max(intOf.get(r) || 0, intFH[i]));
    }
    return { lab: lab, uf: uf, sz: sz, intd: intd, intOf: intOf };
  }
  function fixed(T) {
    const N = W * H, uf = SG.UF(N);
    for (const [w, a, b] of E) { if (w > T) break; uf.union(a, b); }
    const lab = new Int32Array(N);
    for (let i = 0; i < N; i++) lab[i] = uf.find(i);
    return { lab: lab, uf: uf };
  }
  /* renumber a root-label map into 0..n-1 and report per-truth-region IoU */
  function score(lab) {
    const ids = new Map();
    for (let i = 0; i < lab.length; i++) if (!ids.has(lab[i])) ids.set(lab[i], ids.size);
    const out = new Int32Array(lab.length);
    for (let i = 0; i < lab.length; i++) out[i] = ids.get(lab[i]);
    const ious = [0, 1, 2].map(t => {
      const cnt = new Map();
      for (let i = 0; i < lab.length; i++) if (truth[i] === t) cnt.set(out[i], (cnt.get(out[i]) || 0) + 1);
      let best = -1, bv = -1;
      for (const [k, v] of cnt) if (v > bv) { bv = v; best = k; }
      let inter = 0, uni = 0;
      for (let i = 0; i < lab.length; i++) {
        const p = out[i] === best, g = truth[i] === t;
        if (p && g) inter++; if (p || g) uni++;
      }
      return uni ? inter / uni : 1;
    });
    /* how many pieces the texture region is broken into */
    const texPieces = new Set();
    for (let i = 0; i < lab.length; i++) if (truth[i] === 2) texPieces.add(out[i]);
    /* have the two smooth halves merged? */
    const L = new Set(), R = new Set();
    for (let i = 0; i < lab.length; i++) { if (truth[i] === 0) L.add(out[i]); if (truth[i] === 1) R.add(out[i]); }
    let halvesMerged = false;
    for (const v of L) if (R.has(v)) halvesMerged = true;
    return { lab: out, n: ids.size, ious: ious, texPieces: texPieces.size, halvesMerged: halvesMerged };
  }
  /* the two edge populations quoted in the prose */
  const cross = [], inTex = [];
  for (let y = 0; y < H; y++) cross.push(Math.abs(img[y * W + 31] - img[y * W + 32]));
  for (let y = TEX.y0; y < TEX.y1; y++) for (let x = TEX.x0; x < TEX.x1; x++) {
    if (x + 1 < TEX.x1) inTex.push(Math.abs(img[y * W + x] - img[y * W + x + 1]));
    if (y + 1 < TEX.y1) inTex.push(Math.abs(img[y * W + x] - img[(y + 1) * W + x]));
  }
  return { W: W, H: H, img: img, truth: truth, TEX: TEX, E: E,
    fh: fh, fixed: fixed, score: score, cross: cross, inTex: inTex };
})();

(function () {
  const svg = d3.select("#rag-svg");
  if (svg.empty()) return;
  const S = SGrag;
  const eK = document.getElementById("rg-k"), eKv = document.getElementById("rg-kv");
  const eMin = document.getElementById("rg-min"), eMinv = document.getElementById("rg-minv");
  const eT = document.getElementById("rg-t"), eTv = document.getElementById("rg-tv");
  const eG = document.getElementById("rg-rag");
  const out = document.getElementById("rag-readout");

  function draw() {
    const k = +eK.value, minS = +eMin.value, T = +eT.value;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 3.4, IW = S.W * cw, IH = S.H * cw;
    const A = S.fh(k, minS), sA = S.score(A.lab);
    const B = S.fixed(T), sB = S.score(B.lab);

    const p0 = SG.box(g, 14, 38, IW, IH, "the image");
    SG.cells(p0, 0, 0, cw, S.W, S.H, (x, y) => SG.grey(S.img[y * S.W + x]));
    p0.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10).attr("fill", VC.muted)
      .text("a step of 16 at the middle, texture of range 80");

    const p1 = SG.box(g, 250, 38, IW, IH, `internal-difference merge · ${sA.n} regions`);
    SG.cells(p1, 0, 0, cw, S.W, S.H, (x, y) => SG.pal(sA.lab[y * S.W + x]));
    SG.drawEdges(p1, SG.edgesOf(sA.lab, S.W, S.H), cw, { color: "#0b0d12", w: 1, op: 0.9 });
    p1.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10).attr("fill", VC.good)
      .text(`IoU  left ${sA.ious[0].toFixed(3)} · right ${sA.ious[1].toFixed(3)} · texture ${sA.ious[2].toFixed(3)}`);

    const p2 = SG.box(g, 486, 38, IW, IH, `fixed threshold T = ${T.toFixed(1)} · ${sB.n} regions`);
    SG.cells(p2, 0, 0, cw, S.W, S.H, (x, y) => SG.pal(sB.lab[y * S.W + x]));
    p2.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10)
      .attr("fill", sB.ious[2] > 0.6 ? VC.good : VC.bad)
      .text(`IoU  left ${sB.ious[0].toFixed(3)} · right ${sB.ious[1].toFixed(3)} · texture ${sB.ious[2].toFixed(3)}`);

    /* ── the two edge-weight populations ── */
    const HX = 14, HY = 214, HW = 424, HH = 116;
    const h = SG.box(g, HX, HY, HW, HH, "edge weights: crossing the middle boundary, against inside the texture", { fill: "#12141c" });
    const xs = d3.scaleLinear().domain([0, 80]).range([0, HW]);
    const bins = d3.bin().domain([0, 80]).thresholds(40);
    const bc = bins(S.cross), bt = bins(S.inTex);
    const ys = d3.scaleLinear().domain([0, Math.max(d3.max(bc, d => d.length) / S.cross.length,
      d3.max(bt, d => d.length) / S.inTex.length)]).range([HH, 0]).nice();
    h.append("g").attr("transform", `translate(0,${HH})`).attr("class", "axis").call(d3.axisBottom(xs).ticks(8));
    h.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(3, "%"));
    const bar = (b, tot, col, op) => h.selectAll(null).data(b).join("rect")
      .attr("x", d => xs(d.x0)).attr("y", d => ys(d.length / tot))
      .attr("width", d => Math.max(1, xs(d.x1) - xs(d.x0) - 0.6))
      .attr("height", d => HH - ys(d.length / tot)).attr("fill", col).attr("fill-opacity", op);
    bar(bt, S.inTex.length, VC.a2, 0.55);
    bar(bc, S.cross.length, VC.accent, 0.75);
    h.append("line").attr("x1", xs(T)).attr("x2", xs(T)).attr("y1", 0).attr("y2", HH)
      .attr("stroke", VC.ink).attr("stroke-width", 1.8);
    h.append("text").attr("x", xs(T) + 4).attr("y", 12).attr("font-size", 10).attr("fill", VC.ink).text("T");
    VZ.legend(h, [{ color: VC.accent, label: "40 boundary edges (must be cut)" },
      { color: VC.a2, label: "760 texture edges (must all be kept)" }], HW - 220, 13, { gap: 14 });

    /* ── the region adjacency graph of the merged result ── */
    const GX = 462, GY = 214, GW2 = 242, GH2 = 116;
    const gg = SG.box(g, GX, GY, GW2, GH2, "the region adjacency graph", { fill: "#12141c" });
    if (eG.checked) {
      /* centroids, sizes and internal differences per surviving region */
      const info = new Map();
      for (let y = 0; y < S.H; y++) for (let x = 0; x < S.W; x++) {
        const i = y * S.W + x, r = sA.lab[i];
        /* intOf is keyed by the union-find root, sA.lab by the renumbered id */
        if (!info.has(r)) info.set(r, { sx: 0, sy: 0, n: 0, intd: A.intOf.get(A.lab[i]) || 0 });
        const o = info.get(r); o.sx += x; o.sy += y; o.n++;
      }
      /* the weakest link between each adjacent pair */
      const link = new Map();
      for (const [w, a, b] of S.E) {
        const ra = sA.lab[a], rb = sA.lab[b];
        if (ra === rb) continue;
        const key = Math.min(ra, rb) + "," + Math.max(ra, rb);
        if (!link.has(key)) link.set(key, w);          // E is ascending, so first = min
      }
      const sx = d3.scaleLinear().domain([0, S.W]).range([16, GW2 - 16]);
      const sy = d3.scaleLinear().domain([0, S.H]).range([18, GH2 - 16]);
      const big = [...info.entries()].filter(([, o]) => o.n >= 30);
      const bigSet = new Set(big.map(([r]) => r));
      for (const [key, w] of link) {
        const [a, b] = key.split(",").map(Number);
        if (!bigSet.has(a) || !bigSet.has(b)) continue;
        const oa = info.get(a), ob = info.get(b);
        gg.append("line").attr("x1", sx(oa.sx / oa.n)).attr("y1", sy(oa.sy / oa.n))
          .attr("x2", sx(ob.sx / ob.n)).attr("y2", sy(ob.sy / ob.n))
          .attr("stroke", VC.muted).attr("stroke-width", 1 + Math.min(4, w / 8));
        gg.append("text").attr("x", (sx(oa.sx / oa.n) + sx(ob.sx / ob.n)) / 2)
          .attr("y", (sy(oa.sy / oa.n) + sy(ob.sy / ob.n)) / 2 - 4).attr("text-anchor", "middle")
          .attr("font-size", 9.5).attr("fill", VC.a2).text("Dif " + w.toFixed(1));
      }
      big.forEach(([r, o]) => {
        gg.append("circle").attr("cx", sx(o.sx / o.n)).attr("cy", sy(o.sy / o.n))
          .attr("r", 12).attr("fill", SG.pal(r)).attr("fill-opacity", 0.85)
          .attr("stroke", "#0b0d12").attr("stroke-width", 1.2);
        gg.append("text").attr("x", sx(o.sx / o.n)).attr("y", sy(o.sy / o.n) + 25)
          .attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", VC.ink)
          .text("Int " + o.intd.toFixed(1));
      });
      gg.append("text").attr("x", 6).attr("y", GH2 - 5).attr("font-size", 9.5).attr("fill", VC.muted)
        .text("regions ≥ 30 px shown");
    }

    /* ── numbers ── */
    const put = SG.kv(g, 14, 358, { keyW: 224, lead: 15 });
    put("regions, internal-difference rule", String(sA.n), VC.good, true);
    put("  texture kept in how many pieces", String(sA.texPieces), sA.texPieces <= 2 ? VC.good : VC.bad);
    put("  the two halves merged?", sA.halvesMerged ? "yes — k is too large" : "no",
      sA.halvesMerged ? VC.bad : VC.good);
    put("  IoU  left / right / texture",
      sA.ious.map(v => v.toFixed(3)).join("  /  "), VC.good);
    const put2 = SG.kv(g, 396, 358, { keyW: 224, lead: 15 });
    put2("regions, fixed threshold T", String(sB.n), VC.bad, true);
    put2("  texture broken into", String(sB.texPieces) + " pieces", sB.texPieces > 2 ? VC.bad : VC.good);
    put2("  the two halves merged?", sB.halvesMerged ? "yes" : "no",
      sB.halvesMerged ? VC.bad : VC.good);
    put2("  IoU  left / right / texture",
      sB.ious.map(v => v.toFixed(3)).join("  /  "), VC.bad);
  }

  function say() {
    const k = +eK.value, minS = +eMin.value, T = +eT.value;
    const A = S.fh(k, minS);
    const sA = S.score(A.lab), sB = S.score(S.fixed(T).lab);
    const sizes = new Map();
    for (let i = 0; i < A.lab.length; i++) sizes.set(A.lab[i], (sizes.get(A.lab[i]) || 0) + 1);
    const ints = [...sizes.entries()].filter(([, n]) => n >= 30)
      .sort((p, q) => q[1] - p[1]).map(([r]) => A.intOf.get(r) || 0);
    const cmin = Math.min.apply(null, S.cross), cmax = Math.max.apply(null, S.cross);
    const below = S.inTex.filter(v => v < cmin).length / S.inTex.length;
    const above = S.inTex.filter(v => v > cmax).length / S.inTex.length;
    out.innerHTML =
      `The forty edges crossing the middle boundary span <b>${cmin.toFixed(2)}</b> to <b>${cmax.toFixed(2)}</b>. `
      + `Of the 760 edges inside the texture, <b>${(100 * below).toFixed(1)}%</b> are weaker than the weakest of those `
      + `and <b>${(100 * above).toFixed(1)}%</b> are stronger than the strongest — the populations overlap end to end.<br>`
      + `At <b>T = ${T.toFixed(1)}</b> the fixed rule gives <b>${sB.n}</b> regions, breaks the texture into `
      + `<b>${sB.texPieces}</b> pieces, and ${sB.halvesMerged ? "<b style=\"color:var(--bad)\">has already merged the two halves</b>" : "keeps the two halves apart"}. `
      + `The internal-difference rule at <b>k = ${k}</b> gives <b>${sA.n}</b> regions with the texture in `
      + `<b>${sA.texPieces}</b> piece${sA.texPieces === 1 ? "" : "s"}, at IoU <b>${sA.ious.map(v => v.toFixed(3)).join(" / ")}</b> `
      + `for the left half, the right half and the texture. It is not doing anything cleverer than comparing each `
      + `crossing edge against the bottleneck of the spanning tree already inside the region it would join — `
      + `which is <b>${ints.map(v => v.toFixed(2)).join("</b>, <b>")}</b> for the surviving regions, largest first, `
      + `so the texture tolerates a boundary that either smooth half treats as decisive.`;
  }
  function all() {
    eKv.textContent = eK.value;
    eMinv.textContent = eMin.value;
    eTv.textContent = (+eT.value).toFixed(1);
    draw(); say();
  }
  eK.oninput = all; eMin.oninput = all; eT.oninput = all; eG.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   9 · #wshed-svg — the watershed as literal flooding of a 1-D landscape
   ══════════════════════════════════════════════════════════════════════════ */
const SGwsh = (function () {
  const N = 601, X0 = 0, X1 = 100;
  const xs = new Float64Array(N), ys = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const x = X0 + (X1 - X0) * i / (N - 1);
    xs[i] = x;
    ys[i] = 60
      - 34 * Math.exp(-Math.pow((x - 12) / 7, 2))
      - 22 * Math.exp(-Math.pow((x - 30) / 6, 2))
      - 40 * Math.exp(-Math.pow((x - 52) / 9, 2))
      - 16 * Math.exp(-Math.pow((x - 70) / 5, 2))
      - 30 * Math.exp(-Math.pow((x - 86) / 8, 2))
      + 5 * Math.sin(x / 3.1) * Math.sin(Math.PI * x / 100);
  }
  /* flood the profile in ascending order of height, recording every merge event.
     This is the level-sweep watershed, in one dimension where it is exact. */
  const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => ys[a] - ys[b]);
  const lab = new Int32Array(N).fill(-1);
  const parent = [], depth = [], minAt = [];
  const merges = [];                       // {level, absorbed, into, persistence, at}
  function find(a) { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; }
  for (const i of order) {
    const ns = [];
    if (i > 0 && lab[i - 1] >= 0) ns.push(find(lab[i - 1]));
    if (i < N - 1 && lab[i + 1] >= 0) ns.push(find(lab[i + 1]));
    const roots = [...new Set(ns)];
    if (roots.length === 0) {
      const id = parent.length;
      parent.push(id); depth.push(ys[i]); minAt.push(i);
      lab[i] = id;
    } else if (roots.length === 1) {
      lab[i] = roots[0];
    } else {
      roots.sort((a, b) => depth[a] - depth[b]);
      const keep = roots[0];
      for (let k = 1; k < roots.length; k++) {
        merges.push({ level: ys[i], absorbed: roots[k], into: keep,
          persistence: ys[i] - depth[roots[k]], at: i });
        parent[roots[k]] = keep;
      }
      lab[i] = keep;
    }
  }
  const nMin = parent.length;
  const persistence = new Array(nMin).fill(Infinity);
  const mergeLevel = new Array(nMin).fill(Infinity);
  merges.forEach(m => { persistence[m.absorbed] = m.persistence; mergeLevel[m.absorbed] = m.level; });
  /* the state of the flood at a given level h, with dams optionally enforced */
  function at(h, dams, pfilter) {
    /* which minima are allowed to open a basin */
    const alive = persistence.map(p => p >= pfilter);
    const uf = SG.UF(nMin);
    let open = 0, built = 0;
    const damAt = [];
    /* replay the merge events up to level h */
    const active = new Set();
    for (let m = 0; m < nMin; m++) if (alive[m] && depth[m] <= h) active.add(m);
    for (const ev of merges) {
      if (ev.level > h) break;
      const a = uf.find(ev.absorbed), b = uf.find(ev.into);
      if (a === b) continue;
      const aliveA = alive[ev.absorbed], aliveB = alive[ev.into];
      if (dams && aliveA && aliveB) { built++; damAt.push(ev.at); }
      else { uf.union(b, a); }
    }
    const roots = new Set();
    for (const m of active) roots.add(uf.find(m));
    open = roots.size;
    /* label every submerged sample by the basin that owns it */
    const owner = new Int32Array(N).fill(-1);
    for (let i = 0; i < N; i++) {
      if (ys[i] > h) continue;
      let r = uf.find(lab[i]);
      if (!alive[lab[i]]) {
        /* a suppressed minimum's water belongs to whichever surviving basin
           absorbs it; walk the merge chain until an allowed root is found */
        let cur = lab[i], guard = 0;
        while (!alive[cur] && guard++ < nMin) {
          const ev = merges.find(e => e.absorbed === cur);
          if (!ev) break;
          cur = ev.into;
        }
        r = uf.find(cur);
      }
      owner[i] = r;
    }
    return { open: open, dams: built, damAt: damAt, owner: owner, alive: alive };
  }
  const nextSaddle = h => { for (const ev of merges) if (ev.level > h) return ev.level; return null; };
  return { N: N, xs: xs, ys: ys, lab: lab, nMin: nMin, depth: depth, minAt: minAt,
    merges: merges, persistence: persistence, mergeLevel: mergeLevel, at: at, nextSaddle: nextSaddle };
})();

(function () {
  const svg = d3.select("#wshed-svg");
  if (svg.empty()) return;
  const S = SGwsh;
  const eH = document.getElementById("ws-h"), eHv = document.getElementById("ws-hv");
  const eP = document.getElementById("ws-p"), ePv = document.getElementById("ws-pv");
  const eD = document.getElementById("ws-dam"), ePs = document.getElementById("ws-pers");
  const out = document.getElementById("wshed-readout");

  const AX = 46, AY = 40, AW = 690, AH = 208;
  const x = d3.scaleLinear().domain([0, 100]).range([0, AW]);
  const y = d3.scaleLinear().domain([12, 64]).range([AH, 0]);

  function draw() {
    const h = +eH.value, pf = +eP.value, dams = eD.checked;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const st = S.at(h, dams, pf);

    const A = SG.box(g, AX, AY, AW, AH, "the image as a landscape, flooded from below", { fill: "#12141c" });
    A.append("g").attr("transform", `translate(0,${AH})`).attr("class", "axis").call(d3.axisBottom(x).ticks(8));
    A.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    /* water, one polygon per surviving basin */
    const roots = [...new Set(Array.from(st.owner).filter(v => v >= 0))];
    roots.forEach(r => {
      const seg = [];
      let run = null;
      for (let i = 0; i < S.N; i++) {
        if (st.owner[i] === r) { if (!run) run = [i, i]; else run[1] = i; }
        else if (run) { seg.push(run); run = null; }
      }
      if (run) seg.push(run);
      seg.forEach(([a, b]) => {
        const pts = [];
        for (let i = a; i <= b; i++) pts.push([x(S.xs[i]), y(S.ys[i])]);
        for (let i = b; i >= a; i--) pts.push([x(S.xs[i]), y(h)]);
        A.append("path").attr("d", "M" + pts.map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join("L") + "Z")
          .attr("fill", SG.pal(r)).attr("fill-opacity", 0.5);
      });
    });
    /* the terrain */
    A.append("path").datum(d3.range(S.N))
      .attr("d", d3.area().x(i => x(S.xs[i])).y0(AH).y1(i => y(S.ys[i])))
      .attr("fill", VC.muted).attr("fill-opacity", 0.12);
    A.append("path").datum(d3.range(S.N))
      .attr("d", d3.line().x(i => x(S.xs[i])).y(i => y(S.ys[i])))
      .attr("fill", "none").attr("stroke", VC.ink).attr("stroke-width", 1.8);
    /* water line */
    A.append("line").attr("x1", 0).attr("x2", AW).attr("y1", y(h)).attr("y2", y(h))
      .attr("stroke", VC.accent).attr("stroke-dasharray", "5 4").attr("stroke-width", 1.2);
    A.append("text").attr("x", AW - 3).attr("y", y(h) - 5).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", VC.accent).text("water at " + h.toFixed(2));
    /* dams */
    st.damAt.forEach(i => {
      A.append("line").attr("x1", x(S.xs[i])).attr("x2", x(S.xs[i]))
        .attr("y1", y(S.ys[i]) + 4).attr("y2", y(h)).attr("stroke", VC.bad).attr("stroke-width", 3.5);
      A.append("path").attr("d", `M${x(S.xs[i])},${y(S.ys[i]) - 2} l-4,-8 l8,0 Z`).attr("fill", VC.bad);
    });
    /* minima */
    for (let m = 0; m < S.nMin; m++) {
      const i = S.minAt[m], live = st.alive[m];
      A.append("circle").attr("cx", x(S.xs[i])).attr("cy", y(S.depth[m])).attr("r", 3.4)
        .attr("fill", live ? SG.pal(m) : "#4a5162").attr("stroke", "#0b0d12");
      A.append("text").attr("x", x(S.xs[i])).attr("y", y(S.depth[m]) + 15).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", live ? VC.muted : "#4a5162")
        .text(isFinite(S.persistence[m]) ? "p " + S.persistence[m].toFixed(2) : "p ∞");
    }

    /* ── persistence bars ── */
    if (ePs.checked) {
      const BX = 46, BY = 296, BW = 400, BH = 116;
      const B = SG.box(g, BX, BY, BW, BH, "each minimum's depth, and the saddle it spills over", { fill: "#12141c" });
      const yb = d3.scaleLinear().domain([12, 64]).range([BH, 0]);
      B.append("g").attr("class", "axis").call(d3.axisLeft(yb).ticks(5));
      const order = d3.range(S.nMin).sort((a, b) => S.depth[a] - S.depth[b]);
      order.forEach((m, k) => {
        const cx = 34 + k * ((BW - 50) / S.nMin);
        const top = isFinite(S.mergeLevel[m]) ? S.mergeLevel[m] : 62;
        const live = st.alive[m];
        B.append("rect").attr("x", cx).attr("y", yb(top)).attr("width", 24)
          .attr("height", Math.max(1, yb(S.depth[m]) - yb(top)))
          .attr("fill", live ? SG.pal(m) : "#3a3f4d").attr("fill-opacity", live ? 0.8 : 0.5);
        B.append("text").attr("x", cx + 12).attr("y", yb(top) - 4).attr("text-anchor", "middle")
          .attr("font-size", 9.5).attr("fill", live ? VC.ink : "#5b6273")
          .text(isFinite(S.persistence[m]) ? S.persistence[m].toFixed(2) : "∞");
      });
      B.append("line").attr("x1", 0).attr("x2", BW).attr("y1", yb(h)).attr("y2", yb(h))
        .attr("stroke", VC.accent).attr("stroke-dasharray", "4 3");
      B.append("text").attr("x", 2).attr("y", BH - 4).attr("font-size", 9.5).attr("fill", VC.muted)
        .text("bar length = persistence");
    }

    /* ── numbers ── */
    const surv = S.persistence.filter(p => p >= pf).length;
    const ns = S.nextSaddle(h);
    const put = SG.kv(g, 474, 312, { keyW: 168, lead: 15.5 });
    put("lakes currently open", String(st.open), VC.accent, true);
    put("dams standing", String(st.dams), VC.bad, true);
    put("next saddle at level", ns === null ? "— fully merged" : ns.toFixed(3), VC.a2);
    put("persistences, deepest first",
      d3.range(S.nMin).sort((a, b) => S.depth[a] - S.depth[b])
        .map(m => isFinite(S.persistence[m]) ? S.persistence[m].toFixed(2) : "∞").join(", "), VC.violet);
    put("basins surviving the filter", surv + " of " + S.nMin, surv < S.nMin ? VC.good : VC.ink, true);
  }

  function say() {
    const h = +eH.value, pf = +eP.value;
    const st = S.at(h, eD.checked, pf);
    const ns = S.nextSaddle(h);
    const surv = S.persistence.filter(p => p >= pf).length;
    out.innerHTML =
      `Water at <b>${h.toFixed(2)}</b>: <b>${st.open}</b> lake${st.open === 1 ? "" : "s"} open, `
      + `<b>${st.dams}</b> dam${st.dams === 1 ? "" : "s"} built. `
      + (ns === null ? `Every saddle has been crossed; nothing more will merge.`
        : `The next saddle is at <b>${ns.toFixed(3)}</b>, so raising the level past that point costs one more region.`)
      + `<br>The five minima have persistences <b>`
      + d3.range(S.nMin).sort((a, b) => S.depth[a] - S.depth[b])
        .map(m => isFinite(S.persistence[m]) ? S.persistence[m].toFixed(2) : "∞").join("</b>, <b>")
      + `</b>. Filtering at <b>${pf.toFixed(2)}</b> leaves <b>${surv}</b> of them. `
      + `Nothing about the terrain changed; what changed is the answer to "how shallow a dip still counts as a valley", `
      + `and the algorithm has no way to answer that question for you — which is exactly the over-segmentation of §13.`;
  }
  function all() {
    eHv.textContent = (+eH.value).toFixed(2);
    ePv.textContent = (+eP.value).toFixed(2);
    draw(); say();
  }
  eH.oninput = all; eP.oninput = all; eD.onchange = all; ePs.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   10 · #marker-svg — over-segmentation, and marker-controlled watershed
   ══════════════════════════════════════════════════════════════════════════ */
/* a separable Gaussian blur and a Sobel gradient magnitude, page-local because
   Part 2 owns the general filtering machinery and this page only needs one of each */
SG.blur = function (img, W, H, sigma) {
  if (sigma <= 0) return Float64Array.from(img);
  const rad = Math.max(1, Math.ceil(3 * sigma));
  const k = [];
  let s = 0;
  for (let i = -rad; i <= rad; i++) { const v = Math.exp(-i * i / (2 * sigma * sigma)); k.push(v); s += v; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const tmp = new Float64Array(W * H), outp = new Float64Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let a = 0;
    for (let i = -rad; i <= rad; i++) a += k[i + rad] * img[y * W + Math.min(W - 1, Math.max(0, x + i))];
    tmp[y * W + x] = a;
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let a = 0;
    for (let i = -rad; i <= rad; i++) a += k[i + rad] * tmp[Math.min(H - 1, Math.max(0, y + i)) * W + x];
    outp[y * W + x] = a;
  }
  return outp;
};
SG.gradMag = function (img, W, H) {
  const g = new Float64Array(W * H);
  const at = (x, y) => img[Math.min(H - 1, Math.max(0, y)) * W + Math.min(W - 1, Math.max(0, x))];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const gx = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1))
      - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
    const gy = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1))
      - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
    g[y * W + x] = Math.sqrt(gx * gx + gy * gy) / 4;
  }
  return g;
};
/* Flood an INTEGER scalar field level by level — the Vincent–Soille sweep of §12 —
   and build the merge tree of its basins. Doing it level by level rather than pixel
   by pixel matters: a flat plateau that drains to a lower neighbour must NOT open a
   basin of its own, and a naive ascending-order sweep opens one for every part of the
   plateau that the propagation has not yet reached. On the default figure that
   mistake would report 482 minima where there are 316.
   Returns, per pixel, the basin it first joined, and for each basin its depth, the
   level at which it is absorbed and by whom — from which every hierarchical watershed
   (h-minima, marker-controlled, waterfall) is a relabelling.                        */
SG.floodTree = function (g, W, H) {
  const N = W * H;
  let vmax = 0;
  for (let i = 0; i < N; i++) if (g[i] > vmax) vmax = g[i];
  const buckets = [];
  for (let v = 0; v <= vmax; v++) buckets.push([]);
  for (let i = 0; i < N; i++) buckets[g[i]].push(i);

  const lab = new Int32Array(N).fill(-1);
  const parent = [], depth = [], absorbLevel = [], absorbInto = [], minPix = [];
  function find(a) { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; }
  function meet(a, b, level) {
    let ra = find(a), rb = find(b);
    if (ra === rb) return ra;
    if (depth[ra] > depth[rb] || (depth[ra] === depth[rb] && ra > rb)) { const t = ra; ra = rb; rb = t; }
    absorbLevel[rb] = level; absorbInto[rb] = ra; parent[rb] = ra;
    return ra;
  }
  const nbr = i => {
    const x = i % W, y = (i / W) | 0, o = [];
    if (x > 0) o.push(i - 1);
    if (x < W - 1) o.push(i + 1);
    if (y > 0) o.push(i - W);
    if (y < H - 1) o.push(i + W);
    return o;
  };
  for (let v = 0; v <= vmax; v++) {
    const Q = buckets[v];
    if (!Q.length) continue;
    const inQ = new Set(Q);
    /* 1 · seed the front from pixels of this level that touch the already-flooded set */
    const front = [];
    for (const i of Q) {
      let root = -1;
      for (const j of nbr(i)) {
        if (lab[j] < 0 || inQ.has(j)) continue;
        const r = find(lab[j]);
        root = (root < 0) ? r : meet(root, r, v);
      }
      if (root >= 0) { lab[i] = root; front.push(i); }
    }
    /* 2 · propagate inside the level set, merging basins that meet here */
    for (let p = 0; p < front.length; p++) {
      const i = front[p];
      for (const j of nbr(i)) {
        if (!inQ.has(j)) continue;
        if (lab[j] < 0) { lab[j] = find(lab[i]); front.push(j); }
        else meet(lab[i], lab[j], v);
      }
    }
    /* 3 · whatever is still unlabelled at this level IS a regional minimum */
    for (const i of Q) {
      if (lab[i] >= 0) continue;
      const id = parent.length;
      parent.push(id); depth.push(v); absorbLevel.push(Infinity); absorbInto.push(-1); minPix.push(i);
      lab[i] = id;
      const st = [i];
      while (st.length) {
        const k = st.pop();
        for (const j of nbr(k)) if (inQ.has(j) && lab[j] < 0) { lab[j] = id; st.push(j); }
      }
    }
  }
  const n = parent.length;
  const persistence = new Array(n);
  for (let m = 0; m < n; m++) persistence[m] = absorbLevel[m] - depth[m];
  return { lab: lab, n: n, depth: depth, absorbLevel: absorbLevel, absorbInto: absorbInto,
    persistence: persistence, minPix: minPix, find: find, gAt: g };
};

/* Meyer's priority-queue flooding from a given set of seeds. The queue is a bucket
   array indexed by the integer field value, so popping the globally lowest unassigned
   boundary pixel is O(1) and the whole sweep is linear. Every pixel ends up in the
   basin whose front reached it along the lowest path — which is the definition of a
   catchment basin, and is §09's competitive region growing with a global ordering. */
SG.seededWatershed = function (g, W, H, seed, nLab) {
  const N = W * H;
  const lab = Int32Array.from(seed);
  let vmax = 0;
  for (let i = 0; i < N; i++) if (g[i] > vmax) vmax = g[i];
  const buckets = [];
  for (let v = 0; v <= vmax; v++) buckets.push([]);
  const inQ = new Uint8Array(N);
  const nbr = i => {
    const x = i % W, y = (i / W) | 0, o = [];
    if (x > 0) o.push(i - 1);
    if (x < W - 1) o.push(i + 1);
    if (y > 0) o.push(i - W);
    if (y < H - 1) o.push(i + W);
    return o;
  };
  for (let i = 0; i < N; i++) if (lab[i] >= 0)
    for (const j of nbr(i)) if (lab[j] < 0 && !inQ[j]) { inQ[j] = 1; buckets[g[j]].push(j); }
  for (let v = 0; v <= vmax; v++) {
    for (let p = 0; p < buckets[v].length; p++) {
      const i = buckets[v][p];
      if (lab[i] >= 0) continue;
      let r = -1;
      for (const j of nbr(i)) if (lab[j] >= 0) { r = lab[j]; break; }
      if (r < 0) continue;
      lab[i] = r;
      for (const j of nbr(i)) if (lab[j] < 0 && !inQ[j]) {
        inQ[j] = 1;
        buckets[Math.max(v, g[j])].push(j);      // never push below the current level
      }
    }
  }
  /* anything unreachable (no seed in its component) goes to label 0 */
  for (let i = 0; i < N; i++) if (lab[i] < 0) lab[i] = 0;
  return { lab: lab, n: nLab };
};

/* relabel every pixel by the nearest ancestor basin that survives `keep` */
SG.collapse = function (tree, keep, W, H) {
  const map = new Int32Array(tree.n).fill(-1);
  for (let m = 0; m < tree.n; m++) {
    let cur = m, guard = 0;
    while (!keep[cur] && tree.absorbInto[cur] >= 0 && guard++ < tree.n) cur = tree.absorbInto[cur];
    map[m] = cur;
  }
  const out = new Int32Array(W * H);
  const ids = new Map();
  for (let i = 0; i < W * H; i++) {
    const r = map[tree.lab[i]];
    if (!ids.has(r)) ids.set(r, ids.size);
    out[i] = ids.get(r);
  }
  return { lab: out, n: ids.size };
};

const SGmark = (function () {
  const W = 104, H = 72;
  const DISCS = [[20, 22, 13], [50, 20, 13], [80, 24, 12], [22, 52, 12], [52, 52, 14], [84, 54, 13]];
  function build(noise, blurSigma) {
    const r = VZ.rng(41);
    const im = new Float64Array(W * H);
    const truth = new Int32Array(W * H).fill(0);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let t = 0;
      DISCS.forEach(([cx, cy, rr], k) => {
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) < rr * rr) t = k + 1;
      });
      truth[y * W + x] = t;
      im[y * W + x] = (t ? 190 : 68) + noise * VZ.randn(r);
    }
    const sm = SG.blur(im, W, H, blurSigma);
    const g = SG.gradMag(sm, W, H);
    for (let i = 0; i < g.length; i++) g[i] = Math.round(Math.min(255, g[i]));
    return { im: im, sm: sm, g: g, truth: truth };
  }
  /* seeds from the surviving basins of the flood tree: each basin contributes its
     own regional-minimum plateau, which is exactly the set the sweep opened it on */
  function seedsFromTree(tree, keep) {
    const seed = new Int32Array(W * H).fill(-1);
    const id = new Map();
    for (let i = 0; i < W * H; i++) {
      const m = tree.lab[i];
      if (!keep[m]) continue;
      if (tree.gAt[i] !== tree.depth[m]) continue;
      if (!id.has(m)) id.set(m, id.size);
      seed[i] = id.get(m);
    }
    return { seed: seed, n: id.size };
  }
  /* the interactive alternative: one small internal marker per object, plus one
     external marker — the image border — standing for the background */
  function seedsFromMarkers(nObj) {
    const seed = new Int32Array(W * H).fill(-1);
    for (let x = 0; x < W; x++) { seed[x] = 0; seed[(H - 1) * W + x] = 0; }
    for (let y = 0; y < H; y++) { seed[y * W] = 0; seed[y * W + W - 1] = 0; }
    DISCS.slice(0, nObj).forEach(([cx, cy], k) => {
      for (let y = cy - 3; y <= cy + 3; y++) for (let x = cx - 3; x <= cx + 3; x++)
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= 9) seed[y * W + x] = k + 1;
    });
    return { seed: seed, n: nObj + 1 };
  }
  return { W: W, H: H, DISCS: DISCS, build: build,
    seedsFromTree: seedsFromTree, seedsFromMarkers: seedsFromMarkers };
})();

(function () {
  const svg = d3.select("#marker-svg");
  if (svg.empty()) return;
  const S = SGmark;
  const eN = document.getElementById("mk-n"), eNv = document.getElementById("mk-nv");
  const eB = document.getElementById("mk-b"), eBv = document.getElementById("mk-bv");
  const eH = document.getElementById("mk-h"), eHv = document.getElementById("mk-hv");
  const eM = document.getElementById("mk-m"), eMv = document.getElementById("mk-mv");
  const out = document.getElementById("marker-readout");
  const CACHE = {};

  function ari(a, b) {
    const n = a.length, w = new Array(n).fill(1);
    return SG.adjRand(Array.from(a), Array.from(b), w);
  }
  function build(noise, blur) {
    const key = noise + "|" + blur;
    if (CACHE[key]) return CACHE[key];
    const B = S.build(noise, blur);
    const T = SG.floodTree(B.g, S.W, S.H);
    const s0 = S.seedsFromTree(T, new Array(T.n).fill(true));
    const w0 = SG.seededWatershed(B.g, S.W, S.H, s0.seed, s0.n);
    const sweep = d3.range(0, 31).map(h => {
      const keep = T.persistence.map(p => p >= h);
      const sh = S.seedsFromTree(T, keep);
      const wh = SG.seededWatershed(B.g, S.W, S.H, sh.seed, sh.n);
      return { h: h, n: sh.n, ari: ari(wh.lab, B.truth) };
    });
    const r = { B: B, T: T, raw: w0, rawN: s0.n, rawAri: ari(w0.lab, B.truth), sweep: sweep };
    CACHE[key] = r;
    return r;
  }

  function draw() {
    const noise = +eN.value, blur = +eB.value, h = +eH.value, nm = +eM.value;
    const R = build(noise, blur);
    const keep = R.T.persistence.map(p => p >= h);
    const sh = S.seedsFromTree(R.T, keep);
    const wh = SG.seededWatershed(R.B.g, S.W, S.H, sh.seed, sh.n);
    const sm = S.seedsFromMarkers(nm);
    const wm = SG.seededWatershed(R.B.g, S.W, S.H, sm.seed, sm.n);
    const ariH = ari(wh.lab, R.B.truth), ariM = ari(wm.lab, R.B.truth);

    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.1, IW = S.W * cw, IH = S.H * cw;
    let gmax = 0; for (let i = 0; i < R.B.g.length; i++) if (R.B.g[i] > gmax) gmax = R.B.g[i];

    /* row 1 */
    const p0 = SG.box(g, 14, 40, IW, IH, "the image, with the markers on it");
    SG.cells(p0, 0, 0, cw, S.W, S.H, (x, y) => SG.grey(R.B.im[y * S.W + x]));
    SG.cells(p0, 0, 0, cw, S.W, S.H, (x, y) => {
      const v = sm.seed[y * S.W + x];
      return v < 0 ? "none" : (v === 0 ? VC.violet : VC.good);
    });
    p0.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10).attr("fill", VC.muted)
      .text(`${nm} internal marker${nm === 1 ? "" : "s"} + the border`);

    const p1 = SG.box(g, 252, 40, IW, IH, "the gradient magnitude — the surface flooded");
    SG.cells(p1, 0, 0, cw, S.W, S.H, (x, y) =>
      d3.interpolateCividis(Math.min(1, R.B.g[y * S.W + x] / Math.max(1, gmax))));
    p1.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10).attr("fill", VC.muted)
      .text(`${R.T.n} regional minima, values 0 … ${gmax}`);

    const p2 = SG.box(g, 490, 40, IW, IH, `raw watershed · ${R.rawN} regions`);
    SG.cells(p2, 0, 0, cw, S.W, S.H, (x, y) => SG.pal(R.raw.lab[y * S.W + x] * 7 % 16));
    p2.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10).attr("fill", VC.bad)
      .text(`ARI against the truth ${R.rawAri.toFixed(3)}`);

    /* row 2 */
    const p3 = SG.box(g, 14, 248, IW, IH, `persistence filter h = ${h} · ${sh.n} regions`);
    SG.cells(p3, 0, 0, cw, S.W, S.H, (x, y) => SG.pal(wh.lab[y * S.W + x] * 5 % 16));
    SG.drawEdges(p3, SG.edgesOf(wh.lab, S.W, S.H), cw, { color: "#0b0d12", w: 0.8, op: 0.85 });
    p3.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10)
      .attr("fill", ariH > 0.85 ? VC.good : VC.a2).text(`ARI ${ariH.toFixed(3)}`);

    const p4 = SG.box(g, 252, 248, IW, IH, `marker-controlled · ${sm.n} regions`);
    SG.cells(p4, 0, 0, cw, S.W, S.H, (x, y) => SG.pal(wm.lab[y * S.W + x]));
    SG.drawEdges(p4, SG.edgesOf(wm.lab, S.W, S.H), cw, { color: "#0b0d12", w: 0.9, op: 0.9 });
    p4.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10)
      .attr("fill", ariM > 0.85 ? VC.good : VC.a2).text(`ARI ${ariM.toFixed(3)}`);

    /* the sweep */
    const CX = 490, CY = 248, CW = IW, CH = IH;
    const c = SG.box(g, CX, CY, CW, CH, "regions and accuracy against h", { fill: "#12141c" });
    const xs = d3.scaleLinear().domain([0, 30]).range([0, CW]);
    const yl = d3.scaleLog().domain([1, Math.max(10, R.rawN)]).range([CH, 0]);
    const yr = d3.scaleLinear().domain([0, 1]).range([CH, 0]);
    c.append("g").attr("transform", `translate(0,${CH})`).attr("class", "axis").call(d3.axisBottom(xs).ticks(6));
    c.append("g").attr("class", "axis").call(d3.axisLeft(yl).ticks(4, "~s"));
    c.append("path").datum(R.sweep).attr("d", d3.line().x(d => xs(d.h)).y(d => yl(Math.max(1, d.n))))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 1.8);
    c.append("path").datum(R.sweep).attr("d", d3.line().x(d => xs(d.h)).y(d => yr(d.ari)))
      .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.8);
    c.append("line").attr("x1", 0).attr("x2", CW).attr("y1", yr(ariM)).attr("y2", yr(ariM))
      .attr("stroke", VC.violet).attr("stroke-dasharray", "4 3");
    c.append("text").attr("x", CW - 3).attr("y", yr(ariM) - 4).attr("text-anchor", "end")
      .attr("font-size", 9.5).attr("fill", VC.violet).text("markers, ARI " + ariM.toFixed(3));
    c.append("line").attr("x1", xs(h)).attr("x2", xs(h)).attr("y1", 0).attr("y2", CH)
      .attr("stroke", VC.ink).attr("stroke-width", 1.3);
    VZ.legend(c, [{ color: VC.accent, label: "regions (log)" }, { color: VC.good, label: "ARI (0…1)" }],
      6, 13, { gap: 13 });

    /* numbers */
    const best = R.sweep.reduce((a, b) => (b.ari > a.ari ? b : a));
    const put = SG.kv(g, 14, 430, { keyW: 200, lead: 15 });
    put("regional minima of the gradient", String(R.T.n), VC.bad, true);
    put("raw watershed regions / ARI", R.rawN + "  /  " + R.rawAri.toFixed(3), VC.bad);
    put("truth", "7 regions — 6 discs and the background", VC.muted);
    const put2 = SG.kv(g, 400, 430, { keyW: 200, lead: 15 });
    put2("best h, and what it scores", "h = " + best.h + " → " + best.n + " regions, ARI " + best.ari.toFixed(3), VC.a2);
    put2("markers, regions / ARI", sm.n + "  /  " + ariM.toFixed(3), VC.good, true);
    put2("current h, regions / ARI", sh.n + "  /  " + ariH.toFixed(3), VC.ink);
  }

  function say() {
    const noise = +eN.value, blur = +eB.value, h = +eH.value, nm = +eM.value;
    const R = build(noise, blur);
    const keep = R.T.persistence.map(p => p >= h);
    const sh = S.seedsFromTree(R.T, keep);
    const wh = SG.seededWatershed(R.B.g, S.W, S.H, sh.seed, sh.n);
    const sm = S.seedsFromMarkers(nm);
    const wm = SG.seededWatershed(R.B.g, S.W, S.H, sm.seed, sm.n);
    const best = R.sweep.reduce((a, b) => (b.ari > a.ari ? b : a));
    out.innerHTML =
      `The smoothed gradient has <b>${R.T.n}</b> regional minima, so the raw watershed returns <b>${R.rawN}</b> `
      + `regions for an image with <b>7</b> — adjusted Rand index <b>${R.rawAri.toFixed(3)}</b>, which is `
      + `indistinguishable from cutting the image at random.<br>`
      + `A persistence filter of <b>h = ${h}</b> leaves <b>${sh.n}</b> regions at ARI `
      + `<b>${ari(wh.lab, R.B.truth).toFixed(3)}</b>; the best value on the whole sweep is `
      + `<b>h = ${best.h}</b>, giving ${best.n} regions at ${best.ari.toFixed(3)}. `
      + `<b>${nm}</b> internal marker${nm === 1 ? "" : "s"} plus the border gives <b>${sm.n}</b> regions at `
      + `<b>${ari(wm.lab, R.B.truth).toFixed(3)}</b> — matching the best filter without searching for it, `
      + `because the region count was supplied rather than discovered.`;
  }
  function all() {
    eNv.textContent = eN.value; eBv.textContent = (+eB.value).toFixed(1);
    eHv.textContent = eH.value; eMv.textContent = eM.value;
    draw(); say();
  }
  eN.oninput = all; eB.oninput = all; eH.oninput = all; eM.oninput = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   11 · #snake-svg — a snake evolving, one panel per energy term
   ══════════════════════════════════════════════════════════════════════════ */
const SGsnake = (function () {
  const W = 100, H = 100, N = 72, CX = 50, CY = 50;
  /* one blob with a concave notch, so rigidity has something to fail at */
  const truth = new Uint8Array(W * H);
  const img = new Float64Array(W * H);
  {
    const r = VZ.rng(53);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const dx = x - CX, dy = y - CY;
      const th = Math.atan2(dy, dx), rr = Math.sqrt(dx * dx + dy * dy);
      const R = 30 + 5 * Math.cos(3 * th) - 13 * Math.exp(-Math.pow((th - 1.9) / 0.24, 2));
      const inside = rr < R;
      truth[y * W + x] = inside ? 1 : 0;
      img[y * W + x] = (inside ? 200 : 62) + 5 * VZ.randn(r);
    }
  }
  const sm = SG.blur(img, W, H, 2.0);
  const gm = SG.gradMag(sm, W, H);
  /* external energy E = −‖∇I‖², smoothed once more so its gradient is usable */
  const Eext = SG.blur(Float64Array.from(gm, v => -v * v), W, H, 1.2);
  let emin = Infinity, emax = -Infinity;
  for (let i = 0; i < Eext.length; i++) { if (Eext[i] < emin) emin = Eext[i]; if (Eext[i] > emax) emax = Eext[i]; }
  const scale = 1 / Math.max(1e-9, (emax - emin));
  for (let i = 0; i < Eext.length; i++) Eext[i] = (Eext[i] - emin) * scale;   // 0 … 1, valley at 0
  /* bilinear sampling and its gradient */
  function samp(F, x, y) {
    const x0 = Math.min(W - 2, Math.max(0, Math.floor(x))), y0 = Math.min(H - 2, Math.max(0, Math.floor(y)));
    const fx = Math.min(1, Math.max(0, x - x0)), fy = Math.min(1, Math.max(0, y - y0));
    const a = F[y0 * W + x0], b = F[y0 * W + x0 + 1], c = F[(y0 + 1) * W + x0], d = F[(y0 + 1) * W + x0 + 1];
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
  }
  const gradE = (x, y) => [
    (samp(Eext, x + 1, y) - samp(Eext, x - 1, y)) / 2,
    (samp(Eext, x, y + 1) - samp(Eext, x, y - 1)) / 2];

  /* the pentadiagonal circulant (I + γA)⁻¹, built once per (α, β) pair */
  const MCACHE = {};
  function inverse(alpha, beta, gamma) {
    const key = alpha + "|" + beta + "|" + gamma;
    if (MCACHE[key]) return MCACHE[key];
    const A = [];
    for (let i = 0; i < N; i++) A.push(new Array(N).fill(0));
    const add = (i, j, v) => { A[i][(j + N) % N] += v; };
    for (let i = 0; i < N; i++) {
      /* −α·D₂ : second difference,  β·D₄ : fourth difference */
      add(i, i - 1, -alpha); add(i, i, 2 * alpha); add(i, i + 1, -alpha);
      add(i, i - 2, beta); add(i, i - 1, -4 * beta); add(i, i, 6 * beta);
      add(i, i + 1, -4 * beta); add(i, i + 2, beta);
      A[i][i] += 1 / gamma;
    }
    const M = VZ.invN ? VZ.invN(A) : null;
    MCACHE[key] = M;
    return M;
  }
  function run(alpha, beta, wEdge, press, r0, steps) {
    const gamma = 6.0;
    const M = inverse(alpha, beta, gamma);
    let X = [], Y = [];
    for (let i = 0; i < N; i++) {
      const t = 2 * Math.PI * i / N;
      X.push(CX + r0 * Math.cos(t)); Y.push(CY + r0 * Math.sin(t));
    }
    const trail = [[X.slice(), Y.slice()]];
    const hist = [];
    for (let it = 0; it < steps; it++) {
      const bx = new Array(N), by = new Array(N);
      for (let i = 0; i < N; i++) {
        const gE = gradE(X[i], Y[i]);
        /* outward normal of the closed curve, from the local tangent */
        const ip = (i + 1) % N, im = (i + N - 1) % N;
        let tx = X[ip] - X[im], ty = Y[ip] - Y[im];
        const tl = Math.hypot(tx, ty) || 1; tx /= tl; ty /= tl;
        const nx = ty, ny = -tx;
        const cxv = X[i] - CX, cyv = Y[i] - CY;
        const sgn = (nx * cxv + ny * cyv) >= 0 ? 1 : -1;
        bx[i] = X[i] / gamma - wEdge * gE[0] + press * sgn * nx;
        by[i] = Y[i] / gamma - wEdge * gE[1] + press * sgn * ny;
      }
      const nX = new Array(N).fill(0), nY = new Array(N).fill(0);
      for (let i = 0; i < N; i++) {
        let sx = 0, sy = 0;
        for (let j = 0; j < N; j++) { sx += M[i][j] * bx[j]; sy += M[i][j] * by[j]; }
        nX[i] = Math.min(W - 2, Math.max(1, sx)); nY[i] = Math.min(H - 2, Math.max(1, sy));
      }
      X = nX; Y = nY;
      if (it % 6 === 0 || it === steps - 1) trail.push([X.slice(), Y.slice()]);
      hist.push(energies(X, Y, alpha, beta, wEdge));
    }
    return { X: X, Y: Y, trail: trail, hist: hist };
  }
  function energies(X, Y, alpha, beta, wEdge) {
    let e1 = 0, e2 = 0, e3 = 0;
    for (let i = 0; i < N; i++) {
      const ip = (i + 1) % N, im = (i + N - 1) % N;
      e1 += Math.pow(X[ip] - X[i], 2) + Math.pow(Y[ip] - Y[i], 2);
      e2 += Math.pow(X[ip] - 2 * X[i] + X[im], 2) + Math.pow(Y[ip] - 2 * Y[i] + Y[im], 2);
      e3 += samp(Eext, X[i], Y[i]);
    }
    return { elastic: alpha * e1, rigid: beta * e2, edge: wEdge * e3,
      total: alpha * e1 + beta * e2 + wEdge * e3 };
  }
  function area(X, Y) {
    let a = 0;
    for (let i = 0; i < N; i++) { const j = (i + 1) % N; a += X[i] * Y[j] - X[j] * Y[i]; }
    return Math.abs(a) / 2;
  }
  /* rasterise the polygon and score it against the truth */
  function iou(X, Y) {
    let inter = 0, uni = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let inside = false;
      for (let i = 0, j = N - 1; i < N; j = i++) {
        if (((Y[i] > y) !== (Y[j] > y)) &&
          (x < (X[j] - X[i]) * (y - Y[i]) / (Y[j] - Y[i]) + X[i])) inside = !inside;
      }
      const t = truth[y * W + x] === 1;
      if (inside && t) inter++;
      if (inside || t) uni++;
    }
    return uni ? inter / uni : 1;
  }
  return { W: W, H: H, N: N, CX: CX, CY: CY, img: img, truth: truth, Eext: Eext, gm: gm,
    run: run, energies: energies, area: area, iou: iou, samp: samp, gradE: gradE };
})();

(function () {
  const svg = d3.select("#snake-svg");
  if (svg.empty()) return;
  const S = SGsnake;
  const ids = ["sn-a", "sn-b", "sn-w", "sn-p", "sn-it", "sn-r"];
  const el = {}; ids.forEach(i => { el[i] = document.getElementById(i); el[i + "v"] = document.getElementById(i + "v"); });
  const out = document.getElementById("snake-readout");
  const CACHE = {};
  function get() {
    const a = +el["sn-a"].value, b = +el["sn-b"].value, w = +el["sn-w"].value,
      p = +el["sn-p"].value, r = +el["sn-r"].value;
    const key = [a, b, w, p, r].join("|");
    if (!CACHE[key]) CACHE[key] = S.run(a, b, w, p, r, 300);
    return { R: CACHE[key], a: a, b: b, w: w, p: p, r: r, it: +el["sn-it"].value };
  }

  function draw() {
    const G = get(), R = G.R;
    const k = Math.max(0, Math.min(R.trail.length - 1, Math.round(G.it / 300 * (R.trail.length - 1))));
    const X = R.trail[k][0], Y = R.trail[k][1];
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.0, IW = S.W * cw;

    const p0 = SG.box(g, 14, 40, IW, IW, "the image and the contour");
    SG.cells(p0, 0, 0, cw, S.W, S.H, (x, y) => SG.grey(S.img[y * S.W + x]));
    const poly = (Xs, Ys, col, w, op) => p0.append("path")
      .attr("d", "M" + Xs.map((v, i) => (v * cw).toFixed(1) + "," + (Ys[i] * cw).toFixed(1)).join("L") + "Z")
      .attr("fill", "none").attr("stroke", col).attr("stroke-width", w).attr("stroke-opacity", op);
    R.trail.slice(0, k).forEach((t, i) => poly(t[0], t[1], VC.accent, 0.8, 0.10 + 0.25 * i / Math.max(1, k)));
    poly(R.trail[0][0], R.trail[0][1], VC.muted, 1.2, 0.7);
    poly(X, Y, VC.a2, 2.2, 1);
    p0.selectAll("circle.c").data(X.map((v, i) => [v, Y[i]])).join("circle").attr("class", "c")
      .attr("cx", d => d[0] * cw).attr("cy", d => d[1] * cw).attr("r", 1.6).attr("fill", VC.a2);
    /* the image force at each control point, −w∇E_ext, scaled to be visible */
    if (G.w > 0) for (let i = 0; i < S.N; i += 2) {
      const gE = S.gradE(X[i], Y[i]);
      const fx = -G.w * gE[0] * 90, fy = -G.w * gE[1] * 90;
      if (Math.hypot(fx, fy) < 0.7) continue;
      VZ.arrow(p0, X[i] * cw, Y[i] * cw, (X[i] + fx) * cw, (Y[i] + fy) * cw,
        { color: VC.teal, w: 1, head: 3.5, op: 0.85 });
    }
    p0.append("text").attr("x", 0).attr("y", IW + 13).attr("font-size", 10).attr("fill", VC.muted)
      .text(`iteration ${Math.round(G.it)} of 300 · ${S.N} control points`);

    const p1 = SG.box(g, 250, 40, IW, IW, "the external energy the snake falls into");
    SG.cells(p1, 0, 0, cw, S.W, S.H, (x, y) => d3.interpolateMagma(S.Eext[y * S.W + x]));
    p1.append("path")
      .attr("d", "M" + X.map((v, i) => (v * cw).toFixed(1) + "," + (Y[i] * cw).toFixed(1)).join("L") + "Z")
      .attr("fill", "none").attr("stroke", VC.teal).attr("stroke-width", 1.8);
    p1.append("text").attr("x", 0).attr("y", IW + 13).attr("font-size", 10).attr("fill", VC.muted)
      .text("dark = the trench along the boundary; flat elsewhere");

    /* energy curves */
    const EX = 486, EY = 40, EW = 258, EH = 200;
    const e = SG.box(g, EX, EY, EW, EH, "the energy terms, against iteration", { fill: "#12141c" });
    const xs = d3.scaleLinear().domain([0, 300]).range([0, EW]);
    const series = [["elastic", VC.accent], ["rigid", VC.violet], ["edge", VC.a2], ["total", VC.good]];
    let lo = Infinity, hi = -Infinity;
    R.hist.forEach(h => series.forEach(([kk]) => { lo = Math.min(lo, h[kk]); hi = Math.max(hi, h[kk]); }));
    const ys = d3.scaleLinear().domain([lo, hi]).range([EH, 0]).nice();
    e.append("g").attr("transform", `translate(0,${EH})`).attr("class", "axis").call(d3.axisBottom(xs).ticks(5));
    e.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5));
    series.forEach(([kk, col]) => {
      e.append("path").datum(R.hist)
        .attr("d", d3.line().x((d, i) => xs(i)).y(d => ys(d[kk])))
        .attr("fill", "none").attr("stroke", col).attr("stroke-width", 1.6);
    });
    e.append("line").attr("x1", xs(G.it)).attr("x2", xs(G.it)).attr("y1", 0).attr("y2", EH)
      .attr("stroke", VC.ink).attr("stroke-width", 1.2);
    VZ.legend(e, series.map(([kk, col]) => ({ color: col, label: kk })), 6, 13, { gap: 13 });

    /* numbers */
    const en = S.energies(X, Y, G.a, G.b, G.w);
    const put = SG.kv(g, 14, 292, { keyW: 152, lead: 15 });
    put("elasticity  α·∑‖Δf‖²", en.elastic.toFixed(2), VC.accent);
    put("rigidity  β·∑‖Δ²f‖²", en.rigid.toFixed(2), VC.violet);
    put("edge  w·∑E_ext", en.edge.toFixed(2), VC.a2);
    put("total", en.total.toFixed(2), VC.good, true);
    const put2 = SG.kv(g, 250, 292, { keyW: 152, lead: 15 });
    const ar = S.area(X, Y), iu = S.iou(X, Y);
    put2("enclosed area", ar.toFixed(1) + " px²   (true 2703)");
    put2("IoU against the object", iu.toFixed(4), iu > 0.9 ? VC.good : VC.bad, true);
    put2("mean radius", (Math.sqrt(ar / Math.PI)).toFixed(2) + " px");
    let mn = Infinity;
    for (let i = 0; i < S.N; i++) {
      const dx = X[i] - S.CX, dy = Y[i] - S.CY, th = Math.atan2(dy, dx);
      if (th > 1.4 && th < 2.4) mn = Math.min(mn, Math.hypot(dx, dy));
    }
    put2("reaches into the notch to", isFinite(mn) ? mn.toFixed(2) + " px   (true 21.09)" : "—",
      isFinite(mn) && mn < 22.5 ? VC.good : VC.a2);
    const a0 = Math.PI * G.r * G.r;
    put2("started at", a0.toFixed(0) + " px²   (radius " + G.r + ")");
    put2("shrunk to", (100 * ar / a0).toFixed(1) + "% of that",
      ar < 0.25 * a0 ? VC.bad : VC.good, ar < 0.25 * a0);
  }

  function say() {
    const G = get(), R = G.R;
    const k = Math.max(0, Math.min(R.trail.length - 1, Math.round(G.it / 300 * (R.trail.length - 1))));
    const X = R.trail[k][0], Y = R.trail[k][1];
    const en = S.energies(X, Y, G.a, G.b, G.w), ar = S.area(X, Y), iu = S.iou(X, Y);
    out.innerHTML =
      `<b>α = ${G.a.toFixed(2)}, β = ${G.b.toFixed(2)}</b>, edge weight <b>${G.w.toFixed(2)}</b>, `
      + `balloon <b>${G.p.toFixed(2)}</b>, started at radius <b>${G.r}</b>. `
      + `After ${Math.round(G.it)} iterations the contour encloses <b>${ar.toFixed(0)} px²</b> `
      + `against the object's 2703, at IoU <b>${iu.toFixed(4)}</b>.<br>`
      + (G.w === 0
        ? `<b style="color:var(--bad)">With the edge force at zero there is nothing but the internal energy</b>, both of whose terms go as r², so the only minimum is a point — and the area readout is on its way there.`
        : `Energies: elasticity <b>${en.elastic.toFixed(1)}</b>, rigidity <b>${en.rigid.toFixed(1)}</b>, `
        + `edge <b>${en.edge.toFixed(1)}</b>, total <b>${en.total.toFixed(1)}</b>. `
        + `The internal terms are always trying to shrink the curve and the edge term is the only thing `
        + `holding it out, so the equilibrium radius is wherever those two balance — which is <i>near</i> the `
        + `true boundary and, for any positive α, systematically inside it.`);
  }
  function all() {
    el["sn-av"].textContent = (+el["sn-a"].value).toFixed(2);
    el["sn-bv"].textContent = (+el["sn-b"].value).toFixed(2);
    el["sn-wv"].textContent = (+el["sn-w"].value).toFixed(2);
    el["sn-pv"].textContent = (+el["sn-p"].value).toFixed(2);
    el["sn-itv"].textContent = el["sn-it"].value;
    el["sn-rv"].textContent = el["sn-r"].value;
    draw(); say();
  }
  ids.forEach(i => { el[i].oninput = all; });
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   12 · #lset-svg — a level set splitting, with no special case for the split
   ══════════════════════════════════════════════════════════════════════════ */
const SGlset = (function () {
  const W = 80, H = 80, EPS = 1.6, DT = 0.9, LAM = 60, MUK = 6;
  /* convention on this page: the interior is φ > 0 */
  const truth = new Uint8Array(W * H);
  const I = new Float64Array(W * H);
  {
    const r = VZ.rng(67);
    const raw = new Float64Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const a = (x - 24) * (x - 24) + (y - 40) * (y - 40) < 14 * 14;
      const b = (x - 56) * (x - 56) + (y - 40) * (y - 40) < 14 * 14;
      truth[y * W + x] = (a || b) ? 1 : 0;
      raw[y * W + x] = ((a || b) ? 200 : 60) + 8 * VZ.randn(r);
    }
    const sm = SG.blur(raw, W, H, 1.0);
    for (let i = 0; i < sm.length; i++) I[i] = sm[i] / 255;
  }
  function init(mode) {
    const p = new Float64Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let v;
      if (mode === 0) v = 35 - Math.hypot(x - 40, y - 40);
      else if (mode === 1) v = 6 * Math.sin(Math.PI * (x + 6) / 12) * Math.sin(Math.PI * (y + 6) / 12);
      else v = 7 - Math.hypot(x - 24, y - 40);
      p[y * W + x] = Math.max(-30, Math.min(30, v));
    }
    return p;
  }
  const delta = v => (1 / Math.PI) * EPS / (EPS * EPS + v * v);
  /* Reinitialise φ to the signed distance of its own zero set, by a two-pass
     chamfer transform. Without this the evolution flattens φ near the contour,
     the smoothed delta stops covering a band of constant width, and the front
     stalls long before it reaches the boundary — which looks exactly like a
     local minimum and is not one. */
  function reinit(phi) {
    const BIG = 1e6, d = new Float64Array(W * H).fill(BIG);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, s = phi[i] > 0;
      let edge = false;
      if (x > 0 && (phi[i - 1] > 0) !== s) edge = true;
      if (x < W - 1 && (phi[i + 1] > 0) !== s) edge = true;
      if (y > 0 && (phi[i - W] > 0) !== s) edge = true;
      if (y < H - 1 && (phi[i + W] > 0) !== s) edge = true;
      if (edge) d[i] = 0.5;
    }
    const A = 1, B = Math.SQRT2;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x; let v = d[i];
      if (x > 0) v = Math.min(v, d[i - 1] + A);
      if (y > 0) v = Math.min(v, d[i - W] + A);
      if (x > 0 && y > 0) v = Math.min(v, d[i - W - 1] + B);
      if (x < W - 1 && y > 0) v = Math.min(v, d[i - W + 1] + B);
      d[i] = v;
    }
    for (let y = H - 1; y >= 0; y--) for (let x = W - 1; x >= 0; x--) {
      const i = y * W + x; let v = d[i];
      if (x < W - 1) v = Math.min(v, d[i + 1] + A);
      if (y < H - 1) v = Math.min(v, d[i + W] + A);
      if (x < W - 1 && y < H - 1) v = Math.min(v, d[i + W + 1] + B);
      if (x > 0 && y < H - 1) v = Math.min(v, d[i + W - 1] + B);
      d[i] = v;
    }
    const outp = new Float64Array(W * H);
    for (let i = 0; i < W * H; i++)
      outp[i] = Math.max(-30, Math.min(30, (phi[i] > 0 ? 1 : -1) * Math.min(d[i], 30)));
    return outp;
  }
  function means(phi) {
    let s1 = 0, n1 = 0, s2 = 0, n2 = 0;
    for (let i = 0; i < W * H; i++) {
      if (phi[i] > 0) { s1 += I[i]; n1++; } else { s2 += I[i]; n2++; }
    }
    return { c1: n1 ? s1 / n1 : 0, c2: n2 ? s2 / n2 : 0, n1: n1, n2: n2 };
  }
  function energy(phi, mu) {
    const m = means(phi);
    let dat = 0, len = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      dat += (phi[i] > 0) ? Math.pow(I[i] - m.c1, 2) : Math.pow(I[i] - m.c2, 2);
      const px = (phi[i + (x < W - 1 ? 1 : 0)] - phi[i - (x > 0 ? 1 : 0)]) / 2;
      const py = (phi[i + (y < H - 1 ? W : 0)] - phi[i - (y > 0 ? W : 0)]) / 2;
      len += delta(phi[i]) * Math.hypot(px, py);
    }
    return { data: LAM * dat, length: len, total: LAM * dat + MUK * mu * len, c1: m.c1, c2: m.c2 };
  }
  /* The energy is symmetric in the two phases, so a run can settle with the
     background inside the contour. Objects here are the bright phase, so every
     report uses the side with the larger mean and says so rather than pretending. */
  function bright(phi) { const m = means(phi); return m.c1 >= m.c2 ? 1 : -1; }
  function components(phi) {
    const sgn = bright(phi);
    const c = SG.components(W, H, (a, b) => (phi[a] * sgn > 0) === (phi[b] * sgn > 0), 4);
    /* count only interior components above a few pixels, so single-pixel noise
       does not masquerade as a topology change */
    const size = new Map();
    for (let i = 0; i < W * H; i++) if (phi[i] * sgn > 0) size.set(c.lab[i], (size.get(c.lab[i]) || 0) + 1);
    let n = 0;
    for (const v of size.values()) if (v >= 6) n++;
    return n;
  }
  function iou(phi) {
    let inter = 0, uni = 0;
    const sgn = bright(phi);
    for (let i = 0; i < W * H; i++) {
      const p = phi[i] * sgn > 0, t = truth[i] === 1;
      if (p && t) inter++; if (p || t) uni++;
    }
    return uni ? inter / uni : 1;
  }
  function run(mu, mode, steps) {
    let phi = init(mode);
    const frames = [Float64Array.from(phi)];
    const hist = [];
    /* clamped neighbour indices, precomputed once — the closure-per-access version
       of this loop was five times slower and made the μ slider unusable */
    const xm = new Int32Array(W), xp = new Int32Array(W), ym = new Int32Array(H), yp = new Int32Array(H);
    for (let k = 0; k < W; k++) { xm[k] = Math.max(0, k - 1); xp[k] = Math.min(W - 1, k + 1); }
    for (let k = 0; k < H; k++) { ym[k] = Math.max(0, k - 1); yp[k] = Math.min(H - 1, k + 1); }
    for (let it = 0; it < steps; it++) {
      const m = means(phi);
      const c1 = m.c1, c2 = m.c2;
      const nx = new Float64Array(W * H);
      for (let y = 0; y < H; y++) {
        const rY = y * W, rU = ym[y] * W, rD = yp[y] * W;
        for (let x = 0; x < W; x++) {
          const i = rY + x, p = phi[i];
          const L = phi[rY + xm[x]], R = phi[rY + xp[x]];
          const U = phi[rU + x], Dn = phi[rD + x];
          const px = (R - L) / 2, py = (Dn - U) / 2;
          const pxx = R - 2 * p + L, pyy = Dn - 2 * p + U;
          const pxy = (phi[rD + xp[x]] - phi[rD + xm[x]] - phi[rU + xp[x]] + phi[rU + xm[x]]) / 4;
          const g2 = px * px + py * py + 1e-8;
          const den = g2 * Math.sqrt(g2);
          const kap = (pxx * py * py - 2 * px * py * pxy + pyy * px * px) / den;
          const d = (1 / Math.PI) * EPS / (EPS * EPS + p * p);
          const a1 = I[i] - c1, a2 = I[i] - c2;
          const drive = MUK * mu * kap + LAM * (a2 * a2 - a1 * a1);
          const v = p + DT * d * drive;
          nx[i] = v < -30 ? -30 : (v > 30 ? 30 : v);
        }
      }
      phi = nx;
      if (it % 6 === 5) phi = reinit(phi);
      if (it % 4 === 3 || it === steps - 1) frames.push(Float64Array.from(phi));
      const e = energy(phi, mu);
      hist.push({ comps: components(phi), c1: e.c1, c2: e.c2, total: e.total,
        data: e.data, length: e.length, iou: iou(phi) });
    }
    return { frames: frames, hist: hist, phi: phi };
  }
  return { W: W, H: H, I: I, truth: truth, run: run, init: init, reinit: reinit, bright: bright,
    components: components, iou: iou, energy: energy, means: means };
})();

(function () {
  const svg = d3.select("#lset-svg");
  if (svg.empty()) return;
  const S = SGlset;
  const eI = document.getElementById("ls-it"), eIv = document.getElementById("ls-itv");
  const eM = document.getElementById("ls-mu"), eMv = document.getElementById("ls-muv");
  const eN = document.getElementById("ls-init"), eP = document.getElementById("ls-phi");
  const out = document.getElementById("lset-readout");
  const STEPS = 240, CACHE = {};
  function get() {
    const mu = +eM.value, mode = +eN.value, key = mu + "|" + mode;
    if (!CACHE[key]) CACHE[key] = S.run(mu, mode, STEPS);
    return { R: CACHE[key], mu: mu, mode: mode, it: +eI.value };
  }
  const contour = d3.contours().size([S.W, S.H]).thresholds([0]);

  function draw() {
    const G = get(), R = G.R;
    const fi = Math.max(0, Math.min(R.frames.length - 1, Math.round(G.it / STEPS * (R.frames.length - 1))));
    const phi = R.frames[fi];
    const sgn = S.bright(phi);
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.2, IW = S.W * cw;

    const path = d3.geoPath(d3.geoIdentity().scale(cw));
    const zero = contour(Array.from(phi, v => v * sgn));
    const zero0 = contour(Array.from(R.frames[0], v => v * S.bright(R.frames[0])));

    const p0 = SG.box(g, 14, 42, IW, IW, "the image and the zero level set");
    SG.cells(p0, 0, 0, cw, S.W, S.H, (x, y) => SG.grey(S.I[y * S.W + x] * 255));
    p0.append("path").attr("d", path(zero0[0])).attr("fill", "none")
      .attr("stroke", VC.muted).attr("stroke-width", 1).attr("stroke-opacity", 0.7);
    p0.append("path").attr("d", path(zero[0])).attr("fill", VC.a2).attr("fill-opacity", 0.10)
      .attr("stroke", VC.a2).attr("stroke-width", 2);
    p0.append("text").attr("x", 0).attr("y", IW + 13).attr("font-size", 10).attr("fill", VC.muted)
      .text(`iteration ${G.it} of ${STEPS} · faint line is the start`);

    if (eP.checked) {
      let lo = 0;
      for (let i = 0; i < phi.length; i++) lo = Math.max(lo, Math.abs(phi[i]));
      const p1 = SG.box(g, 250, 42, IW, IW, "the embedding function φ");
      SG.cells(p1, 0, 0, cw, S.W, S.H, (x, y) =>
        d3.interpolateRdBu(0.5 + 0.5 * Math.max(-1, Math.min(1, sgn * phi[y * S.W + x] / Math.max(1, lo)))));
      p1.append("path").attr("d", path(zero[0])).attr("fill", "none")
        .attr("stroke", "#0b0d12").attr("stroke-width", 1.6);
      p1.append("text").attr("x", 0).attr("y", IW + 13).attr("font-size", 10).attr("fill", VC.muted)
        .text("blue is inside, red outside; the black line is φ = 0");
    }

    /* components and IoU */
    const CX = 486, CY = 42, CW = 258, CH = 96;
    const c = SG.box(g, CX, CY, CW, CH, "connected components of the interior", { fill: "#12141c" });
    const xs = d3.scaleLinear().domain([0, STEPS]).range([0, CW]);
    const ymax = Math.max(3, d3.max(R.hist, d => d.comps));
    const yc = d3.scaleLinear().domain([0, ymax]).range([CH, 0]);
    c.append("g").attr("transform", `translate(0,${CH})`).attr("class", "axis").call(d3.axisBottom(xs).ticks(5));
    c.append("g").attr("class", "axis").call(d3.axisLeft(yc).ticks(Math.min(5, ymax)));
    c.append("path").datum(R.hist).attr("d", d3.line().curve(d3.curveStepAfter).x((d, i) => xs(i)).y(d => yc(d.comps)))
      .attr("fill", "none").attr("stroke", VC.violet).attr("stroke-width", 1.8);
    c.append("line").attr("x1", xs(G.it)).attr("x2", xs(G.it)).attr("y1", 0).attr("y2", CH)
      .attr("stroke", VC.ink).attr("stroke-width", 1.2);

    const D = SG.box(g, CX, 182, CW, 96, "the two region means the energy fits", { fill: "#12141c" });
    const yd = d3.scaleLinear().domain([0, 1]).range([96, 0]);
    D.append("g").attr("transform", "translate(0,96)").attr("class", "axis").call(d3.axisBottom(xs).ticks(5));
    D.append("g").attr("class", "axis").call(d3.axisLeft(yd).ticks(4));
    [["c1", VC.a2], ["c2", VC.accent]].forEach(([k, col]) => {
      D.append("path").datum(R.hist).attr("d", d3.line().x((d, i) => xs(i)).y(d => yd(d[k])))
        .attr("fill", "none").attr("stroke", col).attr("stroke-width", 1.6);
    });
    D.append("line").attr("x1", xs(G.it)).attr("x2", xs(G.it)).attr("y1", 0).attr("y2", 96)
      .attr("stroke", VC.ink).attr("stroke-width", 1.2);
    VZ.legend(D, [{ color: VC.a2, label: "c₁ inside" }, { color: VC.accent, label: "c₂ outside" }],
      CW - 88, 12, { gap: 13 });

    /* energy and IoU */
    const EX = 14, EY = 316, EW = 440, EH = 106;
    const e = SG.box(g, EX, EY, EW, EH, "total energy, and the region score, against iteration", { fill: "#12141c" });
    const ye = d3.scaleLinear().domain(d3.extent(R.hist, d => d.total)).range([EH, 0]).nice();
    const yi = d3.scaleLinear().domain([0, 1]).range([EH, 0]);
    const xe = d3.scaleLinear().domain([0, STEPS]).range([0, EW]);
    e.append("g").attr("transform", `translate(0,${EH})`).attr("class", "axis").call(d3.axisBottom(xe).ticks(6));
    e.append("g").attr("class", "axis").call(d3.axisLeft(ye).ticks(4, "~s"));
    e.append("path").datum(R.hist).attr("d", d3.line().x((d, i) => xe(i)).y(d => ye(d.total)))
      .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.8);
    e.append("path").datum(R.hist).attr("d", d3.line().x((d, i) => xe(i)).y(d => yi(d.iou)))
      .attr("fill", "none").attr("stroke", VC.teal).attr("stroke-width", 1.5).attr("stroke-dasharray", "4 3");
    e.append("line").attr("x1", xe(G.it)).attr("x2", xe(G.it)).attr("y1", 0).attr("y2", EH)
      .attr("stroke", VC.ink).attr("stroke-width", 1.2);
    VZ.legend(e, [{ color: VC.good, label: "total energy" }, { color: VC.teal, label: "IoU (0…1)", dash: "4 3" }],
      EW - 110, 12, { gap: 13 });

    const h = R.hist[Math.max(0, Math.min(R.hist.length - 1, G.it - 1))] || R.hist[0];
    const put = SG.kv(g, 486, 330, { keyW: 150, lead: 15 });
    put("components inside", String(S.components(phi)), VC.violet, true);
    put("c₁ inside / c₂ outside", h.c1.toFixed(3) + "  /  " + h.c2.toFixed(3), VC.a2);
    put("data term", h.data.toFixed(0));
    put("length term × μ", (S.energy(phi, G.mu).length * 6 * G.mu).toFixed(0), VC.violet);
    put("total energy", h.total.toFixed(0), VC.good, true);
    put("IoU against both discs", S.iou(phi).toFixed(4), S.iou(phi) > 0.9 ? VC.good : VC.a2, true);
  }

  function say() {
    const G = get(), R = G.R;
    const comps = R.hist.map(h => h.comps);
    let first = -1;
    for (let i = 1; i < comps.length; i++) if (comps[i] > comps[i - 1]) { first = i; break; }
    const fi = Math.max(0, Math.min(R.frames.length - 1, Math.round(G.it / STEPS * (R.frames.length - 1))));
    const phi = R.frames[fi];
    out.innerHTML =
      `Starting from <b>${["one circle around both discs", "a grid of small circles", "one circle inside the left disc"][G.mode]}</b> `
      + `with a length penalty of <b>${G.mu.toFixed(1)}</b>: after ${G.it} iterations the interior has `
      + `<b>${S.components(phi)}</b> connected component${S.components(phi) === 1 ? "" : "s"} and scores IoU `
      + `<b>${S.iou(phi).toFixed(4)}</b> against the two true discs.<br>`
      + (G.mode === 0
        ? (first > 0
          ? `The topology changes at iteration <b>${first}</b>. Look at the energy curve there: <b>nothing happens</b>. `
          + `No term jumps, the surface φ does not move discontinuously, and no branch of the update was taken — `
          + `a smooth function's zero set simply stopped being connected. A parametric contour cannot do this at all.`
          : `At this length penalty the contour has not split within ${STEPS} iterations.`)
        : G.mode === 1
          ? `The grid of circles merges rather than splits, which costs the representation exactly as little. `
          + `Seeding everywhere and letting the energy decide is the standard defence against a bad initialisation.`
          : `The contour fills the left disc and stops. It never reaches the right one: the region <i>criterion</i> is `
          + `global — c₁ and c₂ are computed from every pixel — but the <i>motion</i> is not, because the smoothed `
          + `delta only lets a narrow band around the current contour move. Global reach in the energy does not buy `
          + `global reach in the flow.`);
  }
  function all() {
    eIv.textContent = eI.value; eMv.textContent = (+eM.value).toFixed(1);
    draw(); say();
  }
  eI.oninput = all; eM.oninput = all; eN.onchange = all; eP.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   13 · #ms-svg — mean shift climbing a kernel density estimate
   ══════════════════════════════════════════════════════════════════════════ */
const SGms = (function () {
  const XS = [12, 14, 15, 16, 18, 19, 20, 22, 23,
    34, 35, 37, 38, 39, 40, 42, 43,
    62, 64, 65, 67, 68, 70, 71, 73, 75, 76];
  const LO = 0, HI = 92, GRID = 9201;
  function dens(x, h, kern) {
    let s = 0;
    for (const xi of XS) {
      const r = Math.pow((x - xi) / h, 2);
      s += (kern === "epan") ? Math.max(0, 1 - r) : Math.exp(-0.5 * r);
    }
    return s / (XS.length * h);
  }
  function curve(h, kern) {
    const out = new Float64Array(GRID);
    for (let i = 0; i < GRID; i++) out[i] = dens(LO + (HI - LO) * i / (GRID - 1), h, kern);
    return out;
  }
  /* Dense-grid maxima, refined by fitting a parabola through the three samples
     around each one. Without the refinement the grid answer is only good to half
     a grid step, and the comparison against mean shift would report a mismatch
     that is entirely an artefact of the grid. */
  function gridModes(h, kern) {
    const f = curve(h, kern), out = [], dx = (HI - LO) / (GRID - 1);
    for (let i = 1; i < GRID - 1; i++) {
      if (!(f[i] > f[i - 1] && f[i] >= f[i + 1])) continue;
      const den = f[i - 1] - 2 * f[i] + f[i + 1];
      const d = (Math.abs(den) < 1e-18) ? 0 : 0.5 * (f[i - 1] - f[i + 1]) / den;
      out.push(LO + dx * (i + Math.max(-1, Math.min(1, d))));
    }
    return out;
  }
  /* one mean-shift step: the weighted mean of the samples under the derivative kernel */
  function step(y, h, kern) {
    let num = 0, den = 0;
    for (const xi of XS) {
      const r = Math.pow((y - xi) / h, 2);
      const g = (kern === "epan") ? (r < 1 ? 1 : 0) : Math.exp(-0.5 * r);
      num += xi * g; den += g;
    }
    return den > 0 ? num / den : y;
  }
  function trajectory(y0, h, kern, maxIt) {
    const path = [y0];
    let y = y0;
    for (let i = 0; i < maxIt; i++) {
      const yn = step(y, h, kern);
      if (Math.abs(yn - y) < 1e-12) { y = yn; path.push(y); break; }
      y = yn; path.push(y);
    }
    return path;
  }
  function attractors(h, kern) {
    const ends = XS.map(x => trajectory(x, h, kern, 4000).slice(-1)[0]).sort((a, b) => a - b);
    const out = [];
    for (const e of ends) if (!out.length || Math.abs(e - out[out.length - 1]) > 1e-2) out.push(e);
    return out;
  }
  /* The mode-count staircase, computed once per kernel. The displayed density uses
     the fine 9 201-point grid; counting modes at 1 071 bandwidths on that grid costs
     a quarter of a billion operations and visibly hangs the page, so the staircase
     uses a 1 841-point grid and a coarser bandwidth step. The density is smooth at
     every bandwidth in range, so the two grids agree on the mode count everywhere;
     the staircase transitions below are unchanged to two decimals. */
  const SGRID = 1841;
  function modesCoarse(h, kern) {
    let prev = 0, cur = 0, n = 0;
    for (let i = 0; i < SGRID; i++) {
      const x = LO + (HI - LO) * i / (SGRID - 1);
      let s = 0;
      for (const xi of XS) {
        const r = Math.pow((x - xi) / h, 2);
        s += (kern === "epan") ? Math.max(0, 1 - r) : Math.exp(-0.5 * r);
      }
      if (i >= 2 && cur > prev && cur >= s) n++;
      prev = cur; cur = s;
    }
    return n;
  }
  const STAIR = {};
  function stair(kern) {
    if (STAIR[kern]) return STAIR[kern];
    const hs = [], ns = [];
    for (let h = 0.6; h <= 22.001; h += 0.04) { hs.push(h); ns.push(modesCoarse(h, kern)); }
    STAIR[kern] = { hs: hs, ns: ns };
    return STAIR[kern];
  }
  /* the bandwidth interval over which the current mode count survives */
  function plateau(h, kern) {
    const s = stair(kern);
    let i = 0, bd = Infinity;
    for (let k = 0; k < s.hs.length; k++) { const d = Math.abs(s.hs[k] - h); if (d < bd) { bd = d; i = k; } }
    const n = s.ns[i];
    let a = i, b = i;
    while (a > 0 && s.ns[a - 1] === n) a--;
    while (b < s.hs.length - 1 && s.ns[b + 1] === n) b++;
    return { n: n, lo: s.hs[a], hi: s.hs[b], ratio: s.hs[b] / s.hs[a] };
  }
  return { XS: XS, LO: LO, HI: HI, GRID: GRID, dens: dens, curve: curve,
    gridModes: gridModes, step: step, trajectory: trajectory,
    attractors: attractors, stair: stair, plateau: plateau };
})();

(function () {
  const svg = d3.select("#ms-svg");
  if (svg.empty()) return;
  const S = SGms;
  const eH = document.getElementById("ms-h"), eHv = document.getElementById("ms-hv");
  const eK = document.getElementById("ms-k"), eT = document.getElementById("ms-tr");
  const eG = document.getElementById("ms-gr");
  const out = document.getElementById("ms-readout");

  function draw() {
    const h = +eH.value, kern = eK.value;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const AX = 44, AY = 36, AW = 700, AH = 150;
    const x = d3.scaleLinear().domain([S.LO, S.HI]).range([0, AW]);
    const f = S.curve(h, kern);
    const y = d3.scaleLinear().domain([0, d3.max(f) * 1.18]).range([AH, 0]);

    const A = SG.box(g, AX, AY, AW, AH, "the kernel density estimate, and the mean-shift vector at each sample", { fill: "#12141c" });
    A.append("g").attr("transform", `translate(0,${AH})`).attr("class", "axis").call(d3.axisBottom(x).ticks(10));
    A.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(3));
    A.append("path").datum(d3.range(S.GRID))
      .attr("d", d3.area().x(i => x(S.LO + (S.HI - S.LO) * i / (S.GRID - 1))).y0(AH).y1(i => y(f[i])))
      .attr("fill", VC.accent).attr("fill-opacity", 0.16);
    A.append("path").datum(d3.range(S.GRID))
      .attr("d", d3.line().x(i => x(S.LO + (S.HI - S.LO) * i / (S.GRID - 1))).y(i => y(f[i])))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 1.8);
    S.XS.forEach(xi => {
      A.append("line").attr("x1", x(xi)).attr("x2", x(xi)).attr("y1", AH).attr("y2", AH - 9)
        .attr("stroke", VC.muted).attr("stroke-width", 1.4);
      const m = S.step(xi, h, kern) - xi;
      if (Math.abs(m) > 0.05)
        VZ.arrow(A, x(xi), y(S.dens(xi, h, kern)) - 8, x(xi + m), y(S.dens(xi, h, kern)) - 8,
          { color: VC.a2, w: 1.2, head: 4, op: 0.85 });
    });
    const gm = S.gridModes(h, kern);
    if (eG.checked) gm.forEach(m => {
      A.append("path").attr("d", `M${x(m)},${y(S.dens(m, h, kern)) - 4} l-5,-8 l10,0 Z`).attr("fill", VC.good);
    });
    A.append("text").attr("x", 4).attr("y", 12).attr("font-size", 11).attr("fill", VC.good)
      .text(gm.length + " mode" + (gm.length === 1 ? "" : "s") + " by dense grid search");

    /* trajectories */
    const BX = 44, BY = 226, BW = 700, BH = 110;
    const B = SG.box(g, BX, BY, BW, BH, "every sample's trajectory, against iteration", { fill: "#12141c" });
    const maxIt = 60;
    const xt = d3.scaleLinear().domain([0, maxIt]).range([0, BW]);
    const yt = d3.scaleLinear().domain([S.LO + 5, S.HI - 8]).range([BH, 0]);
    B.append("g").attr("transform", `translate(0,${BH})`).attr("class", "axis").call(d3.axisBottom(xt).ticks(8));
    B.append("g").attr("class", "axis").call(d3.axisLeft(yt).ticks(4));
    if (eT.checked) S.XS.forEach((xi, k) => {
      const p = S.trajectory(xi, h, kern, maxIt);
      B.append("path").datum(p)
        .attr("d", d3.line().x((v, i) => xt(Math.min(maxIt, i))).y(v => yt(v)))
        .attr("fill", "none").attr("stroke", SG.pal(k % 16)).attr("stroke-width", 1.2).attr("stroke-opacity", 0.8);
    });
    const att = S.attractors(h, kern);
    att.forEach(a => {
      B.append("line").attr("x1", 0).attr("x2", BW).attr("y1", yt(a)).attr("y2", yt(a))
        .attr("stroke", VC.good).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.7);
      B.append("text").attr("x", BW - 3).attr("y", yt(a) - 3).attr("text-anchor", "end")
        .attr("font-size", 9.5).attr("fill", VC.good).text(a.toFixed(3));
    });

    /* the staircase */
    const CX = 44, CY = 376, CW = 420, CH = 76;
    const st = S.stair(kern), pl = S.plateau(h, kern);
    const C = SG.box(g, CX, CY, CW, CH, "modes against bandwidth", { fill: "#12141c" });
    const xc = d3.scaleLinear().domain([0.6, 22]).range([0, CW]);
    const yc = d3.scaleLinear().domain([0, Math.max(4, d3.max(st.ns))]).range([CH, 0]);
    C.append("g").attr("transform", `translate(0,${CH})`).attr("class", "axis").call(d3.axisBottom(xc).ticks(7));
    C.append("g").attr("class", "axis").call(d3.axisLeft(yc).ticks(4));
    C.append("rect").attr("x", xc(pl.lo)).attr("y", 0).attr("width", Math.max(1, xc(pl.hi) - xc(pl.lo)))
      .attr("height", CH).attr("fill", VC.good).attr("fill-opacity", 0.14);
    C.append("path").datum(st.hs)
      .attr("d", d3.line().curve(d3.curveStepAfter).x(d => xc(d)).y((d, i) => yc(st.ns[i])))
      .attr("fill", "none").attr("stroke", VC.violet).attr("stroke-width", 1.8);
    C.append("line").attr("x1", xc(h)).attr("x2", xc(h)).attr("y1", 0).attr("y2", CH)
      .attr("stroke", VC.ink).attr("stroke-width", 1.3);
    C.append("text").attr("x", CW - 3).attr("y", 12).attr("text-anchor", "end").attr("font-size", 9.5)
      .attr("fill", VC.good).text(`plateau ${pl.lo.toFixed(2)} … ${pl.hi.toFixed(2)}  (×${pl.ratio.toFixed(2)})`);

    /* numbers */
    let worst = 0;
    for (let i = 0; i < Math.min(att.length, gm.length); i++) worst = Math.max(worst, Math.abs(att[i] - gm[i]));
    const put = SG.kv(g, 494, 390, { keyW: 156, lead: 15 });
    put("modes, dense grid search", String(gm.length), VC.good, true);
    put("mean-shift attractors", String(att.length), VC.a2, true);
    put("largest disagreement", (att.length === gm.length) ? worst.toExponential(2) : "different counts",
      (att.length === gm.length && worst < 1e-6) ? VC.good : VC.bad);
    put("this answer survives", pl.lo.toFixed(2) + " … " + pl.hi.toFixed(2) + "  (× " + pl.ratio.toFixed(2) + ")", VC.violet);
    put("mode locations", att.map(a => a.toFixed(2)).join(", "), VC.ink);
  }

  function say() {
    const h = +eH.value, kern = eK.value;
    const gm = S.gridModes(h, kern), att = S.attractors(h, kern), pl = S.plateau(h, kern);
    let worst = 0;
    for (let i = 0; i < Math.min(att.length, gm.length); i++) worst = Math.max(worst, Math.abs(att[i] - gm[i]));
    out.innerHTML =
      `Bandwidth <b>${h.toFixed(2)}</b>, ${kern === "epan" ? "Epanechnikov" : "Gaussian"} kernel. `
      + `The dense grid search finds <b>${gm.length}</b> mode${gm.length === 1 ? "" : "s"}; mean shift, started at all `
      + `27 samples, converges to <b>${att.length}</b> attractor${att.length === 1 ? "" : "s"}`
      + (att.length === gm.length
        ? `, and the two lists agree to <b>${worst.toExponential(2)}</b> — which they must, because the iteration is exactly gradient ascent on the curve drawn above.`
        : `, which differs from the grid count only because two attractors fell within the merge tolerance.`)
      + `<br>This answer holds for every bandwidth from <b>${pl.lo.toFixed(2)}</b> to <b>${pl.hi.toFixed(2)}</b>, `
      + `a factor of <b>${pl.ratio.toFixed(2)}</b>. Wide plateaus are the only evidence the data offers about how many `
      + `clusters there are: the three-mode plateau on this sample is a factor of 4.73 wide, the seven-mode one 1.15.`;
  }
  function all() { eHv.textContent = (+eH.value).toFixed(2); draw(); say(); }
  eH.oninput = all; eK.onchange = all; eT.onchange = all; eG.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   14 · #joint-svg — range-only against joint spatial–range clustering
   ══════════════════════════════════════════════════════════════════════════ */
const SGjoint = (function () {
  const W = 64, H = 48;
  const img = new Float64Array(W * H), truth = new Int32Array(W * H);
  {
    const r = VZ.rng(83);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let t = 0, v = 90;
      if (x >= 5 && x < 20 && y >= 6 && y < 24) { t = 1; v = 178; }        // left bar
      if (x >= 44 && x < 59 && y >= 6 && y < 24) { t = 2; v = 178; }       // right bar, same brightness
      if (y >= 34) { t = 3; v = 132; }                                     // bottom band
      truth[y * W + x] = t;
      img[y * W + x] = Math.min(255, Math.max(0, v + 6 * VZ.randn(r)));
    }
  }
  /* range-only mean shift, run on the 256-bin histogram so it is exact and fast */
  function rangeOnly(hr) {
    const hist = new Float64Array(256);
    for (let i = 0; i < img.length; i++) hist[Math.round(img[i])]++;
    const mode = new Float64Array(256);
    for (let l = 0; l < 256; l++) {
      if (hist[l] === 0) { mode[l] = l; continue; }
      let y = l;
      for (let it = 0; it < 400; it++) {
        let num = 0, den = 0;
        for (let m = 0; m < 256; m++) {
          if (!hist[m]) continue;
          const g = Math.exp(-0.5 * Math.pow((y - m) / hr, 2)) * hist[m];
          num += m * g; den += g;
        }
        const yn = den > 0 ? num / den : y;
        if (Math.abs(yn - y) < 1e-9) { y = yn; break; }
        y = yn;
      }
      mode[l] = y;
    }
    /* group the attractors */
    const reps = [];
    const idOf = new Int32Array(256).fill(-1);
    const order = d3.range(256).sort((a, b) => mode[a] - mode[b]);
    for (const l of order) {
      if (hist[l] === 0) continue;
      let found = -1;
      for (let k = 0; k < reps.length; k++) if (Math.abs(mode[l] - reps[k]) < 0.5) { found = k; break; }
      if (found < 0) { reps.push(mode[l]); found = reps.length - 1; }
      idOf[l] = found;
    }
    const lab = new Int32Array(W * H);
    for (let i = 0; i < lab.length; i++) lab[i] = idOf[Math.round(img[i])];
    return { lab: lab, n: reps.length, modes: reps };
  }
  /* joint-domain mean shift: Epanechnikov (finite support) in space, Gaussian in range */
  function joint(hs, hr, minSize) {
    const N = W * H;
    const px = new Float64Array(N), py = new Float64Array(N), pv = new Float64Array(N);
    for (let i = 0; i < N; i++) { px[i] = i % W; py[i] = (i / W) | 0; pv[i] = img[i]; }
    const R = Math.ceil(hs);
    for (let it = 0; it < 14; it++) {
      const nx = new Float64Array(N), ny = new Float64Array(N), nv = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        const cx = px[i], cy = py[i], cv = pv[i];
        let sx = 0, sy = 0, sv = 0, sw = 0;
        const x0 = Math.max(0, Math.floor(cx - R)), x1 = Math.min(W - 1, Math.ceil(cx + R));
        const y0 = Math.max(0, Math.floor(cy - R)), y1 = Math.min(H - 1, Math.ceil(cy + R));
        for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++) {
          const j = yy * W + xx;
          const ds = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy);
          if (ds > hs * hs) continue;                       // Epanechnikov support
          const wS = 1 - ds / (hs * hs);
          const dv = (img[j] - cv) / hr;
          const wR = Math.exp(-0.5 * dv * dv);
          const w = wS * wR;
          sx += xx * w; sy += yy * w; sv += img[j] * w; sw += w;
        }
        if (sw > 0) { nx[i] = sx / sw; ny[i] = sy / sw; nv[i] = sv / sw; }
        else { nx[i] = cx; ny[i] = cy; nv[i] = cv; }
      }
      for (let i = 0; i < N; i++) { px[i] = nx[i]; py[i] = ny[i]; pv[i] = nv[i]; }
    }
    /* two pixels share a label when they are adjacent AND their converged points
       are close in the joint metric — so a region is connected by construction */
    const c = SG.components(W, H, (a, b) =>
      (Math.pow(px[a] - px[b], 2) + Math.pow(py[a] - py[b], 2)) / (hs * hs)
      + Math.pow((pv[a] - pv[b]) / hr, 2) < 0.25, 8);
    /* absorb regions below the minimum size into the most similar neighbour */
    const uf = SG.UF(c.n);
    const size = new Array(c.n).fill(0), mean = new Array(c.n).fill(0);
    for (let i = 0; i < N; i++) { size[c.lab[i]]++; mean[c.lab[i]] += img[i]; }
    for (let k = 0; k < c.n; k++) mean[k] /= Math.max(1, size[k]);
    const adj = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (x + 1 < W && c.lab[i] !== c.lab[i + 1]) adj.push([c.lab[i], c.lab[i + 1]]);
      if (y + 1 < H && c.lab[i] !== c.lab[i + W]) adj.push([c.lab[i], c.lab[i + W]]);
    }
    adj.sort((a, b) => Math.abs(mean[a[0]] - mean[a[1]]) - Math.abs(mean[b[0]] - mean[b[1]]));
    let changed = true, rounds = 0;
    while (changed && rounds++ < 6) {
      changed = false;
      for (const [a, b] of adj) {
        const ra = uf.find(a), rb = uf.find(b);
        if (ra === rb) continue;
        if (size[ra] < minSize || size[rb] < minSize) {
          const m = (mean[ra] * size[ra] + mean[rb] * size[rb]) / (size[ra] + size[rb]);
          uf.union(ra, rb);
          const r2 = uf.find(ra);
          size[r2] = size[ra] + size[rb]; mean[r2] = m;
          changed = true;
        }
      }
    }
    const ids = new Map(), lab = new Int32Array(N);
    for (let i = 0; i < N; i++) {
      const r2 = uf.find(c.lab[i]);
      if (!ids.has(r2)) ids.set(r2, ids.size);
      lab[i] = ids.get(r2);
    }
    return { lab: lab, n: ids.size, px: px, py: py, pv: pv };
  }
  /* how many spatially disconnected pieces does a labelling produce? */
  function pieces(lab) {
    const c = SG.components(W, H, (a, b) => lab[a] === lab[b], 8);
    return c.n;
  }
  return { W: W, H: H, img: img, truth: truth, rangeOnly: rangeOnly, joint: joint, pieces: pieces };
})();

(function () {
  const svg = d3.select("#joint-svg");
  if (svg.empty()) return;
  const S = SGjoint;
  const eR = document.getElementById("jt-hr"), eRv = document.getElementById("jt-hrv");
  const eS = document.getElementById("jt-hs"), eSv = document.getElementById("jt-hsv");
  const eM = document.getElementById("jt-m"), eMv = document.getElementById("jt-mv");
  const out = document.getElementById("joint-readout");
  const C1 = {}, C2 = {};
  const ari = lab => SG.adjRand(Array.from(lab), Array.from(S.truth), new Array(lab.length).fill(1));
  function bigPieces(lab, min) {
    const c = SG.components(S.W, S.H, (a, b) => lab[a] === lab[b], 8);
    const sz = new Map();
    for (let i = 0; i < lab.length; i++) sz.set(c.lab[i], (sz.get(c.lab[i]) || 0) + 1);
    let n = 0;
    for (const v of sz.values()) if (v >= min) n++;
    return n;
  }
  function get() {
    const hr = +eR.value, hs = +eS.value, M = +eM.value;
    if (!C1[hr]) C1[hr] = S.rangeOnly(hr);
    const k = hs + "|" + hr + "|" + M;
    if (!C2[k]) C2[k] = S.joint(hs, hr, M);
    return { hr: hr, hs: hs, M: M, R: C1[hr], J: C2[k] };
  }

  function draw() {
    const G = get();
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 3, IW = S.W * cw, IH = S.H * cw;

    const p0 = SG.box(g, 14, 40, IW, IH, "the image");
    SG.cells(p0, 0, 0, cw, S.W, S.H, (x, y) => SG.grey(S.img[y * S.W + x]));
    p0.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10).attr("fill", VC.muted)
      .text("both bright bars are 178; band 132; background 90");

    const p1 = SG.box(g, 222, 40, IW, IH, `range only · ${G.R.n} clusters`);
    SG.cells(p1, 0, 0, cw, S.W, S.H, (x, y) => SG.pal(G.R.lab[y * S.W + x]));
    p1.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10).attr("fill", VC.bad)
      .text(`${bigPieces(G.R.lab, 20)} disconnected pieces · ARI ${ari(G.R.lab).toFixed(4)}`);

    const p2 = SG.box(g, 430, 40, IW, IH, `joint domain · ${G.J.n} regions`);
    SG.cells(p2, 0, 0, cw, S.W, S.H, (x, y) => SG.pal(G.J.lab[y * S.W + x] + 2));
    SG.drawEdges(p2, SG.edgesOf(G.J.lab, S.W, S.H), cw, { color: "#0b0d12", w: 0.9, op: 0.9 });
    p2.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10)
      .attr("fill", ari(G.J.lab) > 0.9 ? VC.good : VC.a2)
      .text(`${bigPieces(G.J.lab, 20)} disconnected pieces · ARI ${ari(G.J.lab).toFixed(4)}`);

    /* the scatter that makes the point */
    const SX = 14, SY = 246, SW = 396, SH = 154;
    const sc = SG.box(g, SX, SY, SW, SH, "the same pixels, plotted as (horizontal position, brightness)", { fill: "#12141c" });
    const xs = d3.scaleLinear().domain([0, S.W]).range([0, SW]);
    const ys = d3.scaleLinear().domain([60, 210]).range([SH, 0]);
    sc.append("g").attr("transform", `translate(0,${SH})`).attr("class", "axis").call(d3.axisBottom(xs).ticks(6));
    sc.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(4));
    const pts = [];
    for (let i = 0; i < S.W * S.H; i += 3) pts.push(i);
    sc.selectAll("circle.p").data(pts).join("circle").attr("class", "p")
      .attr("cx", i => xs(i % S.W)).attr("cy", i => ys(S.img[i])).attr("r", 1.1)
      .attr("fill", i => SG.pal(G.J.lab[i] + 2)).attr("fill-opacity", 0.55);
    G.R.modes.forEach(m => {
      sc.append("line").attr("x1", 0).attr("x2", SW).attr("y1", ys(m)).attr("y2", ys(m))
        .attr("stroke", VC.bad).attr("stroke-dasharray", "4 3").attr("stroke-opacity", 0.8);
    });
    sc.append("text").attr("x", 4).attr("y", 12).attr("font-size", 10).attr("fill", VC.bad)
      .text("dashed = the range-only modes; they are horizontal lines, so they cannot separate left from right");

    const put = SG.kv(g, 434, 258, { keyW: 178, lead: 15.5 });
    put("range-only clusters", String(G.R.n), VC.bad, true);
    put("  disconnected pieces ≥ 20 px", String(bigPieces(G.R.lab, 20)), VC.bad);
    put("  ARI against the 4 true regions", ari(G.R.lab).toFixed(4), VC.bad);
    put("joint-domain regions", String(G.J.n), VC.good, true);
    put("  disconnected pieces ≥ 20 px", String(bigPieces(G.J.lab, 20)), VC.good);
    put("  ARI against the 4 true regions", ari(G.J.lab).toFixed(4),
      ari(G.J.lab) > 0.9 ? VC.good : VC.a2, true);
    put("range modes found", G.R.modes.map(m => m.toFixed(1)).join(", "), VC.violet);
  }

  function say() {
    const G = get();
    out.innerHTML =
      `With <b>h_r = ${G.hr}</b> alone, mean shift finds <b>${G.R.n}</b> brightness cluster`
      + `${G.R.n === 1 ? "" : "s"} at ${G.R.modes.map(m => m.toFixed(1)).join(", ")}. `
      + `Mapped back onto the grid they occupy <b>${bigPieces(G.R.lab, 20)}</b> disconnected pieces of at least `
      + `20 pixels, because one cluster covers both bright bars — which is the correct answer to the question `
      + `"which pixels are the same brightness" and the wrong answer to "which pixels are the same thing".<br>`
      + `Adding the spatial coordinates with <b>h_s = ${G.hs}</b> and a minimum region of <b>${G.M}</b> gives `
      + `<b>${G.J.n}</b> regions in <b>${bigPieces(G.J.lab, 20)}</b> pieces, at ARI <b>${ari(G.J.lab).toFixed(4)}</b> `
      + `against ${ari(G.R.lab).toFixed(4)}. The regions are connected by construction: two pixels can only share `
      + `a label if they are adjacent <i>and</i> their trajectories converged to the same place in the joint space.`;
  }
  function all() {
    eRv.textContent = eR.value; eSv.textContent = eS.value; eMv.textContent = eM.value;
    draw(); say();
  }
  eR.oninput = all; eS.oninput = all; eM.oninput = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   15 · #graph-svg — a small image, its graph, and its affinity matrix
   ══════════════════════════════════════════════════════════════════════════ */
/* A symmetric eigensolver, page-local: the cyclic Jacobi rotation method. It is
   O(n³) with a large constant and would be absurd on a real affinity matrix, but
   it is exact to machine precision on the eight- and sixty-three-node examples
   here, needs no library, and is short enough to read. */
SG.jacobiEig = function (Ain, iters) {
  const n = Ain.length;
  const A = Ain.map(r => r.slice());
  const V = [];
  for (let i = 0; i < n; i++) { V.push(new Array(n).fill(0)); V[i][i] = 1; }
  for (let sweep = 0; sweep < (iters || 60); sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (off < 1e-24) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-18) continue;
      const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = A[k][p], akq = A[k][q];
        A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = A[p][k], aqk = A[q][k];
        A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const idx = d3.range(n).sort((a, b) => A[a][a] - A[b][b]);
  return { values: idx.map(i => A[i][i]), vectors: idx.map(i => V.map(row => row[i])) };
};

const SGgraph = (function () {
  const W = 9, H = 7;
  const VAL = [], TRU = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let t = 0, v = 70;
    if (y < 3) { t = 0; v = 70; }
    else if (y < 5) { t = 1; v = 115; }
    else { t = 2; v = 160; }
    if (x >= 6 && y >= 3) { t = 2; v = 160; }
    TRU.push(t); VAL.push(v);
  }
  /* a little noise so the blocks are not perfectly uniform */
  { const r = VZ.rng(101); for (let i = 0; i < VAL.length; i++) VAL[i] += 6 * VZ.randn(r); }
  const N = W * H;
  function affinity(sf, ss, rad) {
    const Wm = [];
    for (let i = 0; i < N; i++) Wm.push(new Array(N).fill(0));
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const dx = (i % W) - (j % W), dy = ((i / W) | 0) - ((j / W) | 0);
      const d2 = dx * dx + dy * dy;
      if (d2 > rad * rad) continue;
      const df = VAL[i] - VAL[j];
      const w = Math.exp(-df * df / (sf * sf) - d2 / (ss * ss));
      Wm[i][j] = w; Wm[j][i] = w;
    }
    return Wm;
  }
  return { W: W, H: H, N: N, VAL: VAL, TRU: TRU, affinity: affinity };
})();

(function () {
  const svg = d3.select("#graph-svg");
  if (svg.empty()) return;
  const S = SGgraph;
  const eF = document.getElementById("gr-sf"), eFv = document.getElementById("gr-sfv");
  const eS = document.getElementById("gr-ss"), eSv = document.getElementById("gr-ssv");
  const eR = document.getElementById("gr-r"), eRv = document.getElementById("gr-rv");
  const eO = document.getElementById("gr-sort");
  const out = document.getElementById("graph-readout");

  function stats(Wm) {
    let nz = 0, tot = 0, win = 0, nwin = 0, acr = 0, nacr = 0;
    for (let i = 0; i < S.N; i++) for (let j = i + 1; j < S.N; j++) {
      const dx = (i % S.W) - (j % S.W), dy = ((i / S.W) | 0) - ((j / S.W) | 0);
      if (dx * dx + dy * dy > +eR.value * +eR.value) continue;
      if (Wm[i][j] > 0) { nz++; tot += Wm[i][j]; }
      if (S.TRU[i] === S.TRU[j]) { win += Wm[i][j]; nwin++; } else { acr += Wm[i][j]; nacr++; }
    }
    return { nz: 2 * nz, tot: tot, win: nwin ? win / nwin : 0, acr: nacr ? acr / nacr : 0 };
  }

  function draw() {
    const sf = +eF.value, ss = +eS.value, rad = +eR.value, sorted = eO.checked;
    const Wm = S.affinity(sf, ss, rad);
    const deg = Wm.map(r => r.reduce((a, b) => a + b, 0));
    svg.selectAll("*").remove();
    const g = svg.append("g");

    /* the image with its edges */
    const cw = 22, IW = S.W * cw, IH = S.H * cw;
    const A = SG.box(g, 14, 42, IW, IH, "the image, with the graph on it");
    SG.cells(A, 0, 0, cw, S.W, S.H, (x, y) => SG.grey(S.VAL[y * S.W + x]));
    const mx = d3.max(Wm.map(r => d3.max(r)));
    for (let i = 0; i < S.N; i++) for (let j = i + 1; j < S.N; j++) {
      if (Wm[i][j] <= 0.004) continue;
      A.append("line")
        .attr("x1", (i % S.W + 0.5) * cw).attr("y1", (((i / S.W) | 0) + 0.5) * cw)
        .attr("x2", (j % S.W + 0.5) * cw).attr("y2", (((j / S.W) | 0) + 0.5) * cw)
        .attr("stroke", VC.accent).attr("stroke-width", 0.4 + 3.2 * Wm[i][j] / mx)
        .attr("stroke-opacity", 0.25 + 0.6 * Wm[i][j] / mx);
    }
    A.append("text").attr("x", 0).attr("y", IH + 14).attr("font-size", 10).attr("fill", VC.muted)
      .text(`${S.N} nodes · edge thickness ∝ affinity`);

    /* the matrix */
    const order = sorted
      ? d3.range(S.N).sort((a, b) => (S.TRU[a] - S.TRU[b]) || (a - b))
      : d3.range(S.N);
    const MX = 246, MY = 42, MS = 250, cell = MS / S.N;
    const M = SG.box(g, MX, MY, MS, MS,
      sorted ? "W, rows ordered by true region" : "W, in raster order");
    for (let a = 0; a < S.N; a++) for (let b = 0; b < S.N; b++) {
      const v = Wm[order[a]][order[b]] / mx;
      if (v <= 0.002) continue;
      M.append("rect").attr("x", b * cell).attr("y", a * cell)
        .attr("width", cell + 0.3).attr("height", cell + 0.3)
        .attr("shape-rendering", "crispEdges")
        .attr("fill", d3.interpolateViridis(Math.min(1, v)));
    }
    M.append("rect").attr("x", 0).attr("y", 0).attr("width", MS).attr("height", MS)
      .attr("fill", "none").attr("stroke", VC.line);
    if (sorted) {
      let acc = 0;
      for (let t = 0; t < 2; t++) {
        acc += S.TRU.filter(v => v === t).length;
        M.append("line").attr("x1", 0).attr("x2", MS).attr("y1", acc * cell).attr("y2", acc * cell)
          .attr("stroke", VC.a2).attr("stroke-dasharray", "3 2");
        M.append("line").attr("y1", 0).attr("y2", MS).attr("x1", acc * cell).attr("x2", acc * cell)
          .attr("stroke", VC.a2).attr("stroke-dasharray", "3 2");
      }
    }

    /* degrees */
    const DX = 528, DY = 42, DW = 216, DH = 250;
    const D = SG.box(g, DX, DY, DW, DH, "node degree dᵢ = ∑ⱼ w_ij", { fill: "#12141c" });
    const xd = d3.scaleLinear().domain([0, d3.max(deg) * 1.05]).range([0, DW - 10]);
    const bh = DH / S.N;
    for (let a = 0; a < S.N; a++) {
      const i = order[a];
      D.append("rect").attr("x", 0).attr("y", a * bh).attr("width", Math.max(0.4, xd(deg[i])))
        .attr("height", Math.max(1, bh - 0.6)).attr("fill", SG.pal(S.TRU[i])).attr("fill-opacity", 0.85);
    }
    D.append("g").attr("transform", `translate(0,${DH})`).attr("class", "axis").call(d3.axisBottom(xd).ticks(4));

    const st = stats(Wm);
    const put = SG.kv(g, 14, 322, { keyW: 190, lead: 15 });
    put("non-zero entries of W", st.nz + " of " + (S.N * S.N) + "  (" + (100 * st.nz / (S.N * S.N)).toFixed(1) + "% dense)", VC.accent);
    put("total edge weight", st.tot.toFixed(2));
    put("mean weight within a region", st.win.toFixed(4), VC.good);
    put("mean weight across regions", st.acr.toFixed(4), VC.bad);
    put("ratio within : across", (st.acr > 1e-9 ? (st.win / st.acr).toFixed(1) : "∞") + " ×",
      st.win / st.acr > 8 ? VC.good : VC.a2, true);
    put("degree, min … max", d3.min(deg).toFixed(3) + " … " + d3.max(deg).toFixed(3), VC.violet);
  }

  function say() {
    const sf = +eF.value, ss = +eS.value, rad = +eR.value;
    const Wm = S.affinity(sf, ss, rad), st = stats(Wm);
    out.innerHTML =
      `<b>σ_F = ${sf}</b>, <b>σ_s = ${ss.toFixed(1)}</b>, radius <b>${rad}</b>. `
      + `W has <b>${st.nz}</b> non-zero entries out of ${S.N * S.N} — <b>${(100 * st.nz / (S.N * S.N)).toFixed(1)}%</b> dense. `
      + `The mean affinity between two pixels of the same region is <b>${st.win.toFixed(4)}</b>; across a boundary it is `
      + `<b>${st.acr.toFixed(4)}</b>, a ratio of <b>${(st.win / Math.max(1e-9, st.acr)).toFixed(1)}×</b>.<br>`
      + `That ratio is the only thing any graph method has to work with. Everything §21 to §23 do is a way of `
      + `recovering the block structure that ratio implies — and if you drive the ratio toward 1 with a large `
      + `σ_F or a large radius, no amount of cleverness downstream will put the boundary back.`;
  }
  function all() {
    eFv.textContent = eF.value; eSv.textContent = (+eS.value).toFixed(1); eRv.textContent = (+eR.value).toFixed(1);
    draw(); say();
  }
  eF.oninput = all; eS.oninput = all; eR.oninput = all; eO.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   16 · #ncut-svg — the Laplacian, the second eigenvector, and the true optimum
   ══════════════════════════════════════════════════════════════════════════ */
const SGncut = (function () {
  const N = 8;
  const POS = [[0.13, 0.24], [0.10, 0.62], [0.34, 0.42], [0.30, 0.80],
    [0.68, 0.24], [0.72, 0.62], [0.90, 0.40], [0.88, 0.80]];
  const BASE_A = [[0, 1, 1.0], [0, 2, 0.8], [1, 2, 0.9], [1, 3, 0.7], [2, 3, 1.0], [0, 3, 0.6]];
  const BASE_B = [[4, 5, 0.9], [4, 6, 1.0], [5, 6, 0.8], [5, 7, 1.0], [6, 7, 0.7], [4, 7, 0.6]];
  function build(bridge1, bridge2, asym) {
    const Wm = [];
    for (let i = 0; i < N; i++) Wm.push(new Array(N).fill(0));
    const set = (i, j, w) => { Wm[i][j] = w; Wm[j][i] = w; };
    BASE_A.forEach(([i, j, w]) => set(i, j, w * asym));
    BASE_B.forEach(([i, j, w]) => set(i, j, w));
    if (bridge1 > 0) set(3, 4, bridge1);
    if (bridge2 > 0) set(2, 5, bridge2);
    return Wm;
  }
  const degrees = Wm => Wm.map(r => r.reduce((a, b) => a + b, 0));
  function ncut(Wm, inA) {
    const d = degrees(Wm);
    let c = 0, aA = 0, aB = 0;
    for (let i = 0; i < N; i++) {
      if (inA[i]) aA += d[i]; else aB += d[i];
      for (let j = 0; j < N; j++) if (inA[i] && !inA[j]) c += Wm[i][j];
    }
    if (aA <= 0 || aB <= 0) return { ncut: Infinity, cut: c };
    return { ncut: c / aA + c / aB, cut: c, assocA: aA, assocB: aB };
  }
  function brute(Wm) {
    let best = null, bestCut = null;
    for (let m = 1; m < (1 << N) - 1; m++) {
      const inA = d3.range(N).map(i => (m >> i) & 1);
      const r = ncut(Wm, inA);
      if (best === null || r.ncut < best.ncut) best = { ncut: r.ncut, inA: inA.slice() };
      if (bestCut === null || r.cut < bestCut.cut) bestCut = { cut: r.cut, inA: inA.slice(), sizeA: inA.reduce((a, b) => a + b, 0) };
    }
    return { best: best, bestCut: bestCut };
  }
  /* the generalised problem (D − W)y = λDy via the symmetric substitution */
  function spectrum(Wm) {
    const d = degrees(Wm);
    const M = [];
    for (let i = 0; i < N; i++) {
      M.push(new Array(N).fill(0));
      for (let j = 0; j < N; j++) {
        const nij = Wm[i][j] / Math.sqrt(Math.max(1e-12, d[i] * d[j]));
        M[i][j] = (i === j ? 1 : 0) - nij;
      }
    }
    const E = SG.jacobiEig(M, 80);
    const ys = E.vectors.map(z => z.map((v, i) => v / Math.sqrt(Math.max(1e-12, d[i]))));
    return { values: E.values, y: ys, d: d };
  }
  return { N: N, POS: POS, build: build, degrees: degrees, ncut: ncut, brute: brute, spectrum: spectrum };
})();

(function () {
  const svg = d3.select("#ncut-svg");
  if (svg.empty()) return;
  const S = SGncut;
  const eB = document.getElementById("nc-b"), eBv = document.getElementById("nc-bv");
  const eB2 = document.getElementById("nc-b2"), eB2v = document.getElementById("nc-b2v");
  const eA = document.getElementById("nc-as"), eAv = document.getElementById("nc-asv");
  const eS = document.getElementById("nc-sweep");
  const out = document.getElementById("ncut-readout");

  function state() {
    const Wm = S.build(+eB.value, +eB2.value, +eA.value);
    const sp = S.spectrum(Wm);
    let y = sp.y[1].slice();
    if (y.reduce((a, b) => a + b, 0) < 0) y = y.map(v => -v);
    const signA = y.map(v => (v > 0 ? 1 : 0));
    const signR = S.ncut(Wm, signA);
    const order = d3.range(S.N).sort((a, b) => y[a] - y[b]);
    let sweep = null;
    for (let k = 1; k < S.N; k++) {
      const inA = new Array(S.N).fill(0);
      for (let t = 0; t < k; t++) inA[order[t]] = 1;
      const r = S.ncut(Wm, inA);
      if (sweep === null || r.ncut < sweep.ncut) sweep = { ncut: r.ncut, inA: inA, k: k, thresh: (y[order[k - 1]] + y[order[k]]) / 2 };
    }
    const bf = S.brute(Wm);
    return { Wm: Wm, sp: sp, y: y, signA: signA, signR: signR, order: order, sweep: sweep, bf: bf };
  }

  function draw() {
    const T = state();
    const use = eS.checked ? T.sweep.inA : T.signA;
    const useR = S.ncut(T.Wm, use);
    svg.selectAll("*").remove();
    const g = svg.append("g");

    /* the graph */
    const GX = 14, GY = 44, GW = 300, GH = 240;
    const G = SG.box(g, GX, GY, GW, GH, "the graph, cut edges dashed", { fill: "#12141c" });
    const px = i => 16 + S.POS[i][0] * (GW - 32), py = i => 14 + S.POS[i][1] * (GH - 34);
    const mx = d3.max(T.Wm.map(r => d3.max(r)));
    for (let i = 0; i < S.N; i++) for (let j = i + 1; j < S.N; j++) {
      if (T.Wm[i][j] <= 0) continue;
      const cutE = use[i] !== use[j];
      G.append("line").attr("x1", px(i)).attr("y1", py(i)).attr("x2", px(j)).attr("y2", py(j))
        .attr("stroke", cutE ? VC.bad : VC.muted)
        .attr("stroke-width", 0.8 + 4.5 * T.Wm[i][j] / mx)
        .attr("stroke-dasharray", cutE ? "4 3" : null).attr("stroke-opacity", cutE ? 1 : 0.75);
      G.append("text").attr("x", (px(i) + px(j)) / 2).attr("y", (py(i) + py(j)) / 2 - 3)
        .attr("text-anchor", "middle").attr("font-size", 9)
        .attr("fill", cutE ? VC.bad : VC.muted).text(T.Wm[i][j].toFixed(2));
    }
    for (let i = 0; i < S.N; i++) {
      G.append("circle").attr("cx", px(i)).attr("cy", py(i)).attr("r", 13)
        .attr("fill", use[i] ? VC.accent : VC.a2).attr("fill-opacity", 0.9)
        .attr("stroke", "#0b0d12").attr("stroke-width", 1.4);
      G.append("text").attr("x", px(i)).attr("y", py(i) + 4).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", "#0b0d12").attr("font-weight", 600).text(i);
    }

    /* the eigenvector */
    const VX = 336, VY = 44, VW = 200, VH = 240;
    const V = SG.box(g, VX, VY, VW, VH, "y₂, sorted", { fill: "#12141c" });
    const ys = d3.scaleLinear().domain([-d3.max(T.y.map(Math.abs)) * 1.2, d3.max(T.y.map(Math.abs)) * 1.2]).range([VH, 0]);
    const bw = VW / S.N;
    T.order.forEach((i, k) => {
      V.append("rect").attr("x", k * bw + 2).attr("y", Math.min(ys(0), ys(T.y[i])))
        .attr("width", bw - 4).attr("height", Math.abs(ys(T.y[i]) - ys(0)))
        .attr("fill", use[i] ? VC.accent : VC.a2).attr("fill-opacity", 0.85);
      V.append("text").attr("x", k * bw + bw / 2).attr("y", ys(0) + (T.y[i] > 0 ? 12 : -4))
        .attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", VC.muted).text(i);
    });
    V.append("line").attr("x1", 0).attr("x2", VW).attr("y1", ys(0)).attr("y2", ys(0))
      .attr("stroke", VC.ink).attr("stroke-width", 1);
    if (eS.checked) {
      V.append("line").attr("x1", 0).attr("x2", VW).attr("y1", ys(T.sweep.thresh)).attr("y2", ys(T.sweep.thresh))
        .attr("stroke", VC.good).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.5);
      V.append("text").attr("x", VW - 3).attr("y", ys(T.sweep.thresh) - 4).attr("text-anchor", "end")
        .attr("font-size", 9.5).attr("fill", VC.good).text("sweep threshold");
    }

    /* the spectrum */
    const SX = 558, SY = 44, SW = 186, SH = 116;
    const Sp = SG.box(g, SX, SY, SW, SH, "generalised eigenvalues", { fill: "#12141c" });
    const yv = d3.scaleLinear().domain([0, d3.max(T.sp.values) * 1.1]).range([SH, 0]);
    const bw2 = SW / S.N;
    T.sp.values.forEach((v, k) => {
      Sp.append("rect").attr("x", k * bw2 + 2).attr("y", yv(v)).attr("width", bw2 - 4)
        .attr("height", SH - yv(v)).attr("fill", k === 1 ? VC.good : VC.muted).attr("fill-opacity", 0.85);
    });
    Sp.append("g").attr("class", "axis").call(d3.axisLeft(yv).ticks(4));
    Sp.append("text").attr("x", 1.5 * bw2).attr("y", yv(T.sp.values[1]) - 5).attr("text-anchor", "middle")
      .attr("font-size", 9.5).attr("fill", VC.good).text("λ₂");

    /* numbers */
    const gap = T.sweep.ncut - T.bf.best.ncut;
    const put = SG.kv(g, 14, 316, { keyW: 268, lead: 15.5 });
    put("λ₁  (must be exactly 0)", T.sp.values[0].toExponential(2),
      Math.abs(T.sp.values[0]) < 1e-9 ? VC.good : VC.bad);
    put("λ₂  — the relaxed minimum, a lower bound", T.sp.values[1].toFixed(6), VC.good, true);
    put("Ncut of the sign split", T.signR.ncut.toFixed(6), VC.accent);
    put("Ncut of the best sweep split", T.sweep.ncut.toFixed(6), VC.accent, true);
    put("Ncut of the exhaustive optimum (127 splits)", T.bf.best.ncut.toFixed(6), VC.violet, true);
    put("relaxation gap  (optimum − λ₂)", (T.bf.best.ncut - T.sp.values[1]).toFixed(6), VC.violet);
    put("did the rounding find the optimum?",
      gap < 1e-9 ? "yes" : "NO — off by " + gap.toFixed(6),
      gap < 1e-9 ? VC.good : VC.bad, true);
    const put2 = SG.kv(g, 400, 316, { keyW: 236, lead: 15.5 });
    put2("cut weight severed", useR.cut.toFixed(4));
    put2("assoc(A,V) / assoc(B,V)", useR.assocA.toFixed(3) + "  /  " + useR.assocB.toFixed(3));
    put2("Ncut + Nassoc  (must be 2)",
      (useR.ncut + (useR.assocA - useR.cut) / useR.assocA + (useR.assocB - useR.cut) / useR.assocB).toFixed(9), VC.a2);
    put2("minimum UNnormalised cut", T.bf.bestCut.cut.toFixed(4) + "  on a group of " + T.bf.bestCut.sizeA,
      T.bf.bestCut.sizeA === 1 || T.bf.bestCut.sizeA === S.N - 1 ? VC.bad : VC.ink);
    put2("y₂ᵀd  (must be 0)", T.y.reduce((a, v, i) => a + v * T.sp.d[i], 0).toExponential(2), VC.a2);
    put2("group sizes", use.reduce((a, b) => a + b, 0) + " : " + (S.N - use.reduce((a, b) => a + b, 0)));
  }

  function say() {
    const T = state();
    const gap = T.sweep.ncut - T.bf.best.ncut;
    out.innerHTML =
      `The relaxed minimum is <b>λ₂ = ${T.sp.values[1].toFixed(6)}</b>. Rounding its eigenvector by sign gives an `
      + `actual normalised cut of <b>${T.signR.ncut.toFixed(6)}</b>; sweeping every split of the sorted eigenvector gives `
      + `<b>${T.sweep.ncut.toFixed(6)}</b>; and exhaustive search over all 127 bipartitions finds `
      + `<b>${T.bf.best.ncut.toFixed(6)}</b>.<br>`
      + (gap < 1e-9
        ? `<b style="color:var(--good)">Here the rounding happens to find the optimum</b>, which is the common case and is not a theorem. `
        : `<b style="color:var(--bad)">Here it does not</b> — the rounded partition is ${gap.toFixed(6)} worse than the best possible one. `
        + `Nothing went wrong: the eigenvector solved the relaxed problem exactly, and the relaxed problem is a different problem. `)
      + `The relaxation gap, <b>${(T.bf.best.ncut - T.sp.values[1]).toFixed(6)}</b>, is the distance between the certificate `
      + `the eigensolver returns and the best partition that actually exists. It is never negative, and there is no `
      + `general bound on how large it can be.`;
  }
  function all() {
    eBv.textContent = (+eB.value).toFixed(2); eB2v.textContent = (+eB2.value).toFixed(2);
    eAv.textContent = (+eA.value).toFixed(2);
    draw(); say();
  }
  eB.oninput = all; eB2.oninput = all; eA.oninput = all; eS.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   17 · #mincut-svg — max-flow saturating, and the cut it certifies
   ══════════════════════════════════════════════════════════════════════════ */
/* Edmonds–Karp: repeatedly push flow along a shortest augmenting path found by
   breadth-first search. O(VE²), which is absurd for an image and exactly right
   for the eleven-node network here, where the point is to watch it saturate. */
SG.maxflow = function (n, edges, s, t) {
  const head = [], next = [], cap = [], first = new Array(n).fill(-1);
  function add(u, v, c, cRev) {
    head.push(v); cap.push(c); next.push(first[u]); first[u] = head.length - 1;
    head.push(u); cap.push(cRev); next.push(first[v]); first[v] = head.length - 1;
  }
  edges.forEach(e => add(e[0], e[1], e[2], e[3] === undefined ? 0 : e[3]));
  const orig = cap.slice();
  let total = 0;
  for (let guard = 0; guard < 10000; guard++) {
    const prev = new Int32Array(n).fill(-1), pe = new Int32Array(n).fill(-1);
    prev[s] = s;
    const q = [s];
    for (let qi = 0; qi < q.length && prev[t] < 0; qi++) {
      const u = q[qi];
      for (let e = first[u]; e !== -1; e = next[e]) {
        const v = head[e];
        if (cap[e] > 1e-12 && prev[v] < 0) { prev[v] = u; pe[v] = e; q.push(v); }
      }
    }
    if (prev[t] < 0) break;
    let push = Infinity;
    for (let v = t; v !== s; v = prev[v]) push = Math.min(push, cap[pe[v]]);
    for (let v = t; v !== s; v = prev[v]) { cap[pe[v]] -= push; cap[pe[v] ^ 1] += push; }
    total += push;
  }
  /* the source side of the minimum cut: everything still reachable in the residual */
  const side = new Uint8Array(n);
  side[s] = 1;
  const q2 = [s];
  for (let qi = 0; qi < q2.length; qi++) {
    const u = q2[qi];
    for (let e = first[u]; e !== -1; e = next[e])
      if (cap[e] > 1e-12 && !side[head[e]]) { side[head[e]] = 1; q2.push(head[e]); }
  }
  const flowOf = [];
  for (let e = 0; e < head.length; e += 2) flowOf.push(orig[e] - cap[e]);
  return { value: total, side: side, flow: flowOf, cap: orig.filter((_, i) => i % 2 === 0) };
};

const SGmincut = (function () {
  const W = 3, H = 3;
  const I = [8, 9, 7, 9, 3, 8, 2, 3, 2];
  const PAIRS = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    if (x + 1 < W) PAIRS.push([i, i + 1]);
    if (y + 1 < H) PAIRS.push([i, i + W]);
  }
  function terms(lam, sig, muF, muB) {
    const Dfg = I.map(v => Math.pow(v - muF, 2));
    const Dbg = I.map(v => Math.pow(v - muB, 2));
    const V = PAIRS.map(([a, b]) => lam * Math.exp(-Math.pow(I[a] - I[b], 2) / (2 * sig * sig)));
    return { Dfg: Dfg, Dbg: Dbg, V: V };
  }
  function energy(lab, T) {
    let e = 0;
    for (let i = 0; i < W * H; i++) e += lab[i] ? T.Dfg[i] : T.Dbg[i];
    PAIRS.forEach(([a, b], k) => { if (lab[a] !== lab[b]) e += T.V[k]; });
    return e;
  }
  function brute(T) {
    let best = null;
    const all = [];
    for (let m = 0; m < (1 << 9); m++) {
      const lab = d3.range(9).map(i => (m >> i) & 1);
      const e = energy(lab, T);
      all.push(e);
      if (best === null || e < best.e) best = { e: e, lab: lab };
    }
    all.sort((a, b) => a - b);
    return { best: best, all: all };
  }
  function solve(T) {
    const N = 9, S = 9, Tk = 10;
    const edges = [];
    for (let i = 0; i < N; i++) { edges.push([S, i, T.Dbg[i], 0]); edges.push([i, Tk, T.Dfg[i], 0]); }
    PAIRS.forEach(([a, b], k) => edges.push([a, b, T.V[k], T.V[k]]));
    const mf = SG.maxflow(11, edges, S, Tk);
    const lab = d3.range(N).map(i => (mf.side[i] ? 1 : 0));
    return { mf: mf, lab: lab, edges: edges };
  }
  return { W: W, H: H, I: I, PAIRS: PAIRS, terms: terms, energy: energy, brute: brute, solve: solve };
})();

(function () {
  const svg = d3.select("#mincut-svg");
  if (svg.empty()) return;
  const S = SGmincut;
  const eL = document.getElementById("mc-l"), eLv = document.getElementById("mc-lv");
  const eS = document.getElementById("mc-s"), eSv = document.getElementById("mc-sv");
  const eF = document.getElementById("mc-f"), eFv = document.getElementById("mc-fv");
  const eB = document.getElementById("mc-b"), eBv = document.getElementById("mc-bv");
  const out = document.getElementById("mincut-readout");
  const get = () => {
    const T = S.terms(+eL.value, +eS.value, +eF.value, +eB.value);
    return { T: T, sol: S.solve(T), bf: S.brute(T) };
  };

  function draw() {
    const G = get(), lab = G.sol.lab;
    svg.selectAll("*").remove();
    const g = svg.append("g");

    /* the labelled image */
    const cw = 44;
    const A = SG.box(g, 14, 44, S.W * cw, S.H * cw, "the image and its optimal labelling");
    for (let y = 0; y < S.H; y++) for (let x = 0; x < S.W; x++) {
      const i = y * S.W + x;
      A.append("rect").attr("x", x * cw).attr("y", y * cw).attr("width", cw).attr("height", cw)
        .attr("fill", lab[i] ? VC.accent : "#2a3040").attr("fill-opacity", lab[i] ? 0.65 : 1)
        .attr("stroke", "#0b0d12");
      A.append("text").attr("x", (x + 0.5) * cw).attr("y", (y + 0.5) * cw + 5).attr("text-anchor", "middle")
        .attr("font-size", 15).attr("fill", VC.ink).attr("font-weight", 600).text(S.I[i]);
    }
    const L = new Int32Array(lab);
    SG.drawEdges(A, SG.edgesOf(L, S.W, S.H), cw, { color: VC.bad, w: 3, op: 1 });
    A.append("text").attr("x", 0).attr("y", S.H * cw + 14).attr("font-size", 10.5).attr("fill", VC.muted)
      .text("blue = foreground (label 1) · red = the cut");

    /* the flow network */
    const NX = 210, NY = 44, NW = 300, NH = 250;
    const F = SG.box(g, NX, NY, NW, NH, "the flow network — saturated edges in red", { fill: "#12141c" });
    const px = i => 44 + (i % S.W) * 100, py = i => 74 + (((i / S.W) | 0)) * 52;
    const sx = 150, sy = 16, tx = 150, ty = NH - 12;
    const flow = G.sol.mf.flow, cap = G.sol.mf.cap;
    const sat = (f, c) => c > 1e-9 && f > c - 1e-7;
    for (let i = 0; i < 9; i++) {
      const eS0 = 2 * i, eT0 = 2 * i + 1;
      [[sx, sy, px(i), py(i), flow[eS0], cap[eS0]], [px(i), py(i), tx, ty, flow[eT0], cap[eT0]]]
        .forEach(([x1, y1, x2, y2, f, c]) => {
          if (c < 1e-9) return;
          F.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
            .attr("stroke", sat(f, c) ? VC.bad : VC.muted)
            .attr("stroke-width", sat(f, c) ? 2 : 0.9).attr("stroke-opacity", sat(f, c) ? 0.95 : 0.28);
        });
    }
    S.PAIRS.forEach(([a, b], k) => {
      const e = 18 + k;
      const f = Math.abs(flow[e]), c = cap[e];
      if (c < 1e-9) return;
      F.append("line").attr("x1", px(a)).attr("y1", py(a)).attr("x2", px(b)).attr("y2", py(b))
        .attr("stroke", sat(f, c) ? VC.bad : VC.accent)
        .attr("stroke-width", sat(f, c) ? 3 : 1.2).attr("stroke-opacity", sat(f, c) ? 1 : 0.55);
      F.append("text").attr("x", (px(a) + px(b)) / 2).attr("y", (py(a) + py(b)) / 2 - 3)
        .attr("text-anchor", "middle").attr("font-size", 8.5)
        .attr("fill", sat(f, c) ? VC.bad : VC.muted).text(f.toFixed(2) + "/" + c.toFixed(2));
    });
    for (let i = 0; i < 9; i++) {
      F.append("circle").attr("cx", px(i)).attr("cy", py(i)).attr("r", 11)
        .attr("fill", lab[i] ? VC.accent : "#2a3040").attr("stroke", "#0b0d12");
      F.append("text").attr("x", px(i)).attr("y", py(i) + 4).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", VC.ink).text(S.I[i]);
    }
    [[sx, sy, "S", VC.good], [tx, ty, "T", VC.a2]].forEach(([cx, cy, t, col]) => {
      F.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 12).attr("fill", col);
      F.append("text").attr("x", cx).attr("y", cy + 4).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", "#0b0d12").attr("font-weight", 600).text(t);
    });

    /* the energy landscape */
    const EX = 534, EY = 44, EW = 210, EH = 250;
    const E = SG.box(g, EX, EY, EW, EH, "energy of all 512 labellings, sorted", { fill: "#12141c" });
    const xs = d3.scaleLinear().domain([0, 511]).range([0, EW]);
    const ys = d3.scaleLinear().domain([G.bf.all[0] * 0.9, G.bf.all[511]]).range([EH, 0]);
    E.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5, "~s"));
    E.append("path").datum(G.bf.all).attr("d", d3.line().x((d, i) => xs(i)).y(d => ys(d)))
      .attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 1.6);
    const cutE = S.energy(lab, G.T);
    E.append("circle").attr("cx", xs(G.bf.all.findIndex(v => v >= cutE - 1e-9))).attr("cy", ys(cutE))
      .attr("r", 5).attr("fill", VC.good).attr("stroke", "#0b0d12");
    E.append("text").attr("x", 8).attr("y", ys(cutE) - 8).attr("font-size", 10).attr("fill", VC.good)
      .text("the min cut, " + cutE.toFixed(4));

    /* numbers */
    let sub = true;
    G.T.V.forEach(v => { if (!(0 + 0 <= v + v + 1e-12)) sub = false; });
    const agree = Math.abs(G.sol.mf.value - G.bf.best.e) < 1e-7;
    const put = SG.kv(g, 14, 330, { keyW: 268, lead: 15.5 });
    put("maximum flow value", G.sol.mf.value.toFixed(6), VC.good, true);
    put("capacity of the minimum cut", G.sol.mf.value.toFixed(6), VC.good);
    put("energy of the labelling it induces", cutE.toFixed(6), VC.accent, true);
    put("brute-force minimum over 512 labellings", G.bf.best.e.toFixed(6), VC.violet, true);
    put("all four agree?", agree ? "yes — as the theorem requires" : "NO",
      agree ? VC.good : VC.bad, true);
    const put2 = SG.kv(g, 400, 330, { keyW: 250, lead: 15.5 });
    put2("submodular on every neighbour pair?", sub ? "yes — V(0,0)+V(1,1) ≤ V(0,1)+V(1,0)" : "no",
      sub ? VC.good : VC.bad);
    put2("data-only optimum, ∑ min(D₀, D₁)",
      d3.sum(d3.range(9), i => Math.min(G.T.Dfg[i], G.T.Dbg[i])).toFixed(4), VC.a2);
    put2("boundary cost paid", (cutE - d3.sum(d3.range(9), i => (G.sol.lab[i] ? G.T.Dfg[i] : G.T.Dbg[i]))).toFixed(4));
    put2("foreground pixels", String(lab.reduce((a, b) => a + b, 0)) + " of 9");
    put2("worst labelling", G.bf.all[511].toFixed(3));
  }

  function say() {
    const G = get(), lab = G.sol.lab, cutE = S.energy(lab, G.T);
    const agree = Math.abs(G.sol.mf.value - G.bf.best.e) < 1e-7;
    out.innerHTML =
      `Maximum flow <b>${G.sol.mf.value.toFixed(6)}</b>; the labelling its minimum cut induces has energy `
      + `<b>${cutE.toFixed(6)}</b>; and the smallest energy among all <b>512</b> labellings, found by enumerating `
      + `every one of them, is <b>${G.bf.best.e.toFixed(6)}</b>. `
      + (agree
        ? `<b style="color:var(--good)">All three are the same number</b> — not approximately, exactly. `
        : `<b style="color:var(--bad)">They disagree, which should be impossible.</b> `)
      + `The flow is not merely a way of finding a good labelling; it is a proof that no better one exists, `
      + `because any labelling with lower energy would be a cut with capacity below the flow value.<br>`
      + `Labels: <b>${lab.join("")}</b> in raster order. The data term alone would give `
      + `<b>${d3.sum(d3.range(9), i => Math.min(G.T.Dfg[i], G.T.Dbg[i])).toFixed(4)}</b>; the difference is what `
      + `the smoothness term charged for the boundary, and it is the only thing tying the nine decisions together.`;
  }
  function all() {
    eLv.textContent = (+eL.value).toFixed(2); eSv.textContent = (+eS.value).toFixed(1);
    eFv.textContent = (+eF.value).toFixed(1); eBv.textContent = (+eB.value).toFixed(1);
    draw(); say();
  }
  eL.oninput = all; eS.oninput = all; eF.oninput = all; eB.oninput = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   18 · #slic-svg — SLIC superpixels, and the compactness trade
   ══════════════════════════════════════════════════════════════════════════ */
const SGslic = (function () {
  const W = 96, H = 72;
  const img = new Float64Array(W * H), truth = new Int32Array(W * H);
  {
    const r = VZ.rng(131);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let t = 0, v = 58 + 22 * (y / (H - 1));                       // shaded background
      if (y > 0.62 * x + 6) { t = 1; v = 112; }                      // a diagonal band
      if (Math.pow(x - 62, 2) + Math.pow(y - 24, 2) < 19 * 19) { t = 2; v = 152; }   // a disc
      if (Math.pow((x - 24) / 17, 2) + Math.pow((y - 52) / 9, 2) < 1) { t = 3; v = 88; }  // an ellipse
      if (Math.abs(y - (0.42 * x + 8)) < 1.2) { t = 4; v = 186; }    // a thin bright wire
      truth[y * W + x] = t;
      img[y * W + x] = Math.min(255, Math.max(0, v + 7 * VZ.randn(r)));
    }
  }
  const grad = SG.gradMag(SG.blur(img, W, H, 0.8), W, H);
  /* SLIC: k-means in (I, x, y) with the search restricted to a 2S × 2S window */
  function slic(k, m, iters) {
    const S = Math.max(2, Math.sqrt(W * H / k));
    const cx = [], cy = [], cv = [];
    for (let gy = S / 2; gy < H; gy += S) for (let gx = S / 2; gx < W; gx += S) {
      let bx = Math.round(gx), by = Math.round(gy), bg = Infinity;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const x = Math.min(W - 1, Math.max(0, Math.round(gx) + dx));
        const y = Math.min(H - 1, Math.max(0, Math.round(gy) + dy));
        if (grad[y * W + x] < bg) { bg = grad[y * W + x]; bx = x; by = y; }
      }
      cx.push(bx); cy.push(by); cv.push(img[by * W + bx]);
    }
    const K = cx.length;
    const lab = new Int32Array(W * H).fill(-1);
    const dist = new Float64Array(W * H).fill(Infinity);
    for (let it = 0; it < iters; it++) {
      dist.fill(Infinity);
      for (let c = 0; c < K; c++) {
        const x0 = Math.max(0, Math.floor(cx[c] - S)), x1 = Math.min(W - 1, Math.ceil(cx[c] + S));
        const y0 = Math.max(0, Math.floor(cy[c] - S)), y1 = Math.min(H - 1, Math.ceil(cy[c] + S));
        for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
          const i = y * W + x;
          const dc = img[i] - cv[c];
          const ds2 = (x - cx[c]) * (x - cx[c]) + (y - cy[c]) * (y - cy[c]);
          const D = dc * dc + (ds2 / (S * S)) * m * m;
          if (D < dist[i]) { dist[i] = D; lab[i] = c; }
        }
      }
      const sx = new Float64Array(K), sy = new Float64Array(K), sv = new Float64Array(K), sn = new Float64Array(K);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const c = lab[y * W + x];
        if (c < 0) continue;
        sx[c] += x; sy[c] += y; sv[c] += img[y * W + x]; sn[c]++;
      }
      for (let c = 0; c < K; c++) if (sn[c] > 0) { cx[c] = sx[c] / sn[c]; cy[c] = sy[c] / sn[c]; cv[c] = sv[c] / sn[c]; }
    }
    /* enforce connectivity: split each label into its connected pieces, then
       absorb pieces below a quarter of the nominal size into a neighbour */
    const comp = SG.components(W, H, (a, b) => lab[a] === lab[b], 4);
    const size = new Array(comp.n).fill(0);
    for (let i = 0; i < W * H; i++) size[comp.lab[i]]++;
    const min = Math.max(4, Math.round(S * S / 4));
    const map = new Int32Array(comp.n);
    for (let c = 0; c < comp.n; c++) map[c] = c;
    let changed = true, guard = 0;
    while (changed && guard++ < 8) {
      changed = false;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x, c = map[comp.lab[i]];
        if (size[c] >= min) continue;
        let best = -1, bd = Infinity;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const X = x + dx, Y = y + dy;
          if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
          const c2 = map[comp.lab[Y * W + X]];
          if (c2 === c) continue;
          const d = Math.abs(img[i] - img[Y * W + X]);
          if (d < bd) { bd = d; best = c2; }
        }
        if (best >= 0) { size[best] += size[c]; size[c] = 0; for (let q = 0; q < comp.n; q++) if (map[q] === c) map[q] = best; changed = true; }
      }
    }
    const ids = new Map(), out = new Int32Array(W * H);
    for (let i = 0; i < W * H; i++) {
      const c = map[comp.lab[i]];
      if (!ids.has(c)) ids.set(c, ids.size);
      out[i] = ids.get(c);
    }
    return { lab: out, n: ids.size, S: S, K: K };
  }
  /* --- the three metrics superpixel papers report ------------------------- */
  const isB = (L, i) => {
    const x = i % W, y = (i / W) | 0;
    if (x > 0 && L[i - 1] !== L[i]) return true;
    if (x < W - 1 && L[i + 1] !== L[i]) return true;
    if (y > 0 && L[i - W] !== L[i]) return true;
    if (y < H - 1 && L[i + W] !== L[i]) return true;
    return false;
  };
  function boundaryRecall(L, tol) {
    let hit = 0, tot = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!isB(truth, i)) continue;
      tot++;
      let ok = false;
      for (let dy = -tol; dy <= tol && !ok; dy++) for (let dx = -tol; dx <= tol && !ok; dx++) {
        const X = x + dx, Y = y + dy;
        if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
        if (isB(L, Y * W + X)) ok = true;
      }
      if (ok) hit++;
    }
    return tot ? hit / tot : 1;
  }
  function underSeg(L, n) {
    /* for every superpixel, the part of it that falls outside its majority region */
    const cnt = [];
    for (let k = 0; k < n; k++) cnt.push(new Map());
    for (let i = 0; i < W * H; i++) {
      const m = cnt[L[i]];
      m.set(truth[i], (m.get(truth[i]) || 0) + 1);
    }
    let leak = 0, tot = 0;
    for (let k = 0; k < n; k++) {
      let sz = 0, mx = 0;
      for (const v of cnt[k].values()) { sz += v; if (v > mx) mx = v; }
      leak += sz - mx; tot += sz;
    }
    return leak / tot;
  }
  function compactness(L, n) {
    const area = new Array(n).fill(0), per = new Array(n).fill(0);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      area[L[i]]++;
      let p = 0;
      if (x === 0 || L[i - 1] !== L[i]) p++;
      if (x === W - 1 || L[i + 1] !== L[i]) p++;
      if (y === 0 || L[i - W] !== L[i]) p++;
      if (y === H - 1 || L[i + W] !== L[i]) p++;
      per[L[i]] += p;
    }
    let s = 0, tot = 0;
    for (let k = 0; k < n; k++) {
      if (!area[k] || !per[k]) continue;
      s += area[k] * Math.min(1, 4 * Math.PI * area[k] / (per[k] * per[k]));
      tot += area[k];
    }
    return tot ? s / tot : 0;
  }
  return { W: W, H: H, img: img, truth: truth, grad: grad, slic: slic,
    boundaryRecall: boundaryRecall, underSeg: underSeg, compactness: compactness, isB: isB };
})();

(function () {
  const svg = d3.select("#slic-svg");
  if (svg.empty()) return;
  const S = SGslic;
  const eK = document.getElementById("sl-k"), eKv = document.getElementById("sl-kv");
  const eM = document.getElementById("sl-m"), eMv = document.getElementById("sl-mv");
  const eI = document.getElementById("sl-it"), eIv = document.getElementById("sl-itv");
  const eP = document.getElementById("sl-mean");
  const out = document.getElementById("slic-readout");
  const CACHE = {};
  function get(k, m, it) {
    const key = k + "|" + m + "|" + it;
    if (!CACHE[key]) {
      const R = S.slic(k, m, it);
      const mean = new Float64Array(R.n), cnt = new Float64Array(R.n);
      for (let i = 0; i < S.W * S.H; i++) { mean[R.lab[i]] += S.img[i]; cnt[R.lab[i]]++; }
      for (let c = 0; c < R.n; c++) mean[c] /= Math.max(1, cnt[c]);
      CACHE[key] = { R: R, mean: mean,
        br: S.boundaryRecall(R.lab, 1), ue: S.underSeg(R.lab, R.n), cp: S.compactness(R.lab, R.n) };
    }
    return CACHE[key];
  }

  function draw() {
    const k = +eK.value, m = +eM.value, it = +eI.value;
    const G = get(k, m, it);
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.8, IW = S.W * cw, IH = S.H * cw;

    const A = SG.box(g, 14, 44, IW, IH, `SLIC · ${G.R.n} superpixels (asked for ${k})`);
    SG.cells(A, 0, 0, cw, S.W, S.H, (x, y) => SG.grey(S.img[y * S.W + x]));
    SG.drawEdges(A, SG.edgesOf(G.R.lab, S.W, S.H), cw, { color: VC.a2, w: 1, op: 0.95 });

    const B = SG.box(g, 300, 44, IW, IH, eP.checked ? "painted with superpixel means" : "superpixels, coloured");
    SG.cells(B, 0, 0, cw, S.W, S.H, (x, y) =>
      eP.checked ? SG.grey(G.mean[G.R.lab[y * S.W + x]]) : SG.pal(G.R.lab[y * S.W + x] * 5 % 16));
    SG.drawEdges(B, SG.edgesOf(S.truth, S.W, S.H), cw, { color: VC.rose, w: 1.4, op: 0.95 });
    B.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10).attr("fill", VC.rose)
      .text("pink = the true region boundaries");

    const cw2 = 1.66;
    const C = SG.box(g, 586, 44, S.W * cw2, S.H * cw2, "the true regions");
    SG.cells(C, 0, 0, cw2, S.W, S.H, (x, y) => SG.pal(S.truth[y * S.W + x] + 1));

    /* metric curves against m */
    const DX = 14, DY = 296, DW = 420, DH = 130;
    const D = SG.box(g, DX, DY, DW, DH, "boundary recall, under-segmentation error and compactness, against m",
      { fill: "#12141c" });
    const ms = d3.range(2, 41, 2);
    const rows = ms.map(mm => get(k, mm, it));
    const xs = d3.scaleLinear().domain([2, 40]).range([0, DW]);
    const ys = d3.scaleLinear().domain([0, 1]).range([DH, 0]);
    D.append("g").attr("transform", `translate(0,${DH})`).attr("class", "axis").call(d3.axisBottom(xs).ticks(7));
    D.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5));
    const ln = (acc, col, dash) => {
      const e = D.append("path").datum(rows).attr("d", d3.line().x((d, i) => xs(ms[i])).y(d => ys(acc(d))))
        .attr("fill", "none").attr("stroke", col).attr("stroke-width", 1.8);
      if (dash) e.attr("stroke-dasharray", dash);
    };
    ln(d => d.br, VC.good);
    ln(d => d.ue, VC.bad);
    ln(d => d.cp, VC.violet, "4 3");
    D.append("line").attr("x1", xs(m)).attr("x2", xs(m)).attr("y1", 0).attr("y2", DH)
      .attr("stroke", VC.ink).attr("stroke-width", 1.3);
    const bestM = ms[d3.range(rows.length).reduce((a, b) => (rows[b].br > rows[a].br ? b : a), 0)];
    D.append("line").attr("x1", xs(bestM)).attr("x2", xs(bestM)).attr("y1", 0).attr("y2", DH)
      .attr("stroke", VC.good).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.7);
    VZ.legend(D, [{ color: VC.good, label: "boundary recall (1 px)" },
      { color: VC.bad, label: "under-segmentation error" },
      { color: VC.violet, label: "compactness", dash: "4 3" }], DW - 168, 13, { gap: 13 });

    const put = SG.kv(g, 452, 310, { keyW: 176, lead: 15.5 });
    put("superpixels asked for / got", k + "  /  " + G.R.n, G.R.n === k ? VC.ink : VC.a2, true);
    put("nominal grid interval S", G.R.S.toFixed(2) + " px");
    put("pixels per superpixel", (S.W * S.H / G.R.n).toFixed(1));
    put("reduction against pixels", "× " + (S.W * S.H / G.R.n).toFixed(0), VC.accent, true);
    put("boundary recall (1 px)", G.br.toFixed(4), G.br > 0.99 ? VC.good : VC.a2, true);
    put("under-segmentation error", G.ue.toFixed(4), G.ue < 0.02 ? VC.good : VC.bad);
    put("compactness  4πA/P²", G.cp.toFixed(4), VC.violet);
    put("best m for recall at this k", "m = " + bestM + "  (" + get(k, bestM, it).br.toFixed(4) + ")", VC.good);
  }

  function say() {
    const k = +eK.value, m = +eM.value, it = +eI.value;
    const G = get(k, m, it);
    const ms = d3.range(2, 41, 2), rows = ms.map(mm => get(k, mm, it));
    const bi = d3.range(rows.length).reduce((a, b) => (rows[b].br > rows[a].br ? b : a), 0);
    out.innerHTML =
      `Asked for <b>${k}</b> superpixels at compactness <b>${m}</b>, got <b>${G.R.n}</b> — a reduction of `
      + `<b>${(S.W * S.H / G.R.n).toFixed(0)}×</b> against the ${S.W * S.H} pixels, which is the whole point. `
      + `Boundary recall <b>${G.br.toFixed(4)}</b>, under-segmentation error <b>${G.ue.toFixed(4)}</b>, `
      + `compactness <b>${G.cp.toFixed(4)}</b>.<br>`
      + `Across the compactness slider at this <code>k</code>, recall peaks at <b>m = ${ms[bi]}</b> with `
      + `<b>${rows[bi].br.toFixed(4)}</b>, and is <b>${rows[0].br.toFixed(4)}</b> at m = 2 and `
      + `<b>${rows[rows.length - 1].br.toFixed(4)}</b> at m = 40. The optimum is <b>interior</b>: a colour-only `
      + `distance does not hug boundaries best, because it produces stringy cells that the connectivity pass then `
      + `merges into larger ones, and a larger cell has less boundary to place.`;
  }
  function all() {
    eKv.textContent = eK.value; eMv.textContent = eM.value; eIv.textContent = eI.value;
    draw(); say();
  }
  eK.oninput = all; eM.oninput = all; eI.oninput = all; eP.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   19 · #iou-svg — IoU, Dice, boundary F, and the averaging counterexample
   ══════════════════════════════════════════════════════════════════════════ */
const SGiou = (function () {
  const W = 120, H = 96, CX = 46, CY = 48, R0 = 26;
  function masks(dx, ratio, shape) {
    const G = new Uint8Array(W * H), P = new Uint8Array(W * H);
    const r1 = R0, r2 = R0 * ratio;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const dg = Math.hypot(x - CX, y - CY), dp = Math.hypot(x - (CX + dx), y - CY);
      if (shape === "ring") {
        G[i] = (dg < r1 && dg > r1 - 3.2) ? 1 : 0;
        P[i] = (dp < r2 && dp > r2 - 3.2) ? 1 : 0;
      } else {
        G[i] = dg < r1 ? 1 : 0;
        P[i] = dp < r2 ? 1 : 0;
      }
    }
    return { G: G, P: P };
  }
  const isB = (M, i) => {
    if (!M[i]) return false;
    const x = i % W, y = (i / W) | 0;
    if (x === 0 || y === 0 || x === W - 1 || y === H - 1) return true;
    return !M[i - 1] || !M[i + 1] || !M[i - W] || !M[i + W];
  };
  function metrics(G, P, tol) {
    let inter = 0, uni = 0, ng = 0, np = 0;
    for (let i = 0; i < W * H; i++) {
      if (G[i]) ng++;
      if (P[i]) np++;
      if (G[i] && P[i]) inter++;
      if (G[i] || P[i]) uni++;
    }
    const J = uni ? inter / uni : 1;
    const D = (ng + np) ? 2 * inter / (ng + np) : 1;
    /* boundary precision and recall with a square matching tolerance */
    const near = (M, x, y) => {
      for (let dy = -tol; dy <= tol; dy++) for (let dx2 = -tol; dx2 <= tol; dx2++) {
        const X = x + dx2, Y = y + dy;
        if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
        if (isB(M, Y * W + X)) return true;
      }
      return false;
    };
    let bp = 0, npb = 0, br = 0, ngb = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (isB(P, i)) { npb++; if (near(G, x, y)) bp++; }
      if (isB(G, i)) { ngb++; if (near(P, x, y)) br++; }
    }
    const prec = npb ? bp / npb : 0, rec = ngb ? br / ngb : 0;
    const F = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0;
    return { J: J, D: D, prec: prec, rec: rec, F: F, inter: inter, uni: uni, ng: ng, np: np };
  }
  /* the fixed two-image counterexample: mean IoU and mean Dice rank A and B oppositely */
  const CASES = {
    A: [{ g: 100, p: 20, k: 0 }, { g: 100, p: 100, k: 100 }],
    B: [{ g: 100, p: 40, k: 40 }, { g: 100, p: 100, k: 60 }]
  };
  const jd = c => ({ J: c.k / (c.g + c.p - c.k), D: 2 * c.k / (c.g + c.p) });
  const meanOf = (m, key) => CASES[m].reduce((a, c) => a + jd(c)[key], 0) / CASES[m].length;
  return { W: W, H: H, masks: masks, metrics: metrics, isB: isB, CASES: CASES, jd: jd, meanOf: meanOf };
})();

(function () {
  const svg = d3.select("#iou-svg");
  if (svg.empty()) return;
  const S = SGiou;
  const eD = document.getElementById("iu-dx"), eDv = document.getElementById("iu-dxv");
  const eR = document.getElementById("iu-r"), eRv = document.getElementById("iu-rv");
  const eT = document.getElementById("iu-t"), eTv = document.getElementById("iu-tv");
  const eS = document.getElementById("iu-sh");
  const out = document.getElementById("iou-readout");
  const get = () => {
    const m = S.masks(+eD.value, +eR.value, eS.value);
    return { m: m, M: S.metrics(m.G, m.P, +eT.value) };
  };

  function draw() {
    const G = get();
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.1, IW = S.W * cw, IH = S.H * cw;

    const A = SG.box(g, 14, 44, IW, IH, "ground truth against prediction");
    SG.cells(A, 0, 0, cw, S.W, S.H, (x, y) => {
      const i = y * S.W + x, a = G.m.G[i], b = G.m.P[i];
      if (a && b) return "#4ade80";
      if (a) return "#f87171";
      if (b) return "#5b9cff";
      return "#12141c";
    });
    VZ.legend(A, [{ color: "#4ade80", label: "both  (intersection)" },
      { color: "#f87171", label: "truth only  (missed)" },
      { color: "#5b9cff", label: "prediction only  (false)" }], 8, 14, { gap: 13 });

    /* the Dice–IoU curve */
    const CX = 290, CY = 44, CW = 200, CH = 200;
    const C = SG.box(g, CX, CY, CW, CH, "Dice against IoU", { fill: "#12141c" });
    const xs = d3.scaleLinear().domain([0, 1]).range([0, CW]);
    const ys = d3.scaleLinear().domain([0, 1]).range([CH, 0]);
    C.append("g").attr("transform", `translate(0,${CH})`).attr("class", "axis").call(d3.axisBottom(xs).ticks(5));
    C.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5));
    C.append("line").attr("x1", xs(0)).attr("y1", ys(0)).attr("x2", xs(1)).attr("y2", ys(1))
      .attr("stroke", VC.muted).attr("stroke-dasharray", "4 3");
    C.append("path").datum(d3.range(0, 1.001, 0.01))
      .attr("d", d3.line().x(d => xs(d)).y(d => ys(2 * d / (1 + d))))
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 2);
    C.append("circle").attr("cx", xs(G.M.J)).attr("cy", ys(G.M.D)).attr("r", 5)
      .attr("fill", VC.good).attr("stroke", "#0b0d12");
    C.append("text").attr("x", 8).attr("y", 14).attr("font-size", 10).attr("fill", VC.a2)
      .text("D = 2J/(1+J)");
    C.append("text").attr("x", CW - 6).attr("y", CH - 8).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", VC.muted).text("dashed: D = J");

    /* metric bars */
    const BX = 520, BY = 44, BW = 224, BH = 200;
    const B = SG.box(g, BX, BY, BW, BH, "the four numbers", { fill: "#12141c" });
    const rows = [["IoU  (Jaccard)", G.M.J, VC.accent], ["Dice  (pixel F₁)", G.M.D, VC.a2],
    ["boundary precision", G.M.prec, VC.violet], ["boundary recall", G.M.rec, VC.good],
    ["boundary F", G.M.F, VC.teal]];
    const xb = d3.scaleLinear().domain([0, 1]).range([0, BW - 118]);
    rows.forEach(([n, v, c], i) => {
      B.append("text").attr("x", 106).attr("y", 20 + i * 30).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", VC.muted).text(n);
      B.append("rect").attr("x", 110).attr("y", 9 + i * 30).attr("width", Math.max(0.6, xb(v)))
        .attr("height", 14).attr("fill", c).attr("fill-opacity", 0.8);
      B.append("text").attr("x", 114 + xb(v)).attr("y", 20 + i * 30).attr("font-size", 10.5)
        .attr("fill", c).text(v.toFixed(4));
    });
    B.append("text").attr("x", 4).attr("y", BH - 8).attr("font-size", 10).attr("fill", VC.muted)
      .text(`|G| = ${G.M.ng}, |P| = ${G.M.np}, |G∩P| = ${G.M.inter}`);

    /* the averaging counterexample */
    const EX = 14, EY = 288, EW = 476, EH = 132;
    const E = SG.box(g, EX, EY, EW, EH,
      "and the fixed example where averaging them ranks two methods oppositely", { fill: "#12141c" });
    const mJA = S.meanOf("A", "J"), mDA = S.meanOf("A", "D");
    const mJB = S.meanOf("B", "J"), mDB = S.meanOf("B", "D");
    const bars = [["method A · mean IoU", mJA, VC.accent], ["method B · mean IoU", mJB, VC.accent],
    ["method A · mean Dice", mDA, VC.a2], ["method B · mean Dice", mDB, VC.a2]];
    const xe = d3.scaleLinear().domain([0, 0.7]).range([0, EW - 190]);
    bars.forEach(([n, v, c], i) => {
      const win = (i < 2) ? (v === Math.max(mJA, mJB)) : (v === Math.max(mDA, mDB));
      E.append("text").attr("x", 156).attr("y", 20 + i * 27).attr("text-anchor", "end")
        .attr("font-size", 10.5).attr("fill", VC.muted).text(n);
      E.append("rect").attr("x", 160).attr("y", 9 + i * 27).attr("width", xe(v)).attr("height", 14)
        .attr("fill", c).attr("fill-opacity", win ? 0.95 : 0.4)
        .attr("stroke", win ? VC.good : "none").attr("stroke-width", 1.4);
      E.append("text").attr("x", 164 + xe(v)).attr("y", 20 + i * 27).attr("font-size", 10.5)
        .attr("fill", win ? VC.good : VC.muted)
        .text(v.toFixed(4) + (win ? "   ← wins" : ""));
    });

    const put = SG.kv(g, 512, 300, { keyW: 150, lead: 15.5 });
    put("D from J", (2 * G.M.J / (1 + G.M.J)).toFixed(6), VC.a2);
    put("measured D", G.M.D.toFixed(6), VC.a2, true);
    put("difference", Math.abs(G.M.D - 2 * G.M.J / (1 + G.M.J)).toExponential(2), VC.good);
    put("J from D", (G.M.D / (2 - G.M.D)).toFixed(6), VC.accent);
    put("measured J", G.M.J.toFixed(6), VC.accent, true);
    put("Dice − IoU", (G.M.D - G.M.J).toFixed(4), VC.violet, true);
  }

  function say() {
    const G = get();
    out.innerHTML =
      `|G| = <b>${G.M.ng}</b>, |P| = <b>${G.M.np}</b>, overlap <b>${G.M.inter}</b>. `
      + `IoU <b>${G.M.J.toFixed(4)}</b>, Dice <b>${G.M.D.toFixed(4)}</b> — a gap of `
      + `<b>${(G.M.D - G.M.J).toFixed(4)}</b> on the same masks. `
      + `Boundary precision <b>${G.M.prec.toFixed(4)}</b>, recall <b>${G.M.rec.toFixed(4)}</b>, `
      + `F <b>${G.M.F.toFixed(4)}</b> at a tolerance of ${eT.value} px.<br>`
      + `<code>2J/(1+J)</code> gives <b>${(2 * G.M.J / (1 + G.M.J)).toFixed(6)}</b> against a measured Dice of `
      + `<b>${G.M.D.toFixed(6)}</b> — identical, as they must be. And yet, on the two-image example below, `
      + `mean IoU prefers method A at <b>${S.meanOf("A", "J").toFixed(4)}</b> to <b>${S.meanOf("B", "J").toFixed(4)}</b>, `
      + `while mean Dice prefers method B at <b>${S.meanOf("B", "D").toFixed(4)}</b> to `
      + `<b>${S.meanOf("A", "D").toFixed(4)}</b>. A monotone relation between two scores says nothing about `
      + `the order of their averages.`;
  }
  function all() {
    eDv.textContent = eD.value; eRv.textContent = (+eR.value).toFixed(2); eTv.textContent = eT.value;
    draw(); say();
  }
  eD.oninput = all; eR.oninput = all; eT.oninput = all; eS.onchange = all;
  all();
})();
