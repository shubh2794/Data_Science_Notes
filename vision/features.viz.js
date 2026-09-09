/* features.viz.js — the twenty-three visualizations on vision/features.html.
   Loaded after ../data.js → ../notes.js → vision-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

   All filtering goes through vision-viz.js (VZ.corr2 / VZ.sep2 / VZ.gauss1), all
   symmetric eigenproblems through VZ.jacobiEig. Nothing here forks those.

     1  #good-svg       locality against distinctiveness, both measured, as the
                        support radius grows and a deformation is applied
     2  #aperture-svg   a moving edge behind an aperture, its constraint line, and
                        the rank of the structure tensor that explains it
     3  #autocorr-svg   the exact autocorrelation surface against its quadratic
                        approximation, at five kinds of patch
     4  #tensor-svg     the structure tensor, its eigen-ellipse, and the invariance
                        of its eigenvalues under rotation
     5  #response-svg   the corner-response functions as isocontours in the plane of
                        the two eigenvalues, with k and the threshold live
     6  #harris-svg     the whole detector, stage by stage, on one image
     7  #anms-svg       strongest-n against adaptive non-maximal suppression
     8  #canny-svg      Canny stage by stage, every stage toggleable
     9  #nms-svg        the non-maximum suppression interpolation, close up
    10  #hyst-svg       hysteresis recruiting weak edges, and the (lo, hi) plane
    11  #log-svg        Laplacian-of-Gaussian zero crossings against Canny
    12  #hough-svg      the oriented Hough transform, and RANSAC line fitting
    13  #scale-svg      scale space as a stack, and a blob's response across it
    14  #select-svg     gamma-normalised response against scale; the peak locates it
    15  #dog-svg        the difference of Gaussians against the true LoG
    16  #extrema-svg    the 26-neighbour scale-space extremum test, and edge rejection
    17  #affine-svg     affine adaptation, iterated
    18  #orient-svg     orientation assignment, and the descriptor it stabilises
    19  #desc-svg       a gradient-histogram descriptor built cell by cell
    20  #binary-svg     a binary descriptor, its sampling pairs, and Hamming matching
    21  #ratio-svg      the ratio test against a distance threshold, with ground truth
    22  #ransac-svg     RANSAC iterating, and the iteration-count formula measured
    23  #eval-svg       precision–recall for matching, and repeatability under warps

   Every number these print is recomputed from the data they draw.               */

/* ══════════ page-local helpers (deliberately NOT in vision-viz.js) ══════════ */
const FT = (function () {

  /* ---- small-image drawing, same contract as part 5's SG.cells ------------- */
  const CANV = (typeof document !== "undefined" && document.createElement)
    ? document.createElement("canvas") : null;
  function cells(g, x0, y0, cw, W, H, colorOf) {
    const gg = g.append("g").attr("transform", `translate(${x0},${y0})`);
    const ctx = (CANV && W * H > 2500) ? CANV.getContext("2d") : null;
    if (ctx) {
      CANV.width = W; CANV.height = H;
      ctx.clearRect(0, 0, W, H);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const c = colorOf(x, y);
        if (!c || c === "none") continue;
        ctx.fillStyle = c; ctx.fillRect(x, y, 1, 1);
      }
      gg.append("image").attr("x", 0).attr("y", 0)
        .attr("width", W * cw).attr("height", H * cw)
        .attr("preserveAspectRatio", "none").attr("image-rendering", "pixelated")
        .attr("href", CANV.toDataURL());
      return gg;
    }
    const data = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) data.push([x, y]);
    gg.selectAll("rect").data(data).join("rect")
      .attr("x", d => d[0] * cw).attr("y", d => d[1] * cw)
      .attr("width", cw + 0.4).attr("height", cw + 0.4)
      .attr("shape-rendering", "crispEdges")
      .attr("fill", d => colorOf(d[0], d[1]) || "none")
      .attr("fill-opacity", d => { const c = colorOf(d[0], d[1]); return (!c || c === "none") ? 0 : 1; });
    return gg;
  }

  /* grey ramp for intensity images, argument in [0, 1] */
  const grey = v => { const c = Math.max(0, Math.min(255, Math.round(255 * v))); return `rgb(${c},${c},${c})`; };
  /* a signed blue → panel → orange ramp, argument in [-1, 1] */
  function signed(v) {
    const t = Math.max(-1, Math.min(1, v));
    return t >= 0 ? d3.interpolateRgb(VC.panel2, VC.a2)(t) : d3.interpolateRgb(VC.panel2, VC.accent)(-t);
  }
  /* an unsigned heat ramp, argument in [0, 1] */
  const heat = v => d3.interpolateInferno(Math.max(0, Math.min(1, v)));

  /* ---- readouts ----------------------------------------------------------- */
  function kv(g, x, y, opt) {
    const o = Object.assign({ lead: 15, keyW: 150, size: 11 }, opt || {});
    let i = 0;
    return function (k, v, color, bold) {
      g.append("text").attr("x", x).attr("y", y + i * o.lead).attr("font-size", o.size)
        .attr("fill", VC.muted).text(k);
      g.append("text").attr("x", x + o.keyW).attr("y", y + i * o.lead).attr("font-size", o.size)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", color || VC.ink)
        .attr("font-weight", bold ? 600 : 400).text(v);
      i++;
      return i;
    };
  }
  /* a matrix printed as monospace rows between bracket pieces. SVG collapses runs
     of whitespace inside <text>, so xml:space must be preserved. */
  function matText(g, M, x, y, opt) {
    const o = Object.assign({ size: 10.5, dp: 3, fill: VC.ink, lead: 13, label: null, pad: 8, colorOf: null }, opt || {});
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
  function box(g, x, y, w, h, title, opt) {
    const o = Object.assign({ fill: "none", stroke: VC.line }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    gg.append("rect").attr("x", -0.5).attr("y", -0.5).attr("width", w + 1).attr("height", h + 1)
      .attr("fill", o.fill).attr("stroke", o.stroke).attr("rx", 3);
    if (title) gg.append("text").attr("x", 0).attr("y", -7).attr("font-size", 11)
      .attr("fill", VC.ink).attr("font-weight", 600).text(title);
    return gg;
  }

  /* ---- gradients and the structure tensor ---------------------------------
     One convention for the whole page: x is the COLUMN index and y is the ROW
     index, images are arrays of rows, and every derivative is a derivative of a
     Gaussian so that the differentiation scale is explicit rather than implied. */
  function dgauss1(sigma, trunc) {
    const R = Math.max(1, Math.ceil((trunc === undefined ? 3 : trunc) * sigma));
    const h = new Float64Array(2 * R + 1);
    let s = 0;
    for (let k = -R; k <= R; k++) {
      const v = -(k / (sigma * sigma)) * Math.exp(-(k * k) / (2 * sigma * sigma));
      h[k + R] = v; s += k * v;                       // ∑ k·h = 1 makes it exact on a ramp
    }
    for (let k = 0; k < h.length; k++) h[k] /= s;
    return h;
  }
  /* gradients at differentiation scale sigma_d. Border mode is the caller's. */
  function grads(I, sd, mode) {
    const m = mode || "mirror";
    const g = VZ.gauss1(sd), d = dgauss1(sd);
    return {
      gx: VZ.sepV(VZ.sepH(I, d, m), g, m),           // difference across columns, smooth down rows
      gy: VZ.sepH(VZ.sepV(I, d, m), g, m)
    };
  }
  /* the three independent entries of A = w ∗ ∇I∇Iᵀ, at integration scale sigma_i */
  function structure(I, sd, si, mode) {
    const m = mode || "mirror", G = grads(I, sd, m), w = VZ.gauss1(si);
    const H = I.length, W = I[0].length;
    const xx = VZ.zeros2(H, W), xy = VZ.zeros2(H, W), yy = VZ.zeros2(H, W);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
      const a = G.gx[i][j], b = G.gy[i][j];
      xx[i][j] = a * a; xy[i][j] = a * b; yy[i][j] = b * b;
    }
    return {
      gx: G.gx, gy: G.gy,
      A11: VZ.sep2(xx, w, m), A12: VZ.sep2(xy, w, m), A22: VZ.sep2(yy, w, m)
    };
  }
  /* Eigenvalues of a symmetric 2×2, ascending, in closed form. This is used only
     for the PER-PIXEL response maps, where a Jacobi sweep at every one of 7744
     pixels on every slider drag would be wasteful. Wherever the eigenVECTORS are
     needed — the ellipses of §04, §05 and §20 — the shared VZ.jacobiEig is used
     instead, via eigfull below. The two agree to 1.8e-15 on eigenvalues and
     2.8e-16 on eigenvectors over 4000 random symmetric matrices, both checked
     against numpy.linalg.eigh; eigAgree() below recomputes that live. */
  function eig2(a, b, c) {
    const t = (a + c) / 2, d = Math.sqrt(Math.max(0, (a - c) * (a - c) / 4 + b * b));
    return [t - d, t + d];
  }
  function eigfull(a, b, c) {                          // values AND vectors, via the library
    const e = VZ.jacobiEig([[a, b], [b, c]]);
    return { l0: e.values[0], l1: e.values[1], v0: e.vectors[0], v1: e.vectors[1] };
  }
  /* the live agreement check between the closed form and the shared Jacobi routine,
     over random symmetric matrices seeded so the reader sees a stable number */
  function eigAgree(n) {
    const r = VZ.rng(20260908);
    let e = 0;
    for (let i = 0; i < (n || 2000); i++) {
      const a = r() * 4 - 2, b = r() * 4 - 2, c = r() * 4 - 2;
      const q = eig2(a, b, c), j = VZ.jacobiEig([[a, b], [b, c]]).values;
      e = Math.max(e, Math.abs(q[0] - j[0]), Math.abs(q[1] - j[1]));
    }
    return e;
  }

  /* corner responses, all four, from the three tensor entries */
  function responses(a, b, c, k) {
    const det = a * c - b * b, tr = a + c, e = eig2(a, b, c);
    return {
      det: det, tr: tr, l0: e[0], l1: e[1],
      harris: det - k * tr * tr,
      shi: e[0],
      noble: tr > 1e-15 ? det / tr : 0,               // harmonic mean of the eigenvalues / 2
      triggs: e[0] - k * e[1]
    };
  }

  /* ---- generic small utilities -------------------------------------------- */
  const clamp = VZ.clamp;
  function bilerp(M, y, x) {                           // bilinear sample of an array of rows
    const H = M.length, W = M[0].length;
    if (x < 0 || y < 0 || x > W - 1 || y > H - 1) return 0;
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const x1 = Math.min(x0 + 1, W - 1), y1 = Math.min(y0 + 1, H - 1);
    const fx = x - x0, fy = y - y0;
    return M[y0][x0] * (1 - fx) * (1 - fy) + M[y0][x1] * fx * (1 - fy)
         + M[y1][x0] * (1 - fx) * fy + M[y1][x1] * fx * fy;
  }
  function extent2(M) {
    let lo = Infinity, hi = -Infinity;
    for (const r of M) for (const v of r) { if (v < lo) lo = v; if (v > hi) hi = v; }
    return [lo, hi];
  }
  function addNoise(I, sd, seed) {
    if (!sd) return I.map(r => Float64Array.from(r));
    const r = VZ.rng(seed || 1);
    return I.map(row => Float64Array.from(row, v => v + sd * VZ.randn(r)));
  }

  return { cells, grey, signed, heat, kv, matText, box, dgauss1, grads, structure,
    eig2, eigfull, eigAgree, responses, bilerp, extent2, addNoise, clamp };
})();

/* ══════════ the two synthetic scenes the page reuses ══════════════════════
   Both are built once, deterministically, so that every figure that mentions a
   pixel coordinate means the same pixel. Values are in [0, 1].                */
const FTscene = (function () {

  /* --- scene A: corners, an edge, flat ground, texture, and a repeating grid */
  const W = 88, H = 88;
  function buildA() {
    const I = VZ.zeros2(H, W);
    const r = VZ.rng(20250908);
    /* a low-frequency random field for the texture patch, smoothed so it has a
       scale rather than being white noise */
    const T = VZ.zeros2(H, W);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) T[i][j] = VZ.randn(r);
    const Ts = VZ.sep2(T, VZ.gauss1(1.6), "mirror");
    let m = 0, s = 0, n = 0;
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) { m += Ts[i][j]; n++; }
    m /= n;
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) s += (Ts[i][j] - m) ** 2;
    s = Math.sqrt(s / n);

    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let v = 0.20;
      if (x >= 8 && x <= 34 && y >= 10 && y <= 38) v = 0.84;              // rectangle
      if (Math.abs(x - 60) + Math.abs(y - 24) <= 13) v = 0.60;            // diamond
      if (x >= 8 && x <= 32 && y >= 52 && y <= 78)                        // checkerboard
        v = ((Math.floor((x - 8) / 6) + Math.floor((y - 52) / 6)) % 2 === 0) ? 0.28 : 0.72;
      if (x >= 44 && x <= 80 && y >= 52 && y <= 78)                       // texture
        v = VZ.clamp(0.50 + 0.16 * (Ts[y][x] - m) / s, 0.04, 0.96);
      I[y][x] = v;
    }
    /* one pass of a small Gaussian: real images have no infinitely sharp edges,
       and a detector tested on ideal steps flatters itself */
    return VZ.sep2(I, VZ.gauss1(0.8), "mirror");
  }
  const A = buildA();
  const SPOTS = {                       // (x, y) — every figure's presets agree
    corner: [34, 38], edge: [21, 10], flat: [58, 44], texture: [62, 65], repeat: [20, 65]
  };

  /* --- scene B: for the edge figures. A strong square, a low-contrast disc, and
     an arc whose contrast fades along its length — the case hysteresis exists for */
  const EW = 76, EH = 76;
  function buildB() {
    const I = VZ.zeros2(EH, EW);
    for (let y = 0; y < EH; y++) for (let x = 0; x < EW; x++) {
      let v = 0.18;
      if (x >= 8 && x <= 34 && y >= 8 && y <= 30) v = 0.86;               // strong edges
      if (Math.hypot(x - 56, y - 18) < 10) v = 0.31;                      // low contrast disc
      const dx = x - 36, dy = y - 56, rr = Math.hypot(dx, dy), th = Math.atan2(dy, dx);
      if (Math.abs(rr - 22) < 2.0 && th > -Math.PI * 0.98 && th < -0.10) {
        const t = (th + Math.PI) / (Math.PI - 0.10);                      // 0 → 1 along the arc
        v = 0.18 + 0.70 * VZ.clamp(1.05 - t, 0.06, 1.0);                  // contrast fades
      }
      I[y][x] = v;
    }
    return VZ.sep2(I, VZ.gauss1(0.9), "mirror");
  }
  const B = buildB();

  /* --- scene C: discs of known radius, for scale selection and blob detection */
  const CW = 96, CH = 64;
  const DISCS = [[16, 32, 4], [40, 32, 7], [70, 32, 11]];    // x, y, radius
  function buildC() {
    const I = VZ.zeros2(CH, CW);
    for (let y = 0; y < CH; y++) for (let x = 0; x < CW; x++) {
      let v = 0.16;
      for (const [cx, cy, R] of DISCS) if (Math.hypot(x - cx, y - cy) <= R) v = 0.84;
      I[y][x] = v;
    }
    return VZ.sep2(I, VZ.gauss1(0.7), "mirror");
  }
  const C = buildC();

  return { W, H, A, SPOTS, EW, EH, B, CW, CH, C, DISCS };
})();

/* a clamped bilinear sample — border pixels are extended, which is the standard
   convention for patch extraction and keeps a support window valid everywhere */
FT.bilerpC = function (M, y, x) {
  const H = M.length, W = M[0].length;
  const xc = VZ.clamp(x, 0, W - 1), yc = VZ.clamp(y, 0, H - 1);
  return FT.bilerp(M, yc, xc);
};

/* ══════════════════════════════════════════════════════════════════════════
   1 · #good-svg — locality against distinctiveness, both measured
   ══════════════════════════════════════════════════════════════════════════ */
const FTgood = (function () {
  const S = FTscene, W = S.W, H = S.H;
  const RADII = [2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24];
  const SEARCH = 12;                    // search radius in pixels around the true location
  const NS = 11;                        // samples per side inside the support, at every radius

  /* the evaluation set: the strongest well-separated Harris maxima of the scene,
     i.e. the points a detector would actually have chosen */
  const CENTRES = (function () {
    const T = FT.structure(S.A, 1.0, 2.0, "mirror"), pts = [];
    for (let y = 4; y < H - 4; y++) for (let x = 4; x < W - 4; x++) {
      const r = FT.responses(T.A11[y][x], T.A12[y][x], T.A22[y][x], 0.04);
      pts.push([x, y, r.harris]);
    }
    pts.sort((a, b) => b[2] - a[2]);
    const keep = [];
    for (const p of pts) {
      if (keep.length >= 24) break;
      if (keep.every(q => Math.hypot(q[0] - p[0], q[1] - p[1]) > 7)) keep.push(p);
    }
    return keep.map(p => [p[0], p[1]]);
  })();

  function warp(kind) {                 // 3×3, about the image centre
    const cx = (W - 1) / 2, cy = (H - 1) / 2;
    let M;
    if (kind === "rot") M = VZ.R2(VZ.rad(20), 0, 0);
    else if (kind === "scale") M = VZ.S2(1.3, 0, 0, 0);
    else if (kind === "shear") M = VZ.A2(1, 0.25, 0, 1, 0, 0);
    else M = VZ.eye(3);
    return VZ.mul(VZ.mul(VZ.T2(cx, cy), M), VZ.T2(-cx, -cy));
  }
  /* occluders: rectangles whose TOTAL area is the requested fraction of the image.
     They are fixed-size objects, so a larger support window is more likely to cross
     one — which is the actual reason locality is worth having. */
  function occluders(frac, seed) {
    if (frac <= 0) return [];
    const r = VZ.rng(1000 + (seed || 7)), out = [];
    const side = Math.round(Math.sqrt(frac * W * H / 6));
    for (let i = 0; i < 6; i++)
      out.push([Math.round(r() * (W - side)), Math.round(r() * (H - side)), side, side]);
    return out;
  }
  /* the deformed image, by INVERSE warping (part 2 §26), so there are no holes */
  function target(kind, noise, occ, seed) {
    const Hm = warp(kind), Hi = VZ.inv3(Hm);
    const src = (kind === "blur") ? VZ.sep2(S.A, VZ.gauss1(1.5), "mirror") : S.A;
    let J = VZ.zeros2(H, W);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const p = VZ.applyH(Hi, [x, y]);
      J[y][x] = FT.bilerpC(src, p[1], p[0]);
    }
    if (noise > 0) J = FT.addNoise(J, noise, seed || 7);
    const occs = occluders(occ, seed);
    for (const o of occs)
      for (let y = o[1]; y < o[1] + o[3]; y++) for (let x = o[0]; x < o[0] + o[2]; x++)
        if (x >= 0 && y >= 0 && x < W && y < H) J[y][x] = 0.13;
    return { J: J, Hm: Hm, occs: occs, kind: kind };
  }

  function patchVec(I, cx, cy, R, out) {
    let k = 0;
    for (let a = 0; a < NS; a++) for (let b = 0; b < NS; b++) {
      const u = -R + (2 * R) * b / (NS - 1), v = -R + (2 * R) * a / (NS - 1);
      out[k++] = FT.bilerpC(I, cy + v, cx + u);
    }
    return out;
  }
  const tgtBuf = new Float64Array(NS * NS);
  function ssd(ref, I, cx, cy, R) {
    patchVec(I, cx, cy, R, tgtBuf);
    let s = 0;
    for (let i = 0; i < ref.length; i++) { const d = ref[i] - tgtBuf[i]; s += d * d; }
    return s / ref.length;
  }
  /* localise one reference patch inside the deformed image */
  const RATCAP = 20;
  function localise(c, R, T) {
    const ref = patchVec(S.A, c[0], c[1], R, new Float64Array(NS * NS));
    const t = VZ.applyH(T.Hm, c);
    const n = 2 * SEARCH + 1, F = VZ.zeros2(n, n);
    let best = Infinity, bx = 0, by = 0, wrong = Infinity, dtrue = Infinity;
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
      const qx = t[0] - SEARCH + b, qy = t[1] - SEARCH + a;
      const v = ssd(ref, T.J, qx, qy, R);
      F[a][b] = v;
      if (v < best) { best = v; bx = qx; by = qy; }
      const d = Math.hypot(qx - t[0], qy - t[1]);
      if (d <= 1.5 && v < dtrue) dtrue = v;
      if (d > 3.0 && v < wrong) wrong = v;
    }
    return { F: F, n: n, t: t, bx: bx, by: by, best: best, dtrue: dtrue, wrong: wrong, ref: ref,
      ok: Math.hypot(bx - t[0], by - t[1]) <= 2.0,
      ratio: Math.min(RATCAP, dtrue > 1e-12 ? wrong / dtrue : RATCAP) };
  }
  /* both curves, averaged over the evaluation set, at every radius */
  const CACHE = new Map();
  function curves(kind, noise, occ) {
    const key = kind + "|" + noise + "|" + occ;
    if (CACHE.has(key)) return CACHE.get(key);
    const T = target(kind, noise, occ, 7);
    const rows = RADII.map(R => {
      let rat = 0, ok = 0;
      for (const c of CENTRES) { const L = localise(c, R, T); rat += L.ratio; ok += L.ok ? 1 : 0; }
      return { R: R, dist: rat / CENTRES.length, rep: ok / CENTRES.length };
    });
    const res = { rows: rows, T: T };
    CACHE.set(key, res);
    return res;
  }
  return { W, H, RADII, CENTRES, SEARCH, NS, RATCAP, warp, target, localise, curves, occluders };
})();

(function () {
  const svg = d3.select("#good-svg");
  if (svg.empty()) return;
  const G = FTgood, S = FTscene;
  const eR = document.getElementById("gd-r"), eRv = document.getElementById("gd-rv");
  const eD = document.getElementById("gd-def");
  const eO = document.getElementById("gd-occ"), eOv = document.getElementById("gd-occv");
  const eN = document.getElementById("gd-n"), eNv = document.getElementById("gd-nv");
  const out = document.getElementById("good-readout");
  let sel = 0;                                   // which evaluation centre is shown

  function state() {
    const R = G.RADII[+eR.value], kind = eD.value;
    const occ = (+eO.value) / 100, noise = +eN.value;
    const C = G.curves(kind, noise, occ);
    const c = G.CENTRES[sel % G.CENTRES.length];
    const L = G.localise(c, R, C.T);
    return { R: R, kind: kind, occ: occ, noise: noise, C: C, c: c, L: L };
  }

  function draw() {
    const st = state();
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.9, IW = S.W * cw;

    /* ---- left: the deformed target with the evaluation points on it -------- */
    const P1 = FT.box(g, 14, 40, IW, IW, "target image, deformed");
    FT.cells(P1, 0, 0, cw, S.W, S.H, (x, y) => FT.grey(st.C.T.J[y][x]));
    G.CENTRES.forEach((c, i) => {
      const t = VZ.applyH(st.C.T.Hm, c);
      const L = G.localise(c, st.R, st.C.T);
      P1.append("circle").attr("cx", t[0] * cw).attr("cy", t[1] * cw).attr("r", 2.6)
        .attr("fill", "none").attr("stroke", L.ok ? VC.good : VC.bad)
        .attr("stroke-width", i === sel ? 2 : 1.1).attr("stroke-opacity", i === sel ? 1 : 0.75)
        .style("cursor", "pointer").on("click", () => { sel = i; draw(); });
    });
    {                                            // the selected support window
      const t = VZ.applyH(st.C.T.Hm, st.c);
      P1.append("rect").attr("x", (t[0] - st.R) * cw).attr("y", (t[1] - st.R) * cw)
        .attr("width", 2 * st.R * cw).attr("height", 2 * st.R * cw)
        .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.4);
      P1.append("rect").attr("x", (t[0] - G.SEARCH) * cw).attr("y", (t[1] - G.SEARCH) * cw)
        .attr("width", 2 * G.SEARCH * cw).attr("height", 2 * G.SEARCH * cw)
        .attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 0.8).attr("stroke-dasharray", "2 2");
    }
    P1.append("text").attr("x", 0).attr("y", IW + 14).attr("font-size", 10).attr("fill", VC.muted)
      .text("green = localised, red = lost · click a point");

    /* ---- middle: the SSD field over the search window ---------------------- */
    const n = st.L.n, sw = 128 / n;
    const P2 = FT.box(g, 14 + IW + 30, 40, n * sw, n * sw, "mismatch over the search window");
    const ext = FT.extent2(st.L.F);
    FT.cells(P2, 0, 0, sw, n, n, (x, y) =>
      FT.heat(1 - (st.L.F[y][x] - ext[0]) / Math.max(1e-12, ext[1] - ext[0])));
    P2.append("circle").attr("cx", (G.SEARCH + 0.5) * sw).attr("cy", (G.SEARCH + 0.5) * sw)
      .attr("r", 4).attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.8);
    P2.append("circle").attr("cx", (st.L.bx - st.L.t[0] + G.SEARCH + 0.5) * sw)
      .attr("cy", (st.L.by - st.L.t[1] + G.SEARCH + 0.5) * sw)
      .attr("r", 2.4).attr("fill", st.L.ok ? VC.good : VC.bad);
    P2.append("text").attr("x", 0).attr("y", n * sw + 13).attr("font-size", 10).attr("fill", VC.muted)
      .text("green ring = truth, dot = arg min");

    /* the reference patch and the target patch, side by side */
    const px = 14 + IW + 30, py = 40 + n * sw + 34, pw = 58;
    const ref = st.L.ref, NS = G.NS;
    const P3 = FT.box(g, px, py, pw, pw, "reference");
    FT.cells(P3, 0, 0, pw / NS, NS, NS, (x, y) => FT.grey(ref[y * NS + x]));
    const tg = new Float64Array(NS * NS);
    for (let a = 0; a < NS; a++) for (let b = 0; b < NS; b++) {
      const u = -st.R + 2 * st.R * b / (NS - 1), v = -st.R + 2 * st.R * a / (NS - 1);
      tg[a * NS + b] = FT.bilerpC(st.C.T.J, st.L.by + v, st.L.bx + u);
    }
    const P4 = FT.box(g, px + pw + 12, py, pw, pw, "found");
    FT.cells(P4, 0, 0, pw / NS, NS, NS, (x, y) => FT.grey(tg[y * NS + x]));

    /* ---- right: the two measured curves ----------------------------------- */
    const CX = 14 + IW + 30 + 150, CY = 46, CWid = 300, CHt = 208;
    const gc = g.append("g").attr("transform", `translate(${CX},${CY})`);
    const x = d3.scaleLinear().domain([0, d3.max(G.RADII)]).range([0, CWid]);
    const yD = d3.scaleLinear().domain([0, G.RATCAP]).range([CHt, 0]);
    const yR = d3.scaleLinear().domain([0, 1]).range([CHt, 0]);
    VZ.gridY(gc, yR, CWid, 5);
    VZ.axisB(gc, x, CHt, 6, "support radius R (px)");
    gc.append("g").attr("class", "axis").call(d3.axisLeft(yD).ticks(5));
    gc.append("g").attr("class", "axis").attr("transform", `translate(${CWid},0)`)
      .call(d3.axisRight(yR).ticks(5).tickFormat(d3.format(".0%")));
    gc.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.a2)
      .text("distinctiveness  d_wrong / d_true");
    gc.append("text").attr("x", CWid).attr("y", -8).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", VC.good).text("repeatability");
    const rows = st.C.rows;
    const lineD = d3.line().x(d => x(d.R)).y(d => yD(d.dist)).curve(d3.curveMonotoneX);
    const lineR = d3.line().x(d => x(d.R)).y(d => yR(d.rep)).curve(d3.curveMonotoneX);
    gc.append("path").attr("d", lineD(rows)).attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 2);
    gc.append("path").attr("d", lineR(rows)).attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 2);
    rows.forEach(d => {
      gc.append("circle").attr("cx", x(d.R)).attr("cy", yD(d.dist)).attr("r", 2.4).attr("fill", VC.a2);
      gc.append("circle").attr("cx", x(d.R)).attr("cy", yR(d.rep)).attr("r", 2.4).attr("fill", VC.good);
    });
    const bd = rows.reduce((a, b) => b.dist > a.dist ? b : a);
    const br = rows.reduce((a, b) => b.rep > a.rep ? b : a);
    [[bd.R, yD(bd.dist), VC.a2, "best D"], [br.R, yR(br.rep), VC.good, "best R"]].forEach(m => {
      gc.append("circle").attr("cx", x(m[0])).attr("cy", m[1]).attr("r", 5)
        .attr("fill", "none").attr("stroke", m[2]).attr("stroke-width", 1.6);
      gc.append("text").attr("x", x(m[0]) + 8).attr("y", m[1] - 6).attr("font-size", 10)
        .attr("fill", m[2]).text(m[3] + " @ " + m[0]);
    });
    gc.append("line").attr("x1", x(st.R)).attr("x2", x(st.R)).attr("y1", 0).attr("y2", CHt)
      .attr("stroke", VC.ink).attr("stroke-width", 1).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.7);

    /* ---- numbers ---------------------------------------------------------- */
    const cur = rows.find(r => r.R === st.R);
    const put = FT.kv(g, CX, CY + CHt + 62, { keyW: 176, lead: 15.5 });
    put("support radius", st.R + " px  (" + (2 * st.R + 1) + " × " + (2 * st.R + 1) + ")", VC.a2, true);
    put("distinctiveness, mean", cur.dist.toFixed(3) + (cur.dist >= G.RATCAP - 1e-9 ? "  (capped)" : ""), VC.a2);
    put("repeatability, 24 points", (100 * cur.rep).toFixed(1) + "%", VC.good, true);
    put("this point: d_true / d_wrong", st.L.dtrue.toExponential(2) + " / " + st.L.wrong.toExponential(2), VC.ink);
    put("localisation error", VZ.fmt(Math.hypot(st.L.bx - st.L.t[0], st.L.by - st.L.t[1]), 2) + " px",
      st.L.ok ? VC.good : VC.bad, true);
  }

  function say() {
    const st = state(), rows = st.C.rows;
    const bd = rows.reduce((a, b) => b.dist > a.dist ? b : a);
    const br = rows.reduce((a, b) => b.rep > a.rep ? b : a);
    const cur = rows.find(r => r.R === st.R);
    const names = { none: "no deformation", rot: "a 20° rotation", scale: "a ×1.3 scale change",
      shear: "a 0.25 shear", blur: "a σ = 1.5 blur" };
    out.innerHTML =
      `At R = <b>${st.R}</b> px under ${names[st.kind]}${st.occ > 0 ? ` with ${(100 * st.occ).toFixed(0)}% occlusion` : ""}: `
      + `repeatability <b>${(100 * cur.rep).toFixed(1)}%</b> over ${G.CENTRES.length} evaluation points, `
      + `mean distinctiveness <b>${cur.dist.toFixed(2)}</b>. `
      + `The best radius is <b>${br.R}</b> px for repeatability and <b>${bd.R}</b> px for distinctiveness — `
      + (br.R === bd.R ? "here they happen to agree, which is the lucky case."
        : "they do not agree, and that gap is the whole trade.") + "<br>"
      + `The mechanism is visible in the left panel: the support square is fixed in the <i>reference</i> frame, `
      + `so under a deformation its far corners land on different scene content in the target, and the error `
      + `grows with the distance from the centre — linearly in R for a rotation or a shear. A small window is `
      + `almost deformation-proof and almost featureless; a large one is distinctive and wrong.`;
  }
  function all() {
    eRv.textContent = G.RADII[+eR.value] + " px";
    eOv.textContent = eO.value + "%";
    eNv.textContent = (+eN.value).toFixed(3);
    draw(); say();
  }
  eR.oninput = all; eD.onchange = all; eO.oninput = all; eN.oninput = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   2 · #aperture-svg — a moving edge behind an aperture
   ══════════════════════════════════════════════════════════════════════════ */
const FTaperture = (function () {
  const N = 64, C = (N - 1) / 2;                  // raster, centred on the aperture
  /* the moving pattern, sampled after a displacement u. Soft edges, so the
     gradient is a real number rather than a delta. */
  function raster(kind, thDeg, u) {
    const th = VZ.rad(thDeg), n = [Math.cos(th), Math.sin(th)];
    const m = [Math.cos(th + Math.PI / 2), Math.sin(th + Math.PI / 2)];
    const I = VZ.zeros2(N, N), s = 1.5;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const p = [x - C - u[0], y - C - u[1]];
      let v;
      const a = n[0] * p[0] + n[1] * p[1], b = m[0] * p[0] + m[1] * p[1];
      if (kind === "flat") v = 0.5;
      else if (kind === "edge") v = 0.5 + 0.33 * Math.tanh(a / s);
      else if (kind === "bar") v = 0.5 + 0.33 * (Math.tanh((a + 5) / s) - Math.tanh((a - 5) / s) - 1);
      else v = 0.5 + 0.33 * Math.min(Math.tanh(a / s), Math.tanh(b / s));   // corner
      I[y][x] = v;
    }
    return I;
  }
  /* structure tensor of what is VISIBLE, i.e. weighted by the aperture mask */
  function tensor(I, R) {
    const G = FT.grads(I, 1.2, "clamp");
    let a = 0, b = 0, c = 0, npix = 0;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      if (Math.hypot(x - C, y - C) > R) continue;
      const gx = G.gx[y][x], gy = G.gy[y][x];
      a += gx * gx; b += gx * gy; c += gy * gy; npix++;
    }
    const e = FT.eigfull(a, b, c);
    return { A: [[a, b], [b, c]], npix: npix, l0: e.l0, l1: e.l1, v0: e.v0, v1: e.v1, G: G };
  }
  /* the edge normals the pattern actually contains — used for the constraint
     lines. These come from the pattern definition, NOT from A's eigenvectors:
     for a symmetric corner A is close to a multiple of the identity and its
     eigenvectors are near-degenerate, so reading the edge orientations off them
     would be wrong. */
  function normals(kind, thDeg) {
    const th = VZ.rad(thDeg);
    if (kind === "flat") return [];
    if (kind === "corner") return [[Math.cos(th), Math.sin(th)],
      [Math.cos(th + Math.PI / 2), Math.sin(th + Math.PI / 2)]];
    return [[Math.cos(th), Math.sin(th)]];
  }
  /* rank, with an absolute floor as well as a relative one: a flat patch has
     lambda_0 and lambda_1 both at 1e-33, whose RATIO is 1 */
  function rank(T, scale) {
    const eps = 1e-6 * (scale === undefined ? 1 : scale);
    if (T.l1 < eps) return 0;
    return T.l0 < 1e-3 * T.l1 ? 1 : 2;
  }
  return { N, C, raster, tensor, normals, rank };
})();

(function () {
  const svg = d3.select("#aperture-svg");
  if (svg.empty()) return;
  const AP = FTaperture, N = AP.N, C = AP.C;
  const eC = document.getElementById("ap-c"), eT = document.getElementById("ap-th");
  const eTv = document.getElementById("ap-thv"), eR = document.getElementById("ap-r");
  const eRv = document.getElementById("ap-rv"), eF = document.getElementById("ap-full");
  const out = document.getElementById("aperture-readout");
  let U = [4.5, -2.6];                              // the true displacement, draggable

  const cw = 3.6, IW = N * cw;
  const VS = 190, VR = 11;                          // velocity panel size and half-range

  function state() {
    const kind = eC.value, th = +eT.value, R = +eR.value;
    const I = AP.raster(kind, th, U);
    const T = AP.tensor(I, R);
    const ns = AP.normals(kind, th);
    const rk = AP.rank(T);
    /* the recoverable part: the projection of U onto the span of the gradient
       directions actually present. Rank 1 -> one projection; rank 2 -> all of U. */
    let un = [0, 0];
    if (rk === 1) { const n = ns[0]; un = VZ.scale(n, VZ.dot(n, U)); }
    else if (rk === 2) un = U.slice();
    return { kind, th, R, I, T, ns, rk, un };
  }

  function draw() {
    const st = state();
    svg.selectAll("*").remove();
    const g = svg.append("g");

    /* ---- left: the aperture view ------------------------------------------ */
    const P1 = FT.box(g, 14, 40, IW, IW, "what the aperture shows");
    const rev = eF.checked;
    FT.cells(P1, 0, 0, cw, N, N, (x, y) => {
      const d = Math.hypot(x - C, y - C);
      if (d <= st.R) return FT.grey(st.I[y][x]);
      return rev ? FT.grey(0.30 + 0.55 * (st.I[y][x] - 0.17)) : "rgb(16,18,25)";
    });
    P1.append("circle").attr("cx", (C + 0.5) * cw).attr("cy", (C + 0.5) * cw)
      .attr("r", st.R * cw).attr("fill", "none").attr("stroke", VC.ink).attr("stroke-width", 1.4);
    /* the true displacement, drawn from the aperture centre */
    VZ.arrow(P1, (C + 0.5) * cw, (C + 0.5) * cw,
      (C + 0.5 + U[0]) * cw, (C + 0.5 + U[1]) * cw, { color: VC.a2, w: 2.2, head: 7 });
    VZ.arrow(P1, (C + 0.5) * cw, (C + 0.5) * cw,
      (C + 0.5 + st.un[0]) * cw, (C + 0.5 + st.un[1]) * cw, { color: VC.good, w: 2.2, head: 7 });
    /* ghosts: the same pattern at two other displacements on the constraint line.
       Inside the circle they are indistinguishable — that is the whole point — so
       they are only drawn as boundary lines, and only once the aperture is lifted. */
    if (rev && st.rk === 1) {
      const n = st.ns[0], t = [-n[1], n[0]];
      [-7, 7].forEach(k => {
        const u = VZ.add(U, VZ.scale(t, k));
        const p0 = VZ.add(u, VZ.scale(t, -60)), p1 = VZ.add(u, VZ.scale(t, 60));
        P1.append("line")
          .attr("x1", (C + 0.5 + p0[0]) * cw).attr("y1", (C + 0.5 + p0[1]) * cw)
          .attr("x2", (C + 0.5 + p1[0]) * cw).attr("y2", (C + 0.5 + p1[1]) * cw)
          .attr("stroke", VC.violet).attr("stroke-width", 1.2).attr("stroke-dasharray", "4 3");
        VZ.arrow(P1, (C + 0.5) * cw, (C + 0.5) * cw, (C + 0.5 + u[0]) * cw, (C + 0.5 + u[1]) * cw,
          { color: VC.violet, w: 1.2, head: 5, op: 0.85, dash: "3 2" });
      });
    }
    P1.append("text").attr("x", 0).attr("y", IW + 14).attr("font-size", 10).attr("fill", VC.muted)
      .text(rev ? (st.rk === 1 ? "aperture lifted: the violet ghosts are DIFFERENT motions with the same aperture view"
                               : "aperture lifted — the true motion is now visible")
                : "orange = true motion, green = the part that is recoverable");

    /* ---- middle: velocity space ------------------------------------------- */
    const VX = 14 + IW + 34;
    const P2 = FT.box(g, VX, 40, VS, VS, "velocity space (u, v)");
    const sx = d3.scaleLinear().domain([-VR, VR]).range([0, VS]);
    const sy = d3.scaleLinear().domain([-VR, VR]).range([0, VS]);
    P2.append("line").attr("x1", 0).attr("x2", VS).attr("y1", sy(0)).attr("y2", sy(0)).attr("stroke", VC.grid);
    P2.append("line").attr("y1", 0).attr("y2", VS).attr("x1", sx(0)).attr("x2", sx(0)).attr("stroke", VC.grid);
    /* one constraint line per distinct gradient orientation present */
    st.ns.forEach((n, i) => {
      const d = VZ.dot(n, U), t = [-n[1], n[0]];
      const p0 = VZ.add(VZ.scale(n, d), VZ.scale(t, -3 * VR));
      const p1 = VZ.add(VZ.scale(n, d), VZ.scale(t, 3 * VR));
      P2.append("line").attr("x1", sx(p0[0])).attr("y1", sy(p0[1]))
        .attr("x2", sx(p1[0])).attr("y2", sy(p1[1]))
        .attr("stroke", i ? VC.violet : VC.accent).attr("stroke-width", 1.6).attr("stroke-dasharray", "5 3");
    });
    if (st.rk > 0) {
      P2.append("circle").attr("cx", sx(st.un[0])).attr("cy", sy(st.un[1])).attr("r", 4).attr("fill", VC.good);
      P2.append("text").attr("x", sx(st.un[0]) + 7).attr("y", sy(st.un[1]) + 3.5).attr("font-size", 10)
        .attr("fill", VC.good).text(st.rk === 1 ? "normal flow — all that survives" : "determined");
    }
    const knob = P2.append("circle").attr("cx", sx(U[0])).attr("cy", sy(U[1])).attr("r", 6)
      .attr("fill", VC.a2).attr("stroke", VC.bg).attr("stroke-width", 1.5).style("cursor", "grab");
    P2.append("text").attr("x", sx(U[0]) + 9).attr("y", sy(U[1]) - 7).attr("font-size", 10)
      .attr("fill", VC.a2).text("true motion — drag");
    knob.call(d3.drag().on("drag", ev => {
      const [mx, my] = d3.pointer(ev, P2.node());
      U = [VZ.clamp(sx.invert(mx), -VR, VR), VZ.clamp(sy.invert(my), -VR, VR)];
      draw(); say();
    }));

    /* ---- right: the structure tensor of the visible content ---------------- */
    const RX = VX + VS + 34;
    FT.matText(g, st.T.A, RX, 62, { dp: 2, pad: 9, label: "A = ∑ ∇I∇Iᵀ over the aperture" });
    const put = FT.kv(g, RX, 118, { keyW: 130, lead: 15.5 });
    put("λ₀  (smaller)", VZ.fmt(st.T.l0, 3), st.rk === 2 ? VC.good : VC.bad, true);
    put("λ₁  (larger)", VZ.fmt(st.T.l1, 3), VC.ink);
    put("det A", st.T.A[0][0] * st.T.A[1][1] - st.T.A[0][1] * st.T.A[0][1] < 1e-9 ? "≈ 0" :
      VZ.fmt(st.T.A[0][0] * st.T.A[1][1] - st.T.A[0][1] * st.T.A[0][1], 3), VC.ink);
    put("rank of A", ["0 — nothing", "1 — one free direction", "2 — determined"][st.rk],
      st.rk === 2 ? VC.good : VC.bad, true);
    put("pixels inside", st.T.npix, VC.muted);
    /* the uncertainty ellipse, semi-axes ∝ 1/√λ, capped so a rank-1 case is legible */
    const EX = RX + 68, EY = 250, sc = 26;
    g.append("text").attr("x", RX).attr("y", 208).attr("font-size", 10.5).attr("fill", VC.muted)
      .text("uncertainty ellipse, semi-axes ∝ 1/√λ");
    const cap = 3.6;
    const a0 = Math.min(cap, st.T.l0 > 1e-9 ? 1 / Math.sqrt(st.T.l0) * Math.sqrt(st.T.l1) : cap);
    const pts = [];
    for (let i = 0; i <= 72; i++) {
      const t = i / 72 * 2 * Math.PI, ca = Math.cos(t) * a0, sa = Math.sin(t) * 1;
      pts.push([EX + sc * (ca * st.T.v0[0] + sa * st.T.v1[0]), EY + sc * (ca * st.T.v0[1] + sa * st.T.v1[1])]);
    }
    VZ.poly(g, pts, { stroke: st.rk === 2 ? VC.good : VC.bad, fill: VC.panel2, fillOp: 0.5, w: 1.6 });
    if (st.rk === 1)
      g.append("text").attr("x", RX).attr("y", EY + 66).attr("font-size", 10).attr("fill", VC.bad)
        .attr("xml:space", "preserve").text("λ₀ = 0 ⇒ the ellipse is an infinite strip");
    g.append("circle").attr("cx", EX).attr("cy", EY).attr("r", 2).attr("fill", VC.ink);
  }

  function say() {
    const st = state();
    const rank = st.rk;
    const lost = VZ.norm(VZ.sub(U, st.un));
    out.innerHTML =
      `True displacement <b>(${VZ.fmt(U[0], 2)}, ${VZ.fmt(U[1], 2)})</b>, magnitude <b>${VZ.fmt(VZ.norm(U), 2)}</b> px. `
      + `Recoverable (normal-flow) part <b>(${VZ.fmt(st.un[0], 2)}, ${VZ.fmt(st.un[1], 2)})</b>, `
      + `so <b>${VZ.fmt(lost, 2)}</b> px of motion is invisible. `
      + `The structure tensor of the visible content has <span style="color:${VC.ink}">λ₀ = ${VZ.fmt(st.T.l0, 3)}</span>, `
      + `<span style="color:${VC.ink}">λ₁ = ${VZ.fmt(st.T.l1, 3)}</span> — rank <b>${rank}</b>.<br>`
      + (rank === 2
        ? `Two gradient orientations, two constraint lines, one intersection: the displacement is determined and the aperture problem is gone. This is what a detector is looking for.`
        : rank === 1
          ? `One gradient orientation. Every point on the dashed constraint line produces an <i>identical</i> image inside the circle — drag the true motion along that line and the left panel does not change. Tick "reveal" to see that the motions were different all along.`
          : `No gradient at all. The constraint is vacuous: every displacement is consistent with the data.`);
  }
  function all() {
    eTv.textContent = eT.value + "°"; eRv.textContent = eR.value + " px";
    draw(); say();
  }
  eC.onchange = all; eT.oninput = all; eR.oninput = all; eF.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   3 · #autocorr-svg — the exact surface against its quadratic approximation
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#autocorr-svg");
  if (svg.empty()) return;
  const S = FTscene;
  const eP = document.getElementById("ac-p"), eS = document.getElementById("ac-s");
  const eSv = document.getElementById("ac-sv"), eR = document.getElementById("ac-r");
  const eRv = document.getElementById("ac-rv"), eQ = document.getElementById("ac-q");
  const out = document.getElementById("autocorr-readout");
  let P = S.SPOTS.corner.slice(), custom = false;

  /* the EXACT surface: shift the weighted patch against itself and accumulate.
     No linearisation anywhere — this is the object §04 approximates. */
  function exact(c, sw, R) {
    const K = Math.ceil(3 * sw), n = 2 * R + 1, F = VZ.zeros2(n, n);
    const wts = [];
    for (let v = -K; v <= K; v++) for (let u = -K; u <= K; u++)
      wts.push([u, v, Math.exp(-(u * u + v * v) / (2 * sw * sw))]);
    let wsum = 0; for (const w of wts) wsum += w[2];
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
      const dx = b - R, dy = a - R;
      let s = 0;
      for (const [u, v, w] of wts) {
        const i0 = FT.bilerpC(S.A, c[1] + v, c[0] + u);
        const i1 = FT.bilerpC(S.A, c[1] + v + dy, c[0] + u + dx);
        s += w * (i1 - i0) * (i1 - i0);
      }
      F[a][b] = s / wsum;
    }
    return F;
  }
  /* the quadratic approximation, from the same weighted window */
  function quad(c, sw, R) {
    const T = FT.structure(S.A, 1.0, sw, "mirror");
    const a = T.A11[c[1]][c[0]], b = T.A12[c[1]][c[0]], d = T.A22[c[1]][c[0]];
    /* the tensor above is normalised by a unit-sum Gaussian, matching exact()'s
       division by wsum, so the two surfaces share a scale without any fitting */
    const n = 2 * R + 1, F = VZ.zeros2(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const dx = j - R, dy = i - R;
      F[i][j] = a * dx * dx + 2 * b * dx * dy + d * dy * dy;
    }
    return { F: F, a: a, b: b, d: d, e: FT.eigfull(a, b, d) };
  }

  function draw() {
    const sw = +eS.value, R = +eR.value;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.9, IW = S.W * cw;
    const E = exact(P, sw, R), Q = quad(P, sw, R);
    const n = 2 * R + 1;
    /* each surface on its OWN scale. They cannot share one: a paraboloid grows
       without bound while the true mismatch saturates once the shift exceeds the
       size of the structure, so a shared scale renders the exact surface black. */
    const hiE = Math.max(FT.extent2(E)[1], 1e-9), hiQ = Math.max(FT.extent2(Q.F)[1], 1e-9);

    /* ---- left: the image, with the window and a click target --------------- */
    const P1 = FT.box(g, 14, 40, IW, IW, "the image · click to move the window");
    FT.cells(P1, 0, 0, cw, S.W, S.H, (x, y) => FT.grey(S.A[y][x]));
    P1.append("rect").attr("x", 0).attr("y", 0).attr("width", IW).attr("height", IW)
      .attr("fill", "transparent").style("cursor", "crosshair")
      .on("click", ev => {
        const [mx, my] = d3.pointer(ev, P1.node());
        P = [VZ.clamp(Math.round(mx / cw), 3, S.W - 4), VZ.clamp(Math.round(my / cw), 3, S.H - 4)];
        custom = true; draw(); say();
      });
    P1.append("circle").attr("cx", (P[0] + 0.5) * cw).attr("cy", (P[1] + 0.5) * cw)
      .attr("r", 2 * sw * cw).attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.5);
    P1.append("circle").attr("cx", (P[0] + 0.5) * cw).attr("cy", (P[1] + 0.5) * cw).attr("r", 2).attr("fill", VC.a2);

    /* ---- the two surfaces, on one colour scale ---------------------------- */
    const sz = 132, sc = sz / n;
    function surf(x0, title, F, col, hi) {
      const B = FT.box(g, x0, 40, sz, sz, title);
      FT.cells(B, 0, 0, sc, n, n, (x, y) => FT.heat(VZ.clamp(F[y][x] / hi, 0, 1)));
      /* contour rings, so the anisotropy is visible as shape and not only colour */
      [0.08, 0.25, 0.55].forEach(lv => {
        const segs = [];
        for (let i = 0; i < n - 1; i++) for (let j = 0; j < n - 1; j++) {
          const v = F[i][j] / hi;
          if ((v < lv) !== (F[i][j + 1] / hi < lv)) segs.push([(j + 1) * sc, i * sc, (j + 1) * sc, (i + 1) * sc]);
          if ((v < lv) !== (F[i + 1][j] / hi < lv)) segs.push([j * sc, (i + 1) * sc, (j + 1) * sc, (i + 1) * sc]);
        }
        B.append("path").attr("d", segs.map(s => `M${s[0]},${s[1]}L${s[2]},${s[3]}`).join(" "))
          .attr("stroke", VC.ink).attr("stroke-opacity", 0.45).attr("stroke-width", 0.7).attr("fill", "none");
      });
      B.append("circle").attr("cx", (R + 0.5) * sc).attr("cy", (R + 0.5) * sc).attr("r", 2.2).attr("fill", col);
      B.append("text").attr("x", 0).attr("y", sz + 13).attr("font-size", 9.5).attr("fill", VC.muted)
        .text("max " + hi.toFixed(3) + " · Δu ∈ [−" + R + ", " + R + "]");
      return B;
    }
    surf(14 + IW + 26, "exact  E_AC(Δu)", E, VC.good, hiE);
    if (eQ.checked) surf(14 + IW + 26 + sz + 22, "quadratic  ΔuᵀAΔu", Q.F, VC.accent, hiQ);

    /* ---- right: the tensor, the ellipse, and the two surfaces' disagreement */
    const RX = 14 + IW + 26 + 2 * sz + 48;
    FT.matText(g, [[Q.a, Q.b], [Q.b, Q.d]], RX, 60, { dp: 5, pad: 9, label: "A at this point" });
    const put = FT.kv(g, RX, 112, { keyW: 128, lead: 15.5 });
    put("λ₀", VZ.fmt(Q.e.l0, 5), VC.accent, true);
    put("λ₁", VZ.fmt(Q.e.l1, 5), VC.a2, true);
    put("λ₁ / λ₀", Q.e.l0 > 1e-12 ? VZ.fmt(Q.e.l1 / Q.e.l0, 1) : "∞", VC.ink);
    put("det A", Q.e.l0 * Q.e.l1 < 1e-10 ? Q.e.l0 * Q.e.l1 < 1e-14 ? "≈ 0" : (Q.e.l0 * Q.e.l1).toExponential(2)
      : VZ.fmt(Q.e.l0 * Q.e.l1, 7), VC.ink);
    /* disagreement, split by shift magnitude: the approximation is claimed only
       for small shifts, so quoting one number over the whole range hides the point */
    const dis = disagree(E, Q.F, R);
    put("error, |Δu| ≤ 3 px", (100 * dis.near).toFixed(1) + "%", dis.near > 0.2 ? VC.a2 : VC.good, true);
    put("error, whole range", (100 * dis.all).toFixed(1) + "%", dis.all > 0.2 ? VC.bad : VC.good, true);
    /* the level ellipse of the quadratic form, drawn at a fixed contour value */
    const EX = RX + 78, EY = 258, S2 = 46;
    g.append("text").attr("x", RX).attr("y", 214).attr("font-size", 10.5).attr("fill", VC.muted)
      .text("level ellipse ΔuᵀAΔu = c, axes 1/√λ");
    const lv = 0.25 * hiQ;   /* level set of the QUADRATIC form, so its own scale */
    const A0 = Q.e.l0 > 1e-12 ? Math.sqrt(lv / Q.e.l0) : 1e6, A1 = Q.e.l1 > 1e-12 ? Math.sqrt(lv / Q.e.l1) : 1e6;
    const k = S2 / Math.max(A0, R), pts = [];
    for (let i = 0; i <= 96; i++) {
      const t = i / 96 * 2 * Math.PI;
      const p = VZ.add(VZ.scale(Q.e.v0, Math.min(A0, 3 * R) * Math.cos(t)), VZ.scale(Q.e.v1, A1 * Math.sin(t)));
      pts.push([EX + k * p[0], EY + k * p[1]]);
    }
    VZ.poly(g, pts, { stroke: VC.a2, fill: VC.panel2, fillOp: 0.55, w: 1.6 });
    g.append("circle").attr("cx", EX).attr("cy", EY).attr("r", 2).attr("fill", VC.ink);
  }

  /* mean |exact − quadratic|, as a fraction of the exact surface's own maximum,
     inside a shift radius and over the whole range */
  function disagree(E, Q, R) {
    const n = 2 * R + 1;
    let mx = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) mx = Math.max(mx, E[i][j]);
    mx = Math.max(mx, 1e-12);
    const acc = rad => {
      let e = 0, c = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        if (Math.hypot(j - R, i - R) > rad) continue;
        e += Math.abs(E[i][j] - Q[i][j]); c++;
      }
      return e / Math.max(1, c) / mx;
    };
    return { near: acc(3), all: acc(R) };
  }

  function say() {
    const sw = +eS.value, R = +eR.value;
    const E = exact(P, sw, R), Q = quad(P, sw, R);
    const dis = disagree(E, Q.F, R);
    const r = Q.e.l0 > 1e-12 ? Q.e.l1 / Q.e.l0 : Infinity;
    const kind = Q.e.l1 < 1e-6 ? "flat" : (r > 50 ? "an edge" : "a corner");
    out.innerHTML =
      `At (${P[0]}, ${P[1]}) with an integration window of <span class="keep">σ</span> = ${sw}: `
      + `<b>λ₀ = ${VZ.fmt(Q.e.l0, 5)}</b>, <b>λ₁ = ${VZ.fmt(Q.e.l1, 5)}</b>, ratio `
      + `<b>${isFinite(r) ? VZ.fmt(r, 1) : "∞"}</b> — ${kind}. `
      + `The quadratic form differs from the exact surface by <b>${(100 * dis.near).toFixed(1)}%</b> of the exact `
      + `maximum for shifts within 3 px, and by <b>${(100 * dis.all).toFixed(1)}%</b> over the whole plotted range.<br>`
      + `The approximation is a <i>first-order Taylor expansion in the shift</i>, so it is accurate near Δu = 0 and `
      + `degrades outward. Note the two maxima printed under the panels: the paraboloid keeps growing while the `
      + `true mismatch saturates once the shift exceeds the size of the local structure, so the two cannot share `
      + `a colour scale and each is drawn on its own. On the repeating grid it fails in a `
      + `qualitatively different way — the exact surface has several deep minima, one per period, and the `
      + `quadratic has exactly one, because a paraboloid cannot have two minima. That is not a numerical error, `
      + `it is the linearisation being blind to structure it never looked at, and it is why a locally excellent `
      + `corner on a repeated pattern is a matching disaster (§25).`;
  }
  function all() {
    if (!custom) P = FTscene.SPOTS[eP.value].slice();
    eSv.textContent = (+eS.value).toFixed(2); eRv.textContent = eR.value;
    draw(); say();
  }
  eP.onchange = all; eS.oninput = all; eR.oninput = all; eQ.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   4 · #tensor-svg — the tensor, its ellipse, and rotation invariance
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#tensor-svg");
  if (svg.empty()) return;
  const S = FTscene;
  const eP = document.getElementById("ts-p"), eR = document.getElementById("ts-rot");
  const eRv = document.getElementById("ts-rotv"), eD = document.getElementById("ts-sd");
  const eDv = document.getElementById("ts-sdv"), eI = document.getElementById("ts-si");
  const eIv = document.getElementById("ts-siv");
  const out = document.getElementById("tensor-readout");

  /* rotate the image about its centre by inverse warping, and carry the sample
     point with it — the same rotation applied to the scene and to the query */
  const CACHE = new Map();
  function rotated(deg) {
    if (CACHE.has(deg)) return CACHE.get(deg);
    const c = (S.W - 1) / 2;
    const Hm = VZ.mul(VZ.mul(VZ.T2(c, c), VZ.R2(VZ.rad(deg), 0, 0)), VZ.T2(-c, -c));
    const Hi = VZ.inv3(Hm), J = VZ.zeros2(S.H, S.W);
    for (let y = 0; y < S.H; y++) for (let x = 0; x < S.W; x++) {
      const p = VZ.applyH(Hi, [x, y]);
      J[y][x] = FT.bilerpC(S.A, p[1], p[0]);
    }
    const r = { J: J, Hm: Hm };
    CACHE.set(deg, r);
    return r;
  }
  function at(deg, spot, sd, si) {
    const R = rotated(deg);
    const p0 = S.SPOTS[spot], p = VZ.applyH(R.Hm, p0);
    const px = Math.round(VZ.clamp(p[0], 0, S.W - 1)), py = Math.round(VZ.clamp(p[1], 0, S.H - 1));
    const T = FT.structure(R.J, sd, si, "mirror");
    const a = T.A11[py][px], b = T.A12[py][px], c = T.A22[py][px];
    return { J: R.J, px, py, a, b, c, e: FT.eigfull(a, b, c), G: T, det: a * c - b * b, tr: a + c };
  }

  function draw() {
    const deg = +eR.value, sd = +eD.value, si = +eI.value, spot = eP.value;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.7, IW = S.W * cw;
    const cur = at(deg, spot, sd, si), ref = at(0, spot, sd, si);

    /* ---- left: the rotated image with the integration window --------------- */
    const P1 = FT.box(g, 14, 40, IW, IW, "image, rotated by " + deg + "°");
    FT.cells(P1, 0, 0, cw, S.W, S.H, (x, y) => FT.grey(cur.J[y][x]));
    P1.append("circle").attr("cx", (cur.px + 0.5) * cw).attr("cy", (cur.py + 0.5) * cw)
      .attr("r", 2 * si * cw).attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.5);
    P1.append("circle").attr("cx", (cur.px + 0.5) * cw).attr("cy", (cur.py + 0.5) * cw)
      .attr("r", 2).attr("fill", VC.a2);
    P1.append("text").attr("x", 0).attr("y", IW + 14).attr("font-size", 10).attr("fill", VC.muted)
      .text("circle = 2σᵢ, the effective pooling radius");

    /* ---- middle: the gradients inside the window, from a common origin ----- */
    const GX = 14 + IW + 30, GS = 176;
    const P2 = FT.box(g, GX, 40, GS, GS, "gradients in the window");
    const half = GS / 2;
    P2.append("line").attr("x1", 0).attr("x2", GS).attr("y1", half).attr("y2", half).attr("stroke", VC.grid);
    P2.append("line").attr("y1", 0).attr("y2", GS).attr("x1", half).attr("x2", half).attr("stroke", VC.grid);
    const rad = Math.ceil(2 * si), gs = [];
    let gmax = 1e-9;
    for (let dy = -rad; dy <= rad; dy++) for (let dx = -rad; dx <= rad; dx++) {
      const y = cur.py + dy, x = cur.px + dx;
      if (x < 0 || y < 0 || x >= S.W || y >= S.H) continue;
      if (dx * dx + dy * dy > rad * rad) continue;
      const gx = cur.G.gx[y][x], gy = cur.G.gy[y][x];
      gs.push([gx, gy, Math.exp(-(dx * dx + dy * dy) / (2 * si * si))]);
      gmax = Math.max(gmax, Math.hypot(gx, gy));
    }
    const k = (half - 12) / gmax;
    gs.forEach(([gx, gy, w]) => {
      P2.append("line").attr("x1", half).attr("y1", half)
        .attr("x2", half + k * gx).attr("y2", half + k * gy)
        .attr("stroke", VC.muted).attr("stroke-width", 1).attr("stroke-opacity", 0.25 + 0.6 * w);
    });
    /* the two eigenvectors, scaled by √λ */
    const lm = Math.sqrt(Math.max(cur.e.l1, 1e-12));
    [[cur.e.v1, cur.e.l1, VC.a2], [cur.e.v0, cur.e.l0, VC.accent]].forEach(([v, l, col]) => {
      const s2 = (half - 16) * Math.sqrt(Math.max(l, 0)) / lm;
      VZ.arrow(P2, half, half, half + s2 * v[0], half + s2 * v[1], { color: col, w: 2, head: 6 });
      VZ.arrow(P2, half, half, half - s2 * v[0], half - s2 * v[1], { color: col, w: 2, head: 6 });
    });
    P2.append("text").attr("x", 0).attr("y", GS + 13).attr("font-size", 9.5).attr("fill", VC.muted)
      .text(gs.length + " gradient vectors, from one origin");

    /* ---- right: the matrix now and at 0°, and the invariants --------------- */
    const RX = GX + GS + 34;
    FT.matText(g, [[cur.a, cur.b], [cur.b, cur.c]], RX, 60,
      { dp: 5, pad: 9, label: "A at " + deg + "° — the ENTRIES change" });
    FT.matText(g, [[ref.a, ref.b], [ref.b, ref.c]], RX, 118,
      { dp: 5, pad: 9, label: "A at 0°, for comparison", fill: VC.muted });
    const put = FT.kv(g, RX, 176, { keyW: 150, lead: 15.5 });
    put("λ₀  now / at 0°", VZ.fmt(cur.e.l0, 5) + "  /  " + VZ.fmt(ref.e.l0, 5), VC.accent, true);
    put("λ₁  now / at 0°", VZ.fmt(cur.e.l1, 5) + "  /  " + VZ.fmt(ref.e.l1, 5), VC.a2, true);
    put("det A  now / at 0°", cur.det.toExponential(3) + " / " + ref.det.toExponential(3), VC.ink);
    put("tr A   now / at 0°", VZ.fmt(cur.tr, 5) + "  /  " + VZ.fmt(ref.tr, 5), VC.ink);
    const drift = Math.max(Math.abs(cur.e.l0 - ref.e.l0), Math.abs(cur.e.l1 - ref.e.l1))
      / Math.max(1e-12, ref.e.l1);
    put("largest λ drift", (100 * drift).toFixed(2) + "%", drift < 0.10 ? VC.good : VC.a2, true);
    put("rank", cur.e.l1 < 1e-7 ? "0" : (cur.e.l0 < 1e-3 * cur.e.l1 ? "1 — aperture problem" : "2"),
      cur.e.l0 < 1e-3 * cur.e.l1 ? VC.bad : VC.good, true);

    /* the level ellipse of the quadratic form, axes 1/√λ */
    const EX = RX + 92, EY = 340, SC = 52;
    g.append("text").attr("x", RX).attr("y", 296).attr("font-size", 10.5).attr("fill", VC.muted)
      .text("level ellipse of ΔuᵀAΔu, axes ∝ 1/√λ");
    const a0 = cur.e.l0 > 1e-10 ? 1 / Math.sqrt(cur.e.l0) : 1e6;
    const a1 = cur.e.l1 > 1e-10 ? 1 / Math.sqrt(cur.e.l1) : 1e6;
    const nrm = Math.min(a0, 40 * a1), pts = [];
    for (let i = 0; i <= 96; i++) {
      const t = i / 96 * 2 * Math.PI;
      const p = VZ.add(VZ.scale(cur.e.v0, Math.min(a0, nrm) * Math.cos(t)),
        VZ.scale(cur.e.v1, a1 * Math.sin(t)));
      pts.push([EX + SC * p[0] / Math.max(nrm, 1e-9), EY + SC * p[1] / Math.max(nrm, 1e-9)]);
    }
    VZ.poly(g, pts, { stroke: VC.a2, fill: VC.panel2, fillOp: 0.55, w: 1.6 });
    g.append("circle").attr("cx", EX).attr("cy", EY).attr("r", 2).attr("fill", VC.ink);
  }

  function say() {
    const deg = +eR.value, sd = +eD.value, si = +eI.value, spot = eP.value;
    const cur = at(deg, spot, sd, si), ref = at(0, spot, sd, si);
    const drift = 100 * Math.max(Math.abs(cur.e.l0 - ref.e.l0), Math.abs(cur.e.l1 - ref.e.l1))
      / Math.max(1e-12, ref.e.l1);
    const tiny = si < 0.3;
    out.innerHTML =
      `At ${deg}°: <code>A = [[${VZ.fmt(cur.a, 5)}, ${VZ.fmt(cur.b, 5)}], [${VZ.fmt(cur.b, 5)}, ${VZ.fmt(cur.c, 5)}]]</code>, `
      + `against <code>[[${VZ.fmt(ref.a, 5)}, ${VZ.fmt(ref.b, 5)}], [${VZ.fmt(ref.b, 5)}, ${VZ.fmt(ref.c, 5)}]]</code> at 0°. `
      + `The entries move; the eigenvalues drift by at most <b>${drift.toFixed(2)}%</b> of λ₁, which is resampling `
      + `and boundary error rather than a failure of the invariance — the ideal operator's drift is exactly zero, `
      + `because <code>A → RARᵀ</code> is a similarity transform. The evidence for that reading is that the drift `
      + `returns to exactly <b>0%</b> at 0° and 90°, the only two rotations here under which the pixel lattice maps `
      + `to itself and no resampling happens at all; in between it peaks near 60° at about 10% of λ₁. `
      + `These eigenvalues come from the shared Jacobi eigensolver, which agrees with the closed-form 2 × 2 `
      + `expression to <b>${FT.eigAgree(2000).toExponential(2)}</b> over 2000 random symmetric matrices — `
      + `recomputed live, here, every time this figure draws.<br>`
      + (tiny
        ? `<b>The integration scale is at its floor.</b> With no pooling, <code>A = ∇I∇Iᵀ</code> is a single outer `
          + `product: rank 1 by construction, <code>det A = ${cur.det.toExponential(2)}</code>, and <i>no</i> pixel in the `
          + `image is a corner. Raise σᵢ and watch the determinant become non-zero. A corner is a property of a `
          + `neighbourhood, not of a pixel.`
        : `Determinant <b>${cur.det.toExponential(3)}</b>, trace <b>${VZ.fmt(cur.tr, 5)}</b>. `
          + `These two numbers are a complete set of rotation invariants for a symmetric 2 × 2 — every response `
          + `function in §06 is built from them and from nothing else.`);
  }
  function all() {
    eRv.textContent = eR.value + "°"; eDv.textContent = (+eD.value).toFixed(1);
    eIv.textContent = (+eI.value).toFixed(2);
    draw(); say();
  }
  eP.onchange = all; eR.oninput = all; eD.oninput = all; eI.oninput = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   5 · #response-svg — the response functions in the eigenvalue plane
   ══════════════════════════════════════════════════════════════════════════ */
const FTresp = (function () {
  const S = FTscene;
  const FNS = {
    harris: { name: "Harris", f: (l0, l1, k) => l0 * l1 - k * (l0 + l1) * (l0 + l1), col: () => VC.a2 },
    shi: { name: "Shi–Tomasi", f: l0 => l0, col: () => VC.good },
    noble: { name: "Noble", f: (l0, l1) => (l0 + l1) > 1e-15 ? l0 * l1 / (l0 + l1) : 0, col: () => VC.violet },
    triggs: { name: "Triggs", f: (l0, l1, k) => l0 - k * l1, col: () => VC.teal }
  };
  /* the response of every function over the whole image, at one (σ_d, σ_i, k) */
  const CACHE = new Map();
  function maps(sd, si, k) {
    const key = sd + "|" + si + "|" + k;
    if (CACHE.has(key)) return CACHE.get(key);
    const T = FT.structure(S.A, sd, si, "mirror");
    const M = {};
    for (const n of Object.keys(FNS)) M[n] = VZ.zeros2(S.H, S.W);
    const L0 = VZ.zeros2(S.H, S.W), L1 = VZ.zeros2(S.H, S.W);
    for (let y = 0; y < S.H; y++) for (let x = 0; x < S.W; x++) {
      const e = FT.eig2(T.A11[y][x], T.A12[y][x], T.A22[y][x]);
      L0[y][x] = e[0]; L1[y][x] = e[1];
      for (const n of Object.keys(FNS)) M[n][y][x] = FNS[n].f(e[0], e[1], k);
    }
    const r = { M: M, L0: L0, L1: L1 };
    CACHE.set(key, r);
    return r;
  }
  /* local maxima of a response map, above a fraction of its own maximum */
  function detect(R, frac, minsep) {
    const H = R.length, W = R[0].length;
    let hi = -Infinity;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) hi = Math.max(hi, R[y][x]);
    const thr = frac * hi, cand = [];
    for (let y = 4; y < H - 4; y++) for (let x = 4; x < W - 4; x++) {
      const v = R[y][x];
      if (v < thr) continue;
      let ok = true;
      for (let b = -1; b <= 1 && ok; b++) for (let a = -1; a <= 1; a++)
        if ((a || b) && R[y + b][x + a] > v) { ok = false; break; }
      if (ok) cand.push([x, y, v]);
    }
    cand.sort((p, q) => q[2] - p[2]);
    const keep = [];
    for (const c of cand) if (keep.every(q => Math.hypot(q[0] - c[0], q[1] - c[1]) >= (minsep || 3))) keep.push(c);
    return { pts: keep, max: hi, thr: thr };
  }
  /* the admissible anisotropy band of the Harris sign test, in closed form */
  function band(k) {
    const disc = 1 - 4 * k;
    if (disc < 0) return null;
    return [((1 - 2 * k) - Math.sqrt(disc)) / (2 * k), ((1 - 2 * k) + Math.sqrt(disc)) / (2 * k)];
  }
  return { FNS, maps, detect, band };
})();

(function () {
  const svg = d3.select("#response-svg");
  if (svg.empty()) return;
  const S = FTscene, RP = FTresp;
  const eF = document.getElementById("rp-f"), eK = document.getElementById("rp-k");
  const eKv = document.getElementById("rp-kv"), eT = document.getElementById("rp-t");
  const eTv = document.getElementById("rp-tv"), eC = document.getElementById("rp-cmp");
  const out = document.getElementById("response-readout");
  let Q = [0.45, 0.80];                          // the draggable point, in (λ₀, λ₁) units of λmax

  const PW = 268, LMAX = 1.0;

  function draw() {
    const fn = eF.value, k = +eK.value, frac = (+eT.value) / 100;
    svg.selectAll("*").remove();
    const g = svg.append("g");

    /* ---- left: the eigenvalue plane --------------------------------------- */
    const P1 = FT.box(g, 44, 40, PW, PW, "the (λ₀, λ₁) plane");
    const x = d3.scaleLinear().domain([0, LMAX]).range([0, PW]);
    const y = d3.scaleLinear().domain([0, LMAX]).range([PW, 0]);
    const NG = 84, cw = PW / NG;
    const F = RP.FNS[fn].f;
    let amax = 1e-12;
    const grid = [];
    for (let j = 0; j < NG; j++) {
      grid.push([]);
      for (let i = 0; i < NG; i++) {
        const l0 = LMAX * (i + 0.5) / NG, l1 = LMAX * (NG - j - 0.5) / NG;
        const v = (l1 >= l0) ? F(l0, l1, k) : NaN;
        grid[j].push(v);
        if (isFinite(v)) amax = Math.max(amax, Math.abs(v));
      }
    }
    FT.cells(P1, 0, 0, cw, NG, NG, (i, j) => {
      const v = grid[j][i];
      return isFinite(v) ? FT.signed(v / amax) : "rgb(14,16,22)";
    });
    /* the zero contour and the threshold contour, traced on the same grid */
    function contour(level, col, dash) {
      const segs = [];
      for (let j = 0; j < NG - 1; j++) for (let i = 0; i < NG - 1; i++) {
        const a = grid[j][i], b = grid[j][i + 1], c = grid[j + 1][i];
        if (!isFinite(a)) continue;
        if (isFinite(b) && ((a < level) !== (b < level))) segs.push([(i + 1) * cw, j * cw, (i + 1) * cw, (j + 1) * cw]);
        if (isFinite(c) && ((a < level) !== (c < level))) segs.push([i * cw, (j + 1) * cw, (i + 1) * cw, (j + 1) * cw]);
      }
      if (!segs.length) return false;
      P1.append("path").attr("d", segs.map(s => `M${s[0]},${s[1]}L${s[2]},${s[3]}`).join(" "))
        .attr("stroke", col).attr("stroke-width", 1.5).attr("fill", "none")
        .attr("stroke-dasharray", dash || null);
      return true;
    }
    const hasZero = contour(0, VC.ink);
    const thrLevel = frac * amax;
    contour(thrLevel, VC.good, "4 3");
    if (eC.checked && fn !== "shi") {
      /* the Shi–Tomasi boundary at the same acceptance FRACTION, for comparison */
      const lvl = frac * LMAX;
      P1.append("line").attr("x1", x(lvl)).attr("x2", x(lvl)).attr("y1", y(lvl)).attr("y2", 0)
        .attr("stroke", VC.violet).attr("stroke-width", 1.4).attr("stroke-dasharray", "2 3");
      P1.append("text").attr("x", x(lvl) + 4).attr("y", 12).attr("font-size", 9.5)
        .attr("fill", VC.violet).text("Shi–Tomasi");
    }
    P1.append("line").attr("x1", 0).attr("y1", PW).attr("x2", PW).attr("y2", 0)
      .attr("stroke", VC.muted).attr("stroke-width", 0.8).attr("stroke-dasharray", "2 2");
    VZ.axisB(P1, x, PW, 5, "λ₀");
    P1.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    P1.append("text").attr("x", -34).attr("y", -8).attr("font-size", 11).attr("fill", VC.muted).text("λ₁");
    [["flat", 0.05, 0.05], ["edge", 0.05, 0.86], ["corner", 0.78, 0.90]].forEach(([t, a, b]) => {
      P1.append("text").attr("x", x(a)).attr("y", y(b)).attr("font-size", 10).attr("fill", VC.muted).text(t);
    });
    const knob = P1.append("circle").attr("cx", x(Q[0])).attr("cy", y(Q[1])).attr("r", 6)
      .attr("fill", VC.ink).attr("stroke", VC.bg).attr("stroke-width", 1.5).style("cursor", "grab");
    knob.call(d3.drag().on("drag", ev => {
      const [mx, my] = d3.pointer(ev, P1.node());
      let l0 = VZ.clamp(x.invert(mx), 0, LMAX), l1 = VZ.clamp(y.invert(my), 0, LMAX);
      if (l1 < l0) { const t = l0; l0 = l1; l1 = t; }
      Q = [l0, l1]; draw(); say();
    }));

    /* ---- middle: all four scores at the dragged point ---------------------- */
    const MX = 44 + PW + 34;
    const r = FT.responses(Q[0], 0, Q[1], k);      // a diagonal A with these eigenvalues
    g.append("text").attr("x", MX).attr("y", 40).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("at the dragged point");
    const put = FT.kv(g, MX, 62, { keyW: 128, lead: 15.5 });
    put("λ₀ , λ₁", VZ.fmt(Q[0], 3) + " , " + VZ.fmt(Q[1], 3), VC.ink, true);
    put("anisotropy λ₁/λ₀", Q[0] > 1e-9 ? VZ.fmt(Q[1] / Q[0], 2) : "∞", VC.ink);
    put("det , tr", VZ.fmt(r.det, 4) + " , " + VZ.fmt(r.tr, 3), VC.muted);
    put("tr² / det", r.det > 1e-12 ? VZ.fmt(r.tr * r.tr / r.det, 2) : "∞", VC.muted);
    put("Harris  det − k·tr²", VZ.fmt(r.harris, 4), r.harris > 0 ? VC.good : VC.bad, fn === "harris");
    put("Shi–Tomasi  λ₀", VZ.fmt(r.shi, 4), VC.ink, fn === "shi");
    put("Noble  det/tr", VZ.fmt(r.noble, 4), VC.ink, fn === "noble");
    put("Triggs  λ₀ − αλ₁", VZ.fmt(r.triggs, 4), r.triggs > 0 ? VC.good : VC.bad, fn === "triggs");
    const bd = RP.band(k);
    put("admissible λ₁/λ₀", bd ? VZ.fmt(bd[0], 4) + " … " + VZ.fmt(bd[1], 3) : "none — k ≥ ¼",
      bd ? VC.a2 : VC.bad, true);
    put("1/k  (= tr²/det limit)", VZ.fmt(1 / k, 2), VC.a2);

    /* ---- right: the four response maps on the image ----------------------- */
    const IX = MX + 216, cw2 = 1.55, IW = S.W * cw2;
    const M = RP.maps(1.0, 2.0, k);
    const names = Object.keys(RP.FNS);
    names.forEach((n, i) => {
      const px = IX + (i % 2) * (IW + 16), py = 44 + Math.floor(i / 2) * (IW + 30);
      const B = FT.box(g, px, py, IW, IW, RP.FNS[n].name);
      const R = M.M[n], D = RP.detect(R, frac, 4);
      let lo = Infinity, hi = -Infinity;
      for (let yy = 0; yy < S.H; yy++) for (let xx = 0; xx < S.W; xx++) { lo = Math.min(lo, R[yy][xx]); hi = Math.max(hi, R[yy][xx]); }
      const sc = Math.max(Math.abs(lo), Math.abs(hi), 1e-12);
      FT.cells(B, 0, 0, cw2, S.W, S.H, (xx, yy) => FT.signed(R[yy][xx] / sc));
      D.pts.forEach(p => B.append("circle").attr("cx", (p[0] + 0.5) * cw2).attr("cy", (p[1] + 0.5) * cw2)
        .attr("r", 2.2).attr("fill", "none").attr("stroke", n === fn ? VC.good : VC.a2).attr("stroke-width", 1.1));
      B.append("text").attr("x", 0).attr("y", IW + 11).attr("font-size", 9).attr("fill", VC.muted)
        .text(D.pts.length + " detections");
    });
  }

  function agree(frac, k) {
    const M = FTresp.maps(1.0, 2.0, k), names = Object.keys(FTresp.FNS);
    const sets = names.map(n => FTresp.detect(M.M[n], frac, 4).pts);
    let common = 0;
    for (const p of sets[0])
      if (sets.slice(1).every(s => s.some(q => Math.hypot(q[0] - p[0], q[1] - p[1]) <= 2))) common++;
    return { common: common, counts: sets.map(s => s.length), names: names };
  }

  function say() {
    const k = +eK.value, frac = (+eT.value) / 100, fn = eF.value;
    const bd = FTresp.band(k), a = agree(frac, k);
    const r = FT.responses(Q[0], 0, Q[1], k);
    out.innerHTML =
      (bd
        ? `At <code>k = ${k.toFixed(3)}</code> the Harris response is positive exactly when the anisotropy ratio `
          + `<code>λ₁/λ₀</code> lies in <b>(${VZ.fmt(bd[0], 4)}, ${VZ.fmt(bd[1], 3)})</b> — equivalently when `
          + `<code>tr²/det &lt; 1/k = ${VZ.fmt(1 / k, 2)}</code>. `
        : `At <code>k = ${k.toFixed(3)} ≥ ¼</code> the discriminant <code>1 − 4k</code> is negative and the response `
          + `is <b>negative everywhere</b>: the zero contour has vanished from the plane and nothing can be detected. `)
      + `At the dragged point the four scores are Harris <b>${VZ.fmt(r.harris, 4)}</b>, Shi–Tomasi `
      + `<b>${VZ.fmt(r.shi, 4)}</b>, Noble <b>${VZ.fmt(r.noble, 4)}</b>, Triggs <b>${VZ.fmt(r.triggs, 4)}</b>.<br>`
      + `On the image at a threshold of ${(100 * frac).toFixed(0)}% of each function's own maximum, the four `
      + `return ${a.counts.join(", ")} detections respectively, and <b>${a.common}</b> of them are found by all `
      + `four within 2 px. The functions differ in the <i>shape</i> of the accepted region, not in what they are `
      + `looking at — which is why their detection sets overlap heavily and their rankings do not.`;
  }
  function all() {
    eKv.textContent = (+eK.value).toFixed(3); eTv.textContent = eT.value + "%";
    draw(); say();
  }
  eF.onchange = all; eK.oninput = all; eT.oninput = all; eC.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   6 · #harris-svg — the whole detector, stage by stage
   ══════════════════════════════════════════════════════════════════════════ */
const FTharris = (function () {
  const S = FTscene;
  /* one full pass, keeping every intermediate the figure needs to draw */
  function pipeline(I, sd, si, k, tau) {
    const T = FT.structure(I, sd, si, "mirror");
    const H = I.length, W = I[0].length;
    const P11 = VZ.zeros2(H, W), P12 = VZ.zeros2(H, W), P22 = VZ.zeros2(H, W), R = VZ.zeros2(H, W);
    let hi = -Infinity;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const gx = T.gx[y][x], gy = T.gy[y][x];
      P11[y][x] = gx * gx; P12[y][x] = gx * gy; P22[y][x] = gy * gy;
      const a = T.A11[y][x], b = T.A12[y][x], c = T.A22[y][x];
      R[y][x] = (a * c - b * b) - k * (a + c) * (a + c);
      if (R[y][x] > hi) hi = R[y][x];
    }
    const thr = tau * hi, mask = VZ.zeros2(H, W), pts = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) mask[y][x] = R[y][x] >= thr ? 1 : 0;
    for (let y = 3; y < H - 3; y++) for (let x = 3; x < W - 3; x++) {
      const v = R[y][x];
      if (v < thr) continue;
      let ok = true;
      for (let b = -1; b <= 1 && ok; b++) for (let a = -1; a <= 1; a++)
        if ((a || b) && R[y + b][x + a] > v) { ok = false; break; }
      if (ok) pts.push({ x: x, y: y, v: v, r: refine(R, x, y) });
    }
    pts.sort((p, q) => q.v - p.v);
    return { T, P11, P12, P22, R, hi, thr, mask, pts };
  }
  /* the quadratic fit of §07 step 6 — returns {x, y, ok} */
  function refine(R, x, y) {
    const gx = (R[y][x + 1] - R[y][x - 1]) / 2, gy = (R[y + 1][x] - R[y - 1][x]) / 2;
    const hxx = R[y][x + 1] - 2 * R[y][x] + R[y][x - 1];
    const hyy = R[y + 1][x] - 2 * R[y][x] + R[y - 1][x];
    const hxy = (R[y + 1][x + 1] - R[y + 1][x - 1] - R[y - 1][x + 1] + R[y - 1][x - 1]) / 4;
    const det = hxx * hyy - hxy * hxy;
    if (Math.abs(det) < 1e-24) return { x: x, y: y, ok: false, ox: 0, oy: 0, hxx, hyy, hxy, gx, gy };
    const ox = -(hyy * gx - hxy * gy) / det, oy = -(-hxy * gx + hxx * gy) / det;
    const ok = Math.abs(ox) <= 1 && Math.abs(oy) <= 1;
    return { x: ok ? x + ox : x, y: ok ? y + oy : y, ok: ok, ox, oy, hxx, hyy, hxy, gx, gy };
  }
  /* shift the scene by an exact sub-pixel amount, by inverse sampling */
  function shifted(dx, dy) {
    const J = VZ.zeros2(S.H, S.W);
    for (let y = 0; y < S.H; y++) for (let x = 0; x < S.W; x++) J[y][x] = FT.bilerpC(S.A, y - dy, x - dx);
    return J;
  }
  const SHIFTS = [[0.25, 0], [0.5, 0], [0.37, 0.21], [0, 0.5], [0.5, 0.5], [0.13, -0.44], [0.75, 0.31]];
  /* localisation accuracy, integer against refined, averaged over the shifts */
  const ACC = new Map();
  function accuracy(sd, si, k, tau) {
    const key = [sd, si, k, tau].join("|");
    if (ACC.has(key)) return ACC.get(key);
    const base = pipeline(S.A, sd, si, k, tau);
    let ei = 0, er = 0, n = 0;
    for (const [dx, dy] of SHIFTS) {
      const p = pipeline(shifted(dx, dy), sd, si, k, tau);
      for (const b of base.pts) {
        let best = null, bd = 3.5;
        for (const q of p.pts) {
          const d = Math.hypot(q.x - (b.x + dx), q.y - (b.y + dy));
          if (d < bd) { bd = d; best = q; }
        }
        if (!best) continue;
        ei += Math.hypot(best.x - (b.x + dx), best.y - (b.y + dy));
        er += Math.hypot(best.r.x - (b.r.x + dx), best.r.y - (b.r.y + dy));
        n++;
      }
    }
    const out = { n: base.pts.length, matched: n, integer: n ? ei / n : NaN, refined: n ? er / n : NaN };
    ACC.set(key, out);
    return out;
  }
  return { pipeline, refine, shifted, SHIFTS, accuracy };
})();

(function () {
  const svg = d3.select("#harris-svg");
  if (svg.empty()) return;
  const S = FTscene, HR = FTharris;
  const eS = document.getElementById("hr-stage");
  const eD = document.getElementById("hr-sd"), eDv = document.getElementById("hr-sdv");
  const eI = document.getElementById("hr-si"), eIv = document.getElementById("hr-siv");
  const eK = document.getElementById("hr-k"), eKv = document.getElementById("hr-kv");
  const eT = document.getElementById("hr-t"), eTv = document.getElementById("hr-tv");
  const out = document.getElementById("harris-readout");
  let peak = 0;                                    // which detection the zoom panel shows

  function draw() {
    const stage = +eS.value, sd = +eD.value, si = +eI.value, k = +eK.value, tau = (+eT.value) / 100;
    const P = HR.pipeline(S.A, sd, si, k, tau);
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.5, IW = S.W * cw;

    /* ---- left: the image with the detections ------------------------------ */
    const B1 = FT.box(g, 14, 40, IW, IW, "detections");
    FT.cells(B1, 0, 0, cw, S.W, S.H, (x, y) => FT.grey(S.A[y][x]));
    if (stage >= 5) P.pts.forEach((p, i) => {
      const q = stage >= 6 ? p.r : p;
      B1.append("circle").attr("cx", (q.x + 0.5) * cw).attr("cy", (q.y + 0.5) * cw).attr("r", 3.2)
        .attr("fill", "none").attr("stroke", i === peak ? VC.good : VC.a2)
        .attr("stroke-width", i === peak ? 2 : 1.2).style("cursor", "pointer")
        .on("click", () => { peak = i; draw(); say(); });
    });
    else if (stage === 4) {
      for (let y = 0; y < S.H; y++) for (let x = 0; x < S.W; x++) if (P.mask[y][x])
        B1.append("rect").attr("x", x * cw).attr("y", y * cw).attr("width", cw).attr("height", cw)
          .attr("fill", VC.a2).attr("fill-opacity", 0.55).attr("shape-rendering", "crispEdges");
    }
    B1.append("text").attr("x", 0).attr("y", IW + 14).attr("font-size", 10).attr("fill", VC.muted)
      .text(stage >= 5 ? P.pts.length + " keypoints" : (stage === 4 ? "the thresholded mask" : "stage " + (stage + 1) + " has no detections yet"));

    /* ---- middle: the intermediate belonging to the stage ------------------- */
    const MX = 14 + IW + 26, mw = (IW - 8) / 2;
    const STAGES = [
      { t: "Iₓ  and  I_y", maps: [P.T.gx, P.T.gy], lbl: ["Iₓ", "I_y"] },
      { t: "Iₓ², IₓI_y, I_y²", maps: [P.P11, P.P12, P.P22], lbl: ["Iₓ²", "IₓI_y", "I_y²"] },
      { t: "A₁₁, A₁₂, A₂₂ after G_σᵢ", maps: [P.T.A11, P.T.A12, P.T.A22], lbl: ["A₁₁", "A₁₂", "A₂₂"] },
      { t: "R = det A − k·tr²A", maps: [P.R], lbl: ["R"] },
      { t: "R ≥ τ·max R", maps: [P.mask], lbl: ["mask"] },
      { t: "local maxima of R", maps: [P.R], lbl: ["R"] },
      { t: "R, with refined peaks", maps: [P.R], lbl: ["R"] }
    ][stage];
    g.append("text").attr("x", MX).attr("y", 32).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text(STAGES.t);
    STAGES.maps.forEach((M, i) => {
      const scw = (STAGES.maps.length > 2 ? 1.35 : 1.9);
      const px = MX + (i % 2) * (S.W * scw + 12), py = 40 + Math.floor(i / 2) * (S.H * scw + 20);
      const B = FT.box(g, px, py, S.W * scw, S.H * scw, null);
      let mx = 1e-15;
      for (const r of M) for (const v of r) mx = Math.max(mx, Math.abs(v));
      FT.cells(B, 0, 0, scw, S.W, S.H, (x, y) => FT.signed(M[y][x] / mx));
      B.append("text").attr("x", 0).attr("y", S.H * scw + 11).attr("font-size", 9).attr("fill", VC.muted)
        .text(STAGES.lbl[i] + "  ±" + mx.toExponential(1));
      if (stage === 5) P.pts.forEach(p => B.append("circle").attr("cx", (p.x + 0.5) * scw)
        .attr("cy", (p.y + 0.5) * scw).attr("r", 2).attr("fill", "none")
        .attr("stroke", VC.good).attr("stroke-width", 0.9));
    });

    /* ---- right top: stability against accuracy, measured ------------------- */
    const RX = MX + 240, RW = 214, RH = 116;
    const gc = g.append("g").attr("transform", `translate(${RX},${52})`);
    const sis = [0.8, 1.2, 1.6, 2.0, 2.6, 3.2, 4.0];
    const rows = sis.map(s => Object.assign({ si: s }, HR.accuracy(sd, s, k, tau)));
    const x = d3.scaleLinear().domain([0.6, 4.2]).range([0, RW]);
    const yN = d3.scaleLinear().domain([0, d3.max(rows, r => r.n) * 1.15 + 1]).range([RH, 0]);
    const yE = d3.scaleLinear().domain([0, Math.max(0.35, d3.max(rows, r => r.refined) * 1.2)]).range([RH, 0]);
    VZ.gridY(gc, yN, RW, 4);
    VZ.axisB(gc, x, RH, 5, "σᵢ");
    gc.append("g").attr("class", "axis").call(d3.axisLeft(yN).ticks(4));
    gc.append("g").attr("class", "axis").attr("transform", `translate(${RW},0)`)
      .call(d3.axisRight(yE).ticks(4));
    gc.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10).attr("fill", VC.accent).text("detections");
    gc.append("text").attr("x", RW).attr("y", -8).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", VC.a2).text("mean error (px, refined)");
    gc.append("path").attr("d", d3.line().x(d => x(d.si)).y(d => yN(d.n))(rows))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
    gc.append("path").attr("d", d3.line().x(d => x(d.si)).y(d => yE(d.refined))(rows))
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 2);
    gc.append("path").attr("d", d3.line().x(d => x(d.si)).y(d => yE(d.integer))(rows))
      .attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
    gc.append("text").attr("x", 4).attr("y", yE(rows[0].integer) - 5).attr("font-size", 9)
      .attr("fill", VC.muted).text("integer");
    gc.append("line").attr("x1", x(si)).attr("x2", x(si)).attr("y1", 0).attr("y2", RH)
      .attr("stroke", VC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.6);

    /* ---- right bottom: one peak, magnified, with its quadratic fit --------- */
    const p = P.pts[Math.min(peak, Math.max(0, P.pts.length - 1))];
    const ZY = 230;
    g.append("text").attr("x", RX).attr("y", ZY - 8).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("one peak, 3 × 3, and its fit");
    if (p) {
      const zc = 34, gz = g.append("g").attr("transform", `translate(${RX},${ZY})`);
      let lo = Infinity, hi2 = -Infinity;
      for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
        lo = Math.min(lo, P.R[p.y + b][p.x + a]); hi2 = Math.max(hi2, P.R[p.y + b][p.x + a]);
      }
      for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
        const v = P.R[p.y + b][p.x + a];
        gz.append("rect").attr("x", (a + 1) * zc).attr("y", (b + 1) * zc).attr("width", zc).attr("height", zc)
          .attr("fill", FT.heat((v - lo) / Math.max(1e-24, hi2 - lo))).attr("stroke", VC.line);
        gz.append("text").attr("x", (a + 1.5) * zc).attr("y", (b + 1.5) * zc + 3.5).attr("text-anchor", "middle")
          .attr("font-size", 8.5).attr("font-family", "SF Mono, Menlo, monospace")
          .attr("fill", (v - lo) / Math.max(1e-24, hi2 - lo) > 0.6 ? "#111" : VC.ink)
          .text(v.toExponential(1));
      }
      gz.append("circle").attr("cx", (1.5 + p.r.ox) * zc).attr("cy", (1.5 + p.r.oy) * zc).attr("r", 4)
        .attr("fill", p.r.ok ? VC.good : VC.bad).attr("stroke", VC.bg).attr("stroke-width", 1.2);
      const put = FT.kv(g, RX + 3 * zc + 14, ZY + 16, { keyW: 46, lead: 15, size: 10.5 });
      put("δx", VZ.fmt(p.r.ox, 4), p.r.ok ? VC.good : VC.bad, true);
      put("δy", VZ.fmt(p.r.oy, 4), p.r.ok ? VC.good : VC.bad, true);
      put("‖δ‖", VZ.fmt(Math.hypot(p.r.ox, p.r.oy), 4), VC.ink);
      put("fit", p.r.ok ? "accepted" : "rejected", p.r.ok ? VC.good : VC.bad, true);
      put("at", "(" + p.x + ", " + p.y + ")", VC.muted);
    }
  }

  function say() {
    const sd = +eD.value, si = +eI.value, k = +eK.value, tau = (+eT.value) / 100;
    const P = HR.pipeline(S.A, sd, si, k, tau);
    const acc = HR.accuracy(sd, si, k, tau);
    const lo = HR.accuracy(sd, 0.8, k, tau), hi = HR.accuracy(sd, 4.0, k, tau);
    out.innerHTML =
      `<b>${P.pts.length}</b> keypoints at σ_d = ${sd.toFixed(1)}, σᵢ = ${si.toFixed(1)}, k = ${k.toFixed(3)}, `
      + `τ = ${(100 * tau).toFixed(0)}% of max R = <b>${P.hi.toExponential(3)}</b>. `
      + `Over the 7 known sub-pixel shifts, <b>${acc.matched}</b> matched detections give a mean localisation error `
      + `of <b>${VZ.fmt(acc.integer, 4)} px</b> at integer resolution and <b>${VZ.fmt(acc.refined, 4)} px</b> after `
      + `the quadratic fit — a factor of <b>${VZ.fmt(acc.integer / acc.refined, 2)}</b>.<br>`
      + `Sweeping only σᵢ: at 0.8 the detector finds ${lo.n} points with refined error `
      + `${VZ.fmt(lo.refined, 4)} px; at 4.0 it finds ${hi.n} with ${VZ.fmt(hi.refined, 4)} px. `
      + `The error curve is <b>U-shaped, not monotone</b>: too small an integration window leaves the response `
      + `map noisy and the 3 × 3 quadratic fits noise, while too large a one merges neighbouring corners and `
      + `drags each surviving peak toward the centroid of local gradient energy. The minimum on this scene sits `
      + `near σᵢ = 2, which is where the conventional σᵢ ≈ 2σ_d puts you.`;
  }
  function all() {
    eDv.textContent = (+eD.value).toFixed(1); eIv.textContent = (+eI.value).toFixed(1);
    eKv.textContent = (+eK.value).toFixed(3); eTv.textContent = eT.value + "%";
    draw(); say();
  }
  eS.onchange = all; eD.oninput = all; eI.oninput = all; eK.oninput = all; eT.oninput = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   7 · #anms-svg — strongest-n against adaptive non-maximal suppression
   ══════════════════════════════════════════════════════════════════════════ */
const FTanms = (function () {
  /* a scene whose texture CONTRAST falls from left to right, which is what makes
     "keep the strongest n" cluster. Everything else on the page uses scene A. */
  const W = 128, HH = 96;
  const I = (function () {
    const r = VZ.rng(4242), N = VZ.zeros2(HH, W);
    for (let y = 0; y < HH; y++) for (let x = 0; x < W; x++) N[y][x] = VZ.randn(r);
    const a = VZ.sep2(N, VZ.gauss1(1.1), "mirror");
    const b = VZ.sep2(N, VZ.gauss1(2.4), "mirror");
    const c = VZ.sep2(N, VZ.gauss1(5.0), "mirror");
    const J = VZ.zeros2(HH, W);
    for (let y = 0; y < HH; y++) for (let x = 0; x < W; x++) {
      const amp = 0.10 + 0.85 * Math.pow(1 - x / (W - 1), 1.7);
      J[y][x] = VZ.clamp(0.5 + amp * (2.2 * a[y][x] + 1.5 * b[y][x] + 0.9 * c[y][x]), 0.02, 0.98);
    }
    for (let y = 14; y < 34; y++) for (let x = 92; x < 118; x++)     // a man-made structure
      J[y][x] = (x < 95 || x > 114 || y < 17 || y > 31) ? 0.86 : 0.24;
    return VZ.sep2(J, VZ.gauss1(0.8), "mirror");
  })();

  /* every local maximum of the Harris response with a positive score */
  const CAND = (function () {
    const T = FT.structure(I, 1.0, 2.0, "mirror"), R = VZ.zeros2(HH, W);
    let hi = -Infinity;
    for (let y = 0; y < HH; y++) for (let x = 0; x < W; x++) {
      const a = T.A11[y][x], b = T.A12[y][x], c = T.A22[y][x];
      R[y][x] = (a * c - b * b) - 0.04 * (a + c) * (a + c);
      hi = Math.max(hi, R[y][x]);
    }
    const out = [];
    for (let y = 4; y < HH - 4; y++) for (let x = 4; x < W - 4; x++) {
      const v = R[y][x];
      if (v <= 1e-4 * hi) continue;
      let ok = true;
      for (let b = -1; b <= 1 && ok; b++) for (let a = -1; a <= 1; a++)
        if ((a || b) && R[y + b][x + a] > v) { ok = false; break; }
      if (ok) out.push({ x: x, y: y, v: v });
    }
    out.sort((p, q) => q.v - p.v);
    return out;
  })();

  /* the suppression radius of every candidate, for a given robustness factor c */
  const RCACHE = new Map();
  function ranked(c) {
    if (RCACHE.has(c)) return RCACHE.get(c);
    const out = CAND.map((p, i) => {
      let r = Infinity;
      for (let j = 0; j < i; j++)                       // CAND is sorted, so j < i ⇒ stronger
        if (CAND[j].v > c * p.v) r = Math.min(r, Math.hypot(CAND[j].x - p.x, CAND[j].y - p.y));
      return { x: p.x, y: p.y, v: p.v, r: r };
    });
    const s = out.slice().sort((a, b) => b.r - a.r);
    RCACHE.set(c, s);
    return s;
  }
  const GX = 8, GY = 6;
  function spread(pts) {
    const occ = new Set();
    for (const p of pts) occ.add(Math.floor(p.x / W * GX) + "," + Math.floor(p.y / HH * GY));
    let nn = 0;
    for (const p of pts) {
      let d = Infinity;
      for (const q of pts) if (q !== p) d = Math.min(d, Math.hypot(q.x - p.x, q.y - p.y));
      nn += isFinite(d) ? d : 0;
    }
    return { cells: occ.size, total: GX * GY, meanNN: pts.length > 1 ? nn / pts.length : 0, occ: occ };
  }
  return { W, H: HH, I, CAND, ranked, spread, GX, GY };
})();

(function () {
  const svg = d3.select("#anms-svg");
  if (svg.empty()) return;
  const AN = FTanms;
  const eN = document.getElementById("an-n"), eNv = document.getElementById("an-nv");
  const eC = document.getElementById("an-c"), eCv = document.getElementById("an-cv");
  const eR = document.getElementById("an-r"), eG = document.getElementById("an-g");
  const out = document.getElementById("anms-readout");

  function draw() {
    const n = Math.min(+eN.value, AN.CAND.length), c = +eC.value;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 1.72, IW = AN.W * cw, IH = AN.H * cw;
    const top = AN.CAND.slice(0, n), ad = AN.ranked(c).slice(0, n);
    const st = AN.spread(top), sa = AN.spread(ad);

    function panel(x0, title, pts, sp, col) {
      const B = FT.box(g, x0, 42, IW, IH, title);
      FT.cells(B, 0, 0, cw, AN.W, AN.H, (x, y) => FT.grey(AN.I[y][x]));
      if (eG.checked) {
        for (let j = 0; j < AN.GY; j++) for (let i = 0; i < AN.GX; i++) {
          const on = sp.occ.has(i + "," + j);
          B.append("rect").attr("x", i * IW / AN.GX).attr("y", j * IH / AN.GY)
            .attr("width", IW / AN.GX).attr("height", IH / AN.GY)
            .attr("fill", on ? VC.good : VC.bad).attr("fill-opacity", on ? 0.10 : 0.16)
            .attr("stroke", VC.line).attr("stroke-width", 0.6);
        }
      }
      if (eR.checked) pts.forEach(p => {
        if (!isFinite(p.r)) return;
        B.append("circle").attr("cx", (p.x + 0.5) * cw).attr("cy", (p.y + 0.5) * cw)
          .attr("r", p.r * cw).attr("fill", "none").attr("stroke", col)
          .attr("stroke-opacity", 0.28).attr("stroke-width", 0.8);
      });
      pts.forEach(p => B.append("circle").attr("cx", (p.x + 0.5) * cw).attr("cy", (p.y + 0.5) * cw)
        .attr("r", 2.4).attr("fill", "none").attr("stroke", col).attr("stroke-width", 1.3));
      B.append("text").attr("x", 0).attr("y", IH + 13).attr("font-size", 10).attr("fill", VC.muted)
        .text(sp.cells + " of " + sp.total + " cells · mean spacing " + sp.meanNN.toFixed(2) + " px");
      return B;
    }
    panel(14, "strongest " + n, top, st, VC.a2);
    panel(14 + IW + 26, "ANMS " + n + ",  c = " + c.toFixed(2), ad, sa, VC.good);

    /* ---- right: coverage and spacing against n ---------------------------- */
    const RX = 14 + 2 * IW + 56, RW = 178, RH = 130;
    const NS = [];
    for (let k = 10; k <= AN.CAND.length; k += 10) NS.push(k);
    if (NS[NS.length - 1] !== AN.CAND.length) NS.push(AN.CAND.length);
    const rows = NS.map(k => ({
      n: k,
      t: AN.spread(AN.CAND.slice(0, k)),
      a: AN.spread(AN.ranked(c).slice(0, k))
    }));
    const x = d3.scaleLinear().domain([0, AN.CAND.length]).range([0, RW]);
    function chart(y0, title, get, fmt, dom) {
      const gc = g.append("g").attr("transform", `translate(${RX},${y0})`);
      const y = d3.scaleLinear().domain(dom).range([RH, 0]);
      VZ.gridY(gc, y, RW, 4);
      VZ.axisB(gc, x, RH, 4, "n");
      gc.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4).tickFormat(fmt));
      gc.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.ink).text(title);
      [["t", VC.a2], ["a", VC.good]].forEach(([k2, col]) => {
        gc.append("path").attr("d", d3.line().x(d => x(d.n)).y(d => y(get(d[k2])))(rows))
          .attr("fill", "none").attr("stroke", col).attr("stroke-width", 2);
      });
      gc.append("line").attr("x1", x(n)).attr("x2", x(n)).attr("y1", 0).attr("y2", RH)
        .attr("stroke", VC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.6);
      return gc;
    }
    chart(52, "grid cells occupied", d => d.cells, d3.format("d"), [0, AN.GX * AN.GY]);
    chart(52 + RH + 52, "mean spacing (px)", d => d.meanNN, d3.format(".0f"),
      [0, d3.max(rows, r => Math.max(r.t.meanNN, r.a.meanNN)) * 1.15]);
    VZ.legend(g, [{ color: VC.a2, label: "strongest n" }, { color: VC.good, label: "ANMS" }],
      RX + 4, 30, { gap: 0, vertical: false, step: 92 });
  }

  function say() {
    const n = Math.min(+eN.value, AN.CAND.length), c = +eC.value;
    const ad = AN.ranked(c);
    const st = AN.spread(AN.CAND.slice(0, n)), sa = AN.spread(ad.slice(0, n));
    const rn = ad[n - 1] ? ad[n - 1].r : Infinity;
    out.innerHTML =
      `From the same <b>${AN.CAND.length}</b> candidate maxima, keeping <b>${n}</b>: `
      + `strongest-n occupies <b>${st.cells} of ${st.total}</b> grid cells with mean spacing `
      + `<b>${st.meanNN.toFixed(2)} px</b>; ANMS occupies <b>${sa.cells} of ${sa.total}</b> with mean spacing `
      + `<b>${sa.meanNN.toFixed(2)} px</b>. The n-th suppression radius is `
      + `<b>${isFinite(rn) ? rn.toFixed(2) + " px" : "∞"}</b> — a direct statement of the spacing the selection `
      + `guarantees.<br>`
      + (n >= AN.CAND.length - 5
        ? `At this n you are keeping essentially every candidate, so the two rules must agree: ANMS is a `
          + `<i>re-ordering</i> of one list, not a different detector, and re-ordering only matters while you are `
          + `discarding something.`
        : `Push n toward ${AN.CAND.length} and the two curves converge, because both rules end up returning the `
          + `same set. The gap between them is exactly what you are throwing away by ranking on strength.`);
  }
  function all() {
    eNv.textContent = eN.value; eCv.textContent = (+eC.value).toFixed(2);
    draw(); say();
  }
  eN.oninput = all; eC.oninput = all; eR.onchange = all; eG.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   the Canny pipeline — shared by figures 8, 9, 10 and 11
   ══════════════════════════════════════════════════════════════════════════ */
const FTcanny = (function () {
  const S = FTscene, W = S.EW, H = S.EH;

  function stages(sigma, noise, lo, hi, useNMS) {
    const src = FT.addNoise(S.B, noise, 11);
    const sm = VZ.sep2(src, VZ.gauss1(sigma), "mirror");
    const G = FT.grads(sm, 1.0, "mirror");        // Scharr-class DoG derivative, part 2 §13
    const mag = VZ.zeros2(H, W), ang = VZ.zeros2(H, W);
    let mx = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const m = Math.hypot(G.gx[y][x], G.gy[y][x]);
      mag[y][x] = m; ang[y][x] = Math.atan2(G.gy[y][x], G.gx[y][x]);
      if (m > mx) mx = m;
    }
    const thin = useNMS ? nms(mag, G.gx, G.gy) : mag;
    const E = hysteresis(thin, lo * mx, hi * mx);
    return { src, sm, gx: G.gx, gy: G.gy, mag, ang, mx, thin, E };
  }

  /* non-maximum suppression: compare the magnitude against BILINEARLY
     INTERPOLATED magnitudes one pixel forward and back along the gradient */
  function nms(mag, gx, gy) {
    const out = VZ.zeros2(H, W);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const m = mag[y][x];
      if (m <= 0) continue;
      const ux = gx[y][x] / m, uy = gy[y][x] / m;
      const a = FT.bilerp(mag, y + uy, x + ux), b = FT.bilerp(mag, y - uy, x - ux);
      if (m >= a && m >= b) out[y][x] = m;
    }
    return out;
  }
  /* the eight-direction rounding variant, kept for the comparison in figure 9 */
  function nmsQuant(mag, gx, gy) {
    const out = VZ.zeros2(H, W);
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      let a = (Math.atan2(gy[y][x], gx[y][x]) * 180 / Math.PI + 180) % 180, p, q;
      if (a < 22.5 || a >= 157.5) { p = mag[y][x - 1]; q = mag[y][x + 1]; }
      else if (a < 67.5) { p = mag[y - 1][x - 1]; q = mag[y + 1][x + 1]; }
      else if (a < 112.5) { p = mag[y - 1][x]; q = mag[y + 1][x]; }
      else { p = mag[y - 1][x + 1]; q = mag[y + 1][x - 1]; }
      if (mag[y][x] >= p && mag[y][x] >= q) out[y][x] = mag[y][x];
    }
    return out;
  }
  /* hysteresis: every 8-connected component containing at least one strong pixel
     is kept whole. Implemented as a flood fill from the strong set. */
  function hysteresis(M, lo, hi) {
    const E = VZ.zeros2(H, W), st = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
      if (M[y][x] >= hi) { E[y][x] = 2; st.push([x, y]); }
    while (st.length) {
      const [x, y] = st.pop();
      for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
        const nx = x + a, ny = y + b;
        if (nx < 0 || ny < 0 || nx >= W || ny >= H || E[ny][nx]) continue;
        if (M[ny][nx] >= lo) { E[ny][nx] = 1; st.push([nx, ny]); }
      }
    }
    return E;                                     // 2 = strong seed, 1 = recruited weak
  }
  function counts(M) {
    let strong = 0, weak = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (M[y][x] === 2) strong++; else if (M[y][x] === 1) weak++;
    }
    return { strong: strong, weak: weak, total: strong + weak };
  }
  /* how many 8-connected pieces the edge map falls into — the number hysteresis
     is really trying to reduce */
  function components(E) {
    const lab = VZ.zeros2(H, W); let n = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (!E[y][x] || lab[y][x]) continue;
      n++; const st = [[x, y]]; lab[y][x] = n;
      while (st.length) {
        const [px, py] = st.pop();
        for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
          const nx = px + a, ny = py + b;
          if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
          if (E[ny][nx] && !lab[ny][nx]) { lab[ny][nx] = n; st.push([nx, ny]); }
        }
      }
    }
    return n;
  }
  /* the arc that fades: how much of it survives. Ground truth from the scene's
     own construction, so this is a real recall measurement and not an impression. */
  function arcMask() {
    const M = VZ.zeros2(H, W);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const dx = x - 36, dy = y - 56, rr = Math.hypot(dx, dy), th = Math.atan2(dy, dx);
      if (Math.abs(rr - 22) < 3.2 && th > -Math.PI * 0.98 && th < -0.10) M[y][x] = 1;
    }
    return M;
  }
  const ARC = arcMask();
  function arcRecall(E) {
    /* fraction of the arc's angular extent that has an edge pixel on it */
    const BINS = 40, hit = new Array(BINS).fill(0);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (!E[y][x] || !ARC[y][x]) continue;
      const th = Math.atan2(y - 56, x - 36);
      const t = (th + Math.PI) / (Math.PI - 0.10);
      const b = Math.floor(VZ.clamp(t, 0, 0.999) * BINS);
      hit[b] = 1;
    }
    return hit.reduce((a, b) => a + b, 0) / BINS;
  }
  return { W, H, stages, nms, nmsQuant, hysteresis, counts, components, ARC, arcRecall };
})();

/* ground truth for scene B, from the way the scene was constructed — so precision
   is a measurement rather than an impression. Shared by figures 8, 10 and 11. */
const FTgt = (function () {
  const C = FTcanny, M = VZ.zeros2(C.H, C.W);
  for (let y = 0; y < C.H; y++) for (let x = 0; x < C.W; x++) {
    let t = false;
    const dsq = Math.max(Math.max(8 - x, x - 34), Math.max(8 - y, y - 30));
    if (Math.abs(dsq) <= 2.5 && x >= 5 && x <= 37 && y >= 5 && y <= 33) t = true;
    if (Math.abs(Math.hypot(x - 56, y - 18) - 10) <= 2.5) t = true;
    const dx = x - 36, dy = y - 56, rr = Math.hypot(dx, dy), th = Math.atan2(dy, dx);
    if (Math.abs(rr - 22) < 3.5 && th > -Math.PI * 0.98 && th < -0.10) t = true;
    M[y][x] = t ? 1 : 0;
  }
  function score(E) {
    let tp = 0, fp = 0;
    for (let y = 0; y < C.H; y++) for (let x = 0; x < C.W; x++)
      if (E[y][x]) { if (M[y][x]) tp++; else fp++; }
    const rec = C.arcRecall(E), prec = (tp + fp) ? tp / (tp + fp) : 1;
    return { tp, fp, prec, rec, f1: (prec + rec) ? 2 * prec * rec / (prec + rec) : 0,
      comp: C.components(E) };
  }
  return { M, score };
})();

/* ══════════════════════════════════════════════════════════════════════════
   8 · #canny-svg — the pipeline, stage by stage
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#canny-svg");
  if (svg.empty()) return;
  const C = FTcanny;
  const eS = document.getElementById("cn-stage");
  const eSg = document.getElementById("cn-s"), eSgv = document.getElementById("cn-sv");
  const eN = document.getElementById("cn-n"), eNv = document.getElementById("cn-nv");
  const eL = document.getElementById("cn-lo"), eLv = document.getElementById("cn-lov");
  const eH = document.getElementById("cn-hi"), eHv = document.getElementById("cn-hiv");
  const eM = document.getElementById("cn-nms");
  const out = document.getElementById("canny-readout");

  function get() {
    const sg = +eSg.value, nz = +eN.value;
    let lo = +eL.value, hi = +eH.value;
    if (lo > hi) lo = hi;                          // the low threshold cannot exceed the high one
    return { sg, nz, lo, hi, S: C.stages(sg, nz, lo, hi, eM.checked) };
  }

  function draw() {
    const st = get(), stage = +eS.value, S = st.S;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 3.0, IW = C.W * cw;

    const B1 = FT.box(g, 14, 40, IW, IW, "input, with noise " + st.nz.toFixed(2));
    FT.cells(B1, 0, 0, cw, C.W, C.H, (x, y) => FT.grey(S.src[y][x]));

    const MX = 14 + IW + 26;
    const titles = ["1 · G_σ ∗ I", "2 · ‖∇I‖", "2b · ∠∇I", "3 · after NMS", "4 · after hysteresis"];
    const B2 = FT.box(g, MX, 40, IW, IW, titles[stage]);
    if (stage === 0) FT.cells(B2, 0, 0, cw, C.W, C.H, (x, y) => FT.grey(S.sm[y][x]));
    else if (stage === 1) FT.cells(B2, 0, 0, cw, C.W, C.H, (x, y) => FT.heat(S.mag[y][x] / S.mx));
    else if (stage === 2) FT.cells(B2, 0, 0, cw, C.W, C.H, (x, y) => {
      const a = (S.ang[y][x] + Math.PI) / (2 * Math.PI);
      const c = d3.hsl(360 * a, 0.65, 0.25 + 0.45 * (S.mag[y][x] / S.mx));
      return c + "";
    });
    else if (stage === 3) FT.cells(B2, 0, 0, cw, C.W, C.H, (x, y) => FT.heat(S.thin[y][x] / S.mx));
    else {
      FT.cells(B2, 0, 0, cw, C.W, C.H, () => "rgb(16,18,25)");
      for (let y = 0; y < C.H; y++) for (let x = 0; x < C.W; x++) if (S.E[y][x])
        B2.append("rect").attr("x", x * cw).attr("y", y * cw).attr("width", cw).attr("height", cw)
          .attr("fill", S.E[y][x] === 2 ? VC.a2 : VC.good).attr("shape-rendering", "crispEdges");
      VZ.legend(B2, [{ color: VC.a2, label: "strong (≥ τ_hi)" }, { color: VC.good, label: "recruited weak" }],
        4, IW - 26, { gap: 13 });
    }

    /* ---- right: the three measurements, against both degenerate settings --- */
    const RX = MX + IW + 30;
    const rows = [
      { n: "hysteresis  " + st.lo.toFixed(2) + " / " + st.hi.toFixed(2), E: S.E, col: VC.good },
      { n: "one high threshold  " + st.hi.toFixed(2), E: C.hysteresis(S.thin, st.hi * S.mx, st.hi * S.mx), col: VC.a2 },
      { n: "one low threshold  " + st.lo.toFixed(2), E: C.hysteresis(S.thin, st.lo * S.mx, st.lo * S.mx), col: VC.violet }
    ].map(r => Object.assign(r, C.counts(r.E), FTgt.score(r.E)));
    g.append("text").attr("x", RX).attr("y", 34).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("the same ridge, thresholded three ways");
    let yy = 56;
    rows.forEach(r => {
      g.append("text").attr("x", RX).attr("y", yy).attr("font-size", 11).attr("fill", r.col)
        .attr("font-weight", 600).text(r.n);
      const put = FT.kv(g, RX + 6, yy + 17, { keyW: 128, lead: 14.5, size: 10.5 });
      put("edge pixels", r.total + (r.weak ? "  (" + r.weak + " recruited)" : ""), VC.ink, true);
      put("spurious", r.fp, r.fp > 20 ? VC.bad : VC.good);
      put("precision", VZ.fmt(r.prec, 3), r.prec > 0.9 ? VC.good : VC.bad, true);
      put("arc recall", VZ.fmt(r.rec, 3), r.rec > 0.85 ? VC.good : VC.bad, true);
      put("components", r.comp, r.comp > 10 ? VC.bad : VC.good, true);
      yy += 118;
    });
  }

  function say() {
    const st = get(), S = st.S;
    const A = FTgt.score(S.E), c = C.counts(S.E);
    const Hi = C.hysteresis(S.thin, st.hi * S.mx, st.hi * S.mx), sHi = FTgt.score(Hi);
    const Lo = C.hysteresis(S.thin, st.lo * S.mx, st.lo * S.mx), sLo = FTgt.score(Lo);
    let above = 0, thin = 0;
    for (let y = 0; y < C.H; y++) for (let x = 0; x < C.W; x++) {
      if (S.mag[y][x] > 0.02 * S.mx) above++;
      if (S.thin[y][x] > 0.02 * S.mx) thin++;
    }
    out.innerHTML =
      `At <span class="keep">σ</span> = ${st.sg.toFixed(1)}, noise ${st.nz.toFixed(2)}, thresholds `
      + `${st.lo.toFixed(2)} / ${st.hi.toFixed(2)}: <b>${c.total}</b> edge pixels of which `
      + `<b>${c.weak}</b> were recruited by hysteresis, precision <b>${VZ.fmt(A.prec, 3)}</b>, `
      + `arc recall <b>${VZ.fmt(A.rec, 3)}</b>, <b>${A.comp}</b> connected components. `
      + (eM.checked
        ? `Suppression thinned ${above} above-threshold gradient pixels to ${thin}, a factor of `
          + `<b>${VZ.fmt(above / thin, 2)}</b>.`
        : `<b>Non-maximum suppression is off</b>, so what you are looking at is a thresholded ridge, `
          + `several pixels wide, and every downstream stage that expects a curve will fail on it.`)
      + `<br>Against the two single-threshold alternatives on the same ridge: the high threshold alone gives `
      + `precision ${VZ.fmt(sHi.prec, 3)} at arc recall ${VZ.fmt(sHi.rec, 3)} in ${sHi.comp} components; the low `
      + `threshold alone gives precision ${VZ.fmt(sLo.prec, 3)} at arc recall ${VZ.fmt(sLo.rec, 3)} in `
      + `${sLo.comp} components. Hysteresis is not a compromise between them — it takes most of the low `
      + `threshold's recall and most of the high threshold's precision, because "connected to something certain" `
      + `is genuinely more information than either threshold has on its own.`;
  }
  function all() {
    eSgv.textContent = (+eSg.value).toFixed(1); eNv.textContent = (+eN.value).toFixed(2);
    eLv.textContent = (+eL.value).toFixed(2); eHv.textContent = (+eH.value).toFixed(2);
    draw(); say();
  }
  eS.onchange = all; eSg.oninput = all; eN.oninput = all; eL.oninput = all;
  eH.oninput = all; eM.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   9 · #nms-svg — the interpolation, close up
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#nms-svg");
  if (svg.empty()) return;
  const C = FTcanny;
  const eV = document.getElementById("nm-v");
  const eS = document.getElementById("nm-s"), eSv = document.getElementById("nm-sv");
  const eN = document.getElementById("nm-n"), eNv = document.getElementById("nm-nv");
  const eP = document.getElementById("nm-p");
  const out = document.getElementById("nms-readout");
  const PROBES = { diag: [22, 42], vert: [35, 20], arc: [57, 53] };
  let P = PROBES.diag.slice(), custom = false;

  function get() {
    const sg = +eS.value, nz = +eN.value;
    const S = C.stages(sg, nz, 0.12, 0.40, true);
    return { sg, nz, S, A: C.nms(S.mag, S.gx, S.gy), B: C.nmsQuant(S.mag, S.gx, S.gy) };
  }
  /* the decision at one pixel, with the two comparison points made explicit */
  function decide(S, x, y) {
    const m = S.mag[y][x];
    if (m <= 0) return null;
    const ux = S.gx[y][x] / m, uy = S.gy[y][x] / m;
    const fwd = FT.bilerp(S.mag, y + uy, x + ux), bwd = FT.bilerp(S.mag, y - uy, x - ux);
    let a = (Math.atan2(S.gy[y][x], S.gx[y][x]) * 180 / Math.PI + 180) % 180, qa, qb, qd;
    if (a < 22.5 || a >= 157.5) { qd = [1, 0]; }
    else if (a < 67.5) { qd = [1, 1]; }
    else if (a < 112.5) { qd = [0, 1]; }
    else { qd = [-1, 1]; }
    qa = S.mag[y + qd[1]] ? S.mag[y + qd[1]][x + qd[0]] : 0;
    qb = S.mag[y - qd[1]] ? S.mag[y - qd[1]][x - qd[0]] : 0;
    return { m, ux, uy, fwd, bwd, keep: m >= fwd && m >= bwd, qa, qb, qd,
      qkeep: m >= qa && m >= qb, ang: Math.atan2(S.gy[y][x], S.gx[y][x]) };
  }

  function draw() {
    const st = get(), S = st.S, mode = eV.value;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.5, IW = C.W * cw;

    /* ---- left: the gradient magnitude, with the probe -------------------- */
    const B1 = FT.box(g, 14, 40, IW, IW, "‖∇I‖ · click to move the probe");
    FT.cells(B1, 0, 0, cw, C.W, C.H, (x, y) => FT.heat(S.mag[y][x] / S.mx));
    B1.append("rect").attr("width", IW).attr("height", IW).attr("fill", "transparent")
      .style("cursor", "crosshair").on("click", ev => {
        const [mx, my] = d3.pointer(ev, B1.node());
        P = [VZ.clamp(Math.round(mx / cw), 5, C.W - 6), VZ.clamp(Math.round(my / cw), 5, C.H - 6)];
        custom = true; draw(); say();
      });
    const D = decide(S, P[0], P[1]);
    B1.append("rect").attr("x", (P[0] - 4) * cw).attr("y", (P[1] - 4) * cw)
      .attr("width", 9 * cw).attr("height", 9 * cw).attr("fill", "none")
      .attr("stroke", VC.ink).attr("stroke-width", 1.2);

    /* ---- middle: the 9 × 9 neighbourhood, magnified ---------------------- */
    const MX = 14 + IW + 26, zc = 20;
    const B2 = FT.box(g, MX, 40, 9 * zc, 9 * zc, "the 9 × 9 around the probe");
    let lo = Infinity, hi = -Infinity;
    for (let b = -4; b <= 4; b++) for (let a = -4; a <= 4; a++) {
      lo = Math.min(lo, S.mag[P[1] + b][P[0] + a]); hi = Math.max(hi, S.mag[P[1] + b][P[0] + a]);
    }
    for (let b = -4; b <= 4; b++) for (let a = -4; a <= 4; a++) {
      const v = S.mag[P[1] + b][P[0] + a];
      B2.append("rect").attr("x", (a + 4) * zc).attr("y", (b + 4) * zc)
        .attr("width", zc).attr("height", zc)
        .attr("fill", FT.heat((v - lo) / Math.max(1e-12, hi - lo)))
        .attr("stroke", VC.line).attr("stroke-width", 0.4);
    }
    if (D) {
      const cx = 4.5 * zc, cy = 4.5 * zc;
      VZ.arrow(B2, cx, cy, cx + 2.2 * zc * D.ux, cy + 2.2 * zc * D.uy, { color: VC.ink, w: 2, head: 6 });
      if (mode !== "quant") {
        [[D.ux, D.uy, D.fwd], [-D.ux, -D.uy, D.bwd]].forEach(([dx, dy, v]) => {
          B2.append("circle").attr("cx", cx + dx * zc).attr("cy", cy + dy * zc).attr("r", 5)
            .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
          B2.append("text").attr("x", cx + dx * zc).attr("y", cy + dy * zc - 9).attr("text-anchor", "middle")
            .attr("font-size", 9).attr("font-family", "SF Mono, Menlo, monospace")
            .attr("fill", VC.accent).text(v.toFixed(4));
        });
      }
      if (mode !== "bilin") {
        [[D.qd[0], D.qd[1], D.qa], [-D.qd[0], -D.qd[1], D.qb]].forEach(([dx, dy, v]) => {
          B2.append("rect").attr("x", cx + dx * zc - 6).attr("y", cy + dy * zc - 6)
            .attr("width", 12).attr("height", 12).attr("fill", "none")
            .attr("stroke", VC.violet).attr("stroke-width", 2);
          B2.append("text").attr("x", cx + dx * zc).attr("y", cy + dy * zc + 17).attr("text-anchor", "middle")
            .attr("font-size", 9).attr("font-family", "SF Mono, Menlo, monospace")
            .attr("fill", VC.violet).text(v.toFixed(4));
        });
      }
      B2.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 3.5).attr("fill", VC.good);
    }
    /* the magnitude profile along the gradient */
    const PY = 40 + 9 * zc + 40, PW2 = 9 * zc, PH = 78;
    const gp = g.append("g").attr("transform", `translate(${MX},${PY})`);
    g.append("text").attr("x", MX).attr("y", PY - 8).attr("font-size", 10.5).attr("fill", VC.muted)
      .text("‖∇I‖ sampled along ∇I through the probe");
    if (D) {
      const ts = VZ.linspace(-4, 4, 81);
      const vs = ts.map(t => FT.bilerp(S.mag, P[1] + t * D.uy, P[0] + t * D.ux));
      const x = d3.scaleLinear().domain([-4, 4]).range([0, PW2]);
      const y = d3.scaleLinear().domain([0, Math.max(1e-9, d3.max(vs)) * 1.15]).range([PH, 0]);
      VZ.axisB(gp, x, PH, 5, "px along ∇I");
      gp.append("path").attr("d", d3.line().x((d, i) => x(ts[i])).y(d => y(d))(vs))
        .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 2);
      [[-1, D.bwd], [0, D.m], [1, D.fwd]].forEach(([t, v]) =>
        gp.append("circle").attr("cx", x(t)).attr("cy", y(v)).attr("r", 3.2)
          .attr("fill", t === 0 ? VC.good : VC.accent));
    }

    /* ---- right: the suppressed map, and where the variants disagree ------ */
    const RX = MX + 9 * zc + 30, cw3 = 2.3, IW3 = C.W * cw3;
    const B3 = FT.box(g, RX, 40, IW3, IW3, mode === "quant" ? "8-direction rounding" :
      (mode === "both" ? "disagreements" : "bilinear"));
    const M = mode === "quant" ? st.B : st.A;
    FT.cells(B3, 0, 0, cw3, C.W, C.H, (x, y) => {
      if (mode === "both") {
        const a = st.A[y][x] > 0, b = st.B[y][x] > 0;
        if (a && b) return "rgb(58,64,80)";
        if (a) return VC.accent;
        if (b) return VC.violet;
        return "rgb(16,18,25)";
      }
      return M[y][x] > 0 ? FT.heat(0.25 + 0.75 * M[y][x] / S.mx) : "rgb(16,18,25)";
    });
    if (mode === "both") VZ.legend(B3, [{ color: "rgb(58,64,80)", label: "both keep" },
      { color: VC.accent, label: "bilinear only" }, { color: VC.violet, label: "rounding only" }],
      4, IW3 - 40, { gap: 13 });
    let agree = 0, tot = 0;
    for (let y = 1; y < C.H - 1; y++) for (let x = 1; x < C.W - 1; x++) {
      tot++; if ((st.A[y][x] > 0) === (st.B[y][x] > 0)) agree++;
    }
    const put = FT.kv(g, RX, 40 + IW3 + 26, { keyW: 132, lead: 15.5 });
    put("variants agree on", (100 * agree / tot).toFixed(2) + "% of pixels", VC.ink, true);
    put("disagreements", (tot - agree), (tot - agree) > 200 ? VC.a2 : VC.good, true);
    if (D) {
      put("‖∇I‖ at the probe", VZ.fmt(D.m, 5), VC.good, true);
      put("interpolated forward", VZ.fmt(D.fwd, 5), D.m >= D.fwd ? VC.good : VC.bad);
      put("interpolated backward", VZ.fmt(D.bwd, 5), D.m >= D.bwd ? VC.good : VC.bad);
      put("bilinear decision", D.keep ? "KEEP" : "suppress", D.keep ? VC.good : VC.bad, true);
      put("rounding decision", D.qkeep ? "KEEP" : "suppress", D.qkeep ? VC.good : VC.bad, true);
      put("∠∇I", VZ.fmt(VZ.deg(D.ang), 1) + "°", VC.muted);
    }
  }

  function say() {
    const st = get(), S = st.S, D = decide(S, P[0], P[1]);
    let agree = 0, tot = 0;
    for (let y = 1; y < C.H - 1; y++) for (let x = 1; x < C.W - 1; x++) {
      tot++; if ((st.A[y][x] > 0) === (st.B[y][x] > 0)) agree++;
    }
    out.innerHTML = D
      ? `At (${P[0]}, ${P[1]}) the gradient points at <b>${VZ.fmt(VZ.deg(D.ang), 1)}°</b> with magnitude `
        + `<b>${VZ.fmt(D.m, 5)}</b>. One pixel forward along it the interpolated magnitude is `
        + `<b>${VZ.fmt(D.fwd, 5)}</b>, one back it is <b>${VZ.fmt(D.bwd, 5)}</b> — so the pixel is `
        + `<b>${D.keep ? "kept" : "suppressed"}</b>. The rounding variant compares against the pixel centres at `
        + `(${D.qd[0]}, ${D.qd[1]}), values <b>${VZ.fmt(D.qa, 5)}</b> and <b>${VZ.fmt(D.qb, 5)}</b>, and decides `
        + `<b>${D.qkeep ? "keep" : "suppress"}</b>${D.keep === D.qkeep ? " — the same" : " — differently"}.<br>`
        + `Over the whole interior the two agree on <b>${(100 * agree / tot).toFixed(2)}%</b> of pixels. `
        + `Increase the noise or decrease <span class="keep">σ</span> and the agreement falls, because the `
        + `gradient <i>direction</i> becomes unreliable before its magnitude does, and rounding a direction that `
        + `is already wrong by 20° to the nearest 45° is a second error on top of the first.`
      : "The gradient vanishes at this pixel, so there is no direction to compare along and nothing to decide.";
  }
  function all() {
    if (!custom) P = PROBES[eP.value].slice();
    eSv.textContent = (+eS.value).toFixed(1); eNv.textContent = (+eN.value).toFixed(2);
    draw(); say();
  }
  eV.onchange = all; eS.oninput = all; eN.oninput = all;
  eP.onchange = () => { custom = false; all(); };
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   10 · #hyst-svg — the (τ_lo, τ_hi) plane
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#hyst-svg");
  if (svg.empty()) return;
  const C = FTcanny;
  const eS = document.getElementById("hy-s"), eSv = document.getElementById("hy-sv");
  const eN = document.getElementById("hy-n"), eNv = document.getElementById("hy-nv");
  const eM = document.getElementById("hy-m");
  const out = document.getElementById("hyst-readout");
  let TL = 0.12, TH = 0.40;
  const NG = 26, TMAX = 0.65;
  const PLANE = new Map();

  function plane(sg, nz) {
    const key = sg + "|" + nz;
    if (PLANE.has(key)) return PLANE.get(key);
    const S = C.stages(sg, nz, 0.1, 0.4, true), grid = [];
    for (let j = 0; j < NG; j++) {                 // j = high threshold index (row)
      grid.push([]);
      for (let i = 0; i < NG; i++) {
        const lo = TMAX * (i + 0.5) / NG, hi = TMAX * (NG - j - 0.5) / NG;
        if (hi < lo) { grid[j].push(null); continue; }
        const E = C.hysteresis(S.thin, lo * S.mx, hi * S.mx);
        grid[j].push(Object.assign({ lo, hi }, C.counts(E), FTgt.score(E)));
      }
    }
    const r = { S: S, grid: grid };
    PLANE.set(key, r);
    return r;
  }
  const METRIC = {
    f1: { get: c => c.f1, name: "F on the arc", fmt: v => VZ.fmt(v, 3), big: true },
    rec: { get: c => c.rec, name: "arc recall", fmt: v => VZ.fmt(v, 3), big: true },
    prec: { get: c => c.prec, name: "precision", fmt: v => VZ.fmt(v, 3), big: true },
    comp: { get: c => c.comp, name: "components", fmt: v => v, big: false }
  };

  function draw() {
    const sg = +eS.value, nz = +eN.value, mk = eM.value, M = METRIC[mk];
    const PL = plane(sg, nz), S = PL.S;
    if (TL > TH) TL = TH;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.8, IW = C.W * cw;
    const E = C.hysteresis(S.thin, TL * S.mx, TH * S.mx);
    const cnt = C.counts(E);

    /* ---- left: the edge map, strong against recruited --------------------- */
    const B1 = FT.box(g, 14, 40, IW, IW, "τ_lo = " + TL.toFixed(2) + ", τ_hi = " + TH.toFixed(2));
    FT.cells(B1, 0, 0, cw, C.W, C.H, (x, y) =>
      E[y][x] === 2 ? VC.a2 : (E[y][x] === 1 ? VC.good : "rgb(16,18,25)"));
    VZ.legend(B1, [{ color: VC.a2, label: cnt.strong + " strong" },
      { color: VC.good, label: cnt.weak + " recruited" }], 5, IW - 26, { gap: 13 });

    /* ---- middle: the threshold plane -------------------------------------- */
    const MX = 14 + IW + 30, PS = 210, pc = PS / NG;
    const B2 = FT.box(g, MX, 40, PS, PS, "the (τ_lo, τ_hi) plane · " + M.name);
    let lo = Infinity, hi = -Infinity;
    for (const row of PL.grid) for (const c of row) if (c) { const v = M.get(c); lo = Math.min(lo, v); hi = Math.max(hi, v); }
    FT.cells(B2, 0, 0, pc, NG, NG, (i, j) => {
      const c = PL.grid[j][i];
      if (!c) return "rgb(14,16,22)";
      let t = (M.get(c) - lo) / Math.max(1e-12, hi - lo);
      if (!M.big) t = 1 - t;
      return FT.heat(t);
    });
    B2.append("line").attr("x1", 0).attr("y1", PS).attr("x2", PS).attr("y2", 0)
      .attr("stroke", VC.ink).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
    B2.append("text").attr("x", PS - 6).attr("y", 12).attr("text-anchor", "end").attr("font-size", 9.5)
      .attr("fill", VC.ink).text("τ_lo = τ_hi : one threshold");
    const sx = d3.scaleLinear().domain([0, TMAX]).range([0, PS]);
    const sy = d3.scaleLinear().domain([0, TMAX]).range([PS, 0]);
    VZ.axisB(B2, sx, PS, 4, "τ_lo");
    B2.append("g").attr("class", "axis").call(d3.axisLeft(sy).ticks(4));
    B2.append("text").attr("x", -30).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.muted).text("τ_hi");
    /* the best cell under this metric */
    let best = null;
    for (const row of PL.grid) for (const c of row) if (c && (!best || (M.big ? M.get(c) > M.get(best) : M.get(c) < M.get(best)))) best = c;
    if (best) {
      B2.append("circle").attr("cx", sx(best.lo)).attr("cy", sy(best.hi)).attr("r", 5)
        .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 2);
      B2.append("text").attr("x", sx(best.lo) + 8).attr("y", sy(best.hi) - 6).attr("font-size", 9.5)
        .attr("fill", VC.good).text("best " + M.fmt(M.get(best)));
    }
    const knob = B2.append("circle").attr("cx", sx(TL)).attr("cy", sy(TH)).attr("r", 6)
      .attr("fill", VC.violet).attr("stroke", VC.bg).attr("stroke-width", 1.5).style("cursor", "grab");
    knob.call(d3.drag().on("drag", ev => {
      const [mx, my] = d3.pointer(ev, B2.node());
      TL = VZ.clamp(sx.invert(mx), 0.01, TMAX);
      TH = VZ.clamp(sy.invert(my), 0.01, TMAX);
      if (TH < TL) TH = TL;
      draw(); say();
    }));

    /* ---- right: the three settings side by side --------------------------- */
    const RX = MX + PS + 34;
    const Ehi = C.hysteresis(S.thin, TH * S.mx, TH * S.mx);
    const Elo = C.hysteresis(S.thin, TL * S.mx, TL * S.mx);
    const rows = [
      { n: "hysteresis", E: E, col: VC.good },
      { n: "τ_hi alone", E: Ehi, col: VC.a2 },
      { n: "τ_lo alone", E: Elo, col: VC.violet }
    ].map(r => Object.assign(r, C.counts(r.E), FTgt.score(r.E)));
    g.append("text").attr("x", RX).attr("y", 34).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("against both single thresholds");
    const cw2 = 0.92;
    rows.forEach((r, i) => {
      const yy = 46 + i * 118;
      const B = FT.box(g, RX, yy, C.W * cw2, C.H * cw2, null);
      FT.cells(B, 0, 0, cw2, C.W, C.H, (x, y) => r.E[y][x] ? r.col : "rgb(16,18,25)");
      const put = FT.kv(g, RX + C.W * cw2 + 12, yy + 12, { keyW: 96, lead: 14.5, size: 10.5 });
      put(r.n, "", r.col, true);
      put("edge px", r.total, VC.ink);
      put("precision", VZ.fmt(r.prec, 3), r.prec > 0.9 ? VC.good : VC.bad);
      put("arc recall", VZ.fmt(r.rec, 3), r.rec > 0.85 ? VC.good : VC.bad);
      put("components", r.comp, r.comp > 10 ? VC.bad : VC.good);
    });
  }

  function say() {
    const sg = +eS.value, nz = +eN.value;
    const S = plane(sg, nz).S;
    const E = C.hysteresis(S.thin, TL * S.mx, TH * S.mx);
    const cnt = C.counts(E), sc = FTgt.score(E);
    const shi = FTgt.score(C.hysteresis(S.thin, TH * S.mx, TH * S.mx));
    const slo = FTgt.score(C.hysteresis(S.thin, TL * S.mx, TL * S.mx));
    out.innerHTML =
      `At (${TL.toFixed(2)}, ${TH.toFixed(2)}), ratio <b>${VZ.fmt(TH / TL, 2)} : 1</b> — `
      + `<b>${cnt.strong}</b> strong pixels recruit <b>${cnt.weak}</b> weak ones, giving precision `
      + `<b>${VZ.fmt(sc.prec, 3)}</b>, arc recall <b>${VZ.fmt(sc.rec, 3)}</b>, F <b>${VZ.fmt(sc.f1, 3)}</b>, `
      + `in <b>${sc.comp}</b> components. The two single-threshold settings on the same ridge score `
      + `F <b>${VZ.fmt(shi.f1, 3)}</b> (τ_hi alone) and F <b>${VZ.fmt(slo.f1, 3)}</b> (τ_lo alone).<br>`
      + `The dashed diagonal is where the two thresholds coincide, so every point on it is an ordinary `
      + `single-threshold detector. Read the map as a claim about the <i>interior</i>: if the best cell sat on `
      + `the diagonal, hysteresis would be buying nothing on this image. It does not — the interior wins, and `
      + `the size of the win is the vertical distance from the diagonal to the marked optimum. Push the noise `
      + `to zero and watch that distance shrink, because with no noise floor a single low threshold is already `
      + `safe and there is nothing left for a second threshold to protect against.`;
  }
  function all() {
    eSv.textContent = (+eS.value).toFixed(1); eNv.textContent = (+eN.value).toFixed(2);
    draw(); say();
  }
  eS.oninput = all; eN.oninput = all; eM.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   11 · #log-svg — zero crossings against gradient maxima
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#log-svg");
  if (svg.empty()) return;
  const C = FTcanny, S = FTscene;
  const eS = document.getElementById("lg-s"), eSv = document.getElementById("lg-sv");
  const eN = document.getElementById("lg-n"), eNv = document.getElementById("lg-nv");
  const eG = document.getElementById("lg-g"), eGv = document.getElementById("lg-gv");
  const eO = document.getElementById("lg-o");
  const out = document.getElementById("log-readout");
  /* scene B's disc, drawn only as a reference circle — its rasterised boundary is
     NOT exactly at radius 10, which is why the measurement panel builds its own. */
  const DISC = { cx: 56, cy: 18, R: 10 };

  /* the LoG as the sum of two separable filters, as part 2 §13 describes */
  function logFilter(I, sigma) {
    const R = Math.max(1, Math.ceil(3 * sigma));
    const g = VZ.gauss1(sigma), d2 = new Float64Array(2 * R + 1);
    for (let k = -R; k <= R; k++)
      d2[k + R] = ((k * k) / Math.pow(sigma, 4) - 1 / (sigma * sigma))
        * Math.exp(-(k * k) / (2 * sigma * sigma));
    /* normalise the 1-D second derivative so that it sums to zero exactly */
    let s = 0; for (const v of d2) s += v;
    for (let k = 0; k < d2.length; k++) d2[k] -= s / d2.length;
    const a = VZ.sepV(VZ.sepH(I, d2, "mirror"), g, "mirror");
    const b = VZ.sepH(VZ.sepV(I, d2, "mirror"), g, "mirror");
    const O = VZ.zeros2(I.length, I[0].length);
    for (let y = 0; y < I.length; y++) for (let x = 0; x < I[0].length; x++) O[y][x] = a[y][x] + b[y][x];
    return O;
  }
  /* zero-crossing edgels, with the sub-pixel x-intercept of §13 */
  function crossings(L, mag, gate, mx) {
    const out = [], H = L.length, W = L[0].length;
    for (let y = 0; y < H - 1; y++) for (let x = 0; x < W - 1; x++) {
      const a = L[y][x];
      [[1, 0], [0, 1]].forEach(([dx, dy]) => {
        const b = L[y + dy][x + dx];
        if ((a > 0) === (b > 0) || a === b) return;
        const t = Math.abs(a) / (Math.abs(a) + Math.abs(b));
        const gx2 = x + t * dx, gy2 = y + t * dy;
        if (FT.bilerp(mag, gy2, gx2) < gate * mx) return;
        out.push([gx2, gy2]);
      });
    }
    return out;
  }
  /* the measured radius of the zero-crossing loop around the known disc */
  function discRadius(pts) {
    const near = pts.filter(p => Math.abs(Math.hypot(p[0] - DISC.cx, p[1] - DISC.cy) - DISC.R) < 5);
    if (!near.length) return NaN;
    return near.reduce((s, p) => s + Math.hypot(p[0] - DISC.cx, p[1] - DISC.cy), 0) / near.length;
  }

  function get() {
    const sg = +eS.value, nz = +eN.value, gate = +eG.value;
    const st = C.stages(sg, nz, 0.12, 0.40, true);
    const L = logFilter(FT.addNoise(S.B, nz, 11), sg);
    const zc = crossings(L, st.mag, gate, st.mx);
    return { sg, nz, gate, st, L, zc, r: discRadius(zc) };
  }

  function draw() {
    const g0 = get(), st = g0.st, mode = eO.value;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 3.0, IW = C.W * cw;

    const B1 = FT.box(g, 14, 40, IW, IW, "both edge maps on one image");
    FT.cells(B1, 0, 0, cw, C.W, C.H, (x, y) => FT.grey(0.25 + 0.5 * st.sm[y][x]));
    if (mode !== "log") for (let y = 0; y < C.H; y++) for (let x = 0; x < C.W; x++) if (st.E[y][x])
      B1.append("rect").attr("x", x * cw).attr("y", y * cw).attr("width", cw).attr("height", cw)
        .attr("fill", VC.a2).attr("fill-opacity", 0.9).attr("shape-rendering", "crispEdges");
    if (mode !== "canny") g0.zc.forEach(p =>
      B1.append("circle").attr("cx", (p[0] + 0.5) * cw).attr("cy", (p[1] + 0.5) * cw)
        .attr("r", 1.1).attr("fill", VC.accent));
    B1.append("circle").attr("cx", (DISC.cx + 0.5) * cw).attr("cy", (DISC.cy + 0.5) * cw)
      .attr("r", DISC.R * cw).attr("fill", "none").attr("stroke", VC.good)
      .attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
    VZ.legend(B1, [{ color: VC.a2, label: "Canny" }, { color: VC.accent, label: "LoG zero crossings" },
      { color: VC.good, label: "scene disc, nominal R = 10" }], 5, IW - 40, { gap: 13 });

    /* ---- middle: the signed LoG response ---------------------------------- */
    const MX = 14 + IW + 26;
    const B2 = FT.box(g, MX, 40, IW, IW, "∇²G ∗ I, signed");
    let mx = 1e-12;
    for (const r of g0.L) for (const v of r) mx = Math.max(mx, Math.abs(v));
    FT.cells(B2, 0, 0, cw, C.W, C.H, (x, y) => FT.signed(g0.L[y][x] / mx));
    g0.zc.forEach(p => B2.append("circle").attr("cx", (p[0] + 0.5) * cw).attr("cy", (p[1] + 0.5) * cw)
      .attr("r", 0.9).attr("fill", VC.ink));
    const putm = FT.kv(g, MX, 40 + IW + 24, { keyW: 158, lead: 15.5 });
    putm("zero crossings kept", g0.zc.length, VC.accent, true);
    putm("Canny edge pixels", C.counts(st.E).total, VC.a2, true);

    /* ---- right: a purpose-built disc of exactly known radius --------------
       Scene B's disc is a rasterised threshold, so its true edge is not exactly
       at an integer radius. This panel builds its own, anti-aliased by 8 × 8
       supersampling, so that every offset printed is a real measurement against
       a known truth rather than against a rounding. */
    const R0 = +document.getElementById("lg-r").value;
    const N2 = 81, c2 = 40;
    const DI = (function () {
      const I = VZ.zeros2(N2, N2);
      for (let y = 0; y < N2; y++) for (let x = 0; x < N2; x++) {
        let s = 0;
        for (let a = 0; a < 8; a++) for (let b = 0; b < 8; b++)
          s += (Math.hypot(x + (b + 0.5) / 8 - 0.5 - c2, y + (a + 0.5) / 8 - 0.5 - c2) <= R0) ? 1 : 0;
        I[y][x] = 0.18 + 0.67 * s / 64;
      }
      return I;
    })();
    const sm2 = VZ.sep2(DI, VZ.gauss1(g0.sg, 5), "clamp");
    const G2 = FT.grads(sm2, 1.0, "clamp");
    const L2 = logFilter(DI, g0.sg);
    const KMAX = Math.min(N2 - c2 - 2, R0 + 12);
    const ks = [], iv = [], mv = [], lv = [];
    for (let k = 0; k <= KMAX; k++) {
      ks.push(k); iv.push(sm2[c2][c2 + k]);
      mv.push(Math.hypot(G2.gx[c2][c2 + k], G2.gy[c2][c2 + k]));
      lv.push(L2[c2][c2 + k]);
    }
    /* the three measured positions, all sub-pixel */
    const mid = (0.18 + 0.85) / 2;
    let lvl = NaN;
    for (let k = 1; k < KMAX; k++) if ((iv[k] > mid) !== (iv[k + 1] > mid)) { lvl = k + (iv[k] - mid) / (iv[k] - iv[k + 1]); break; }
    let bi = 1;
    for (let k = 1; k < KMAX; k++) if (mv[k] > mv[bi]) bi = k;
    const den = mv[bi - 1] - 2 * mv[bi] + mv[bi + 1];
    const gmax = bi + (Math.abs(den) < 1e-15 ? 0 : 0.5 * (mv[bi - 1] - mv[bi + 1]) / den);
    let zero = NaN;
    for (let k = 1; k < KMAX; k++) if ((lv[k] > 0) !== (lv[k + 1] > 0)) {
      const t = Math.abs(lv[k]) / (Math.abs(lv[k]) + Math.abs(lv[k + 1])), cc = k + t;
      if (isNaN(zero) || Math.abs(cc - R0) < Math.abs(zero - R0)) zero = cc;
    }
    const pred = g0.sg * g0.sg / (2 * R0), shrink = Math.sqrt(Math.max(0, R0 * R0 - g0.sg * g0.sg)) - R0;

    const RX = MX + IW + 30, PW = 214, PH = 96;
    g.append("text").attr("x", RX).attr("y", 32).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("a clean disc, R = " + R0 + " exactly");
    function chart(y0, title, vals, col, marks) {
      const gc = g.append("g").attr("transform", `translate(${RX},${y0})`);
      const x = d3.scaleLinear().domain([0, KMAX]).range([0, PW]);
      const lo = d3.min(vals), hi = d3.max(vals);
      const y = d3.scaleLinear().domain([lo - 0.1 * (hi - lo) - 1e-9, hi + 0.1 * (hi - lo) + 1e-9]).range([PH, 0]);
      VZ.axisB(gc, x, PH, 5, "radius (px)");
      gc.append("text").attr("x", 0).attr("y", -7).attr("font-size", 10.5).attr("fill", col).text(title);
      if (lo < 0 && hi > 0) gc.append("line").attr("x1", 0).attr("x2", PW).attr("y1", y(0)).attr("y2", y(0))
        .attr("stroke", VC.grid);
      gc.append("path").attr("d", d3.line().x((d, i) => x(ks[i])).y(d => y(d))(vals))
        .attr("fill", "none").attr("stroke", col).attr("stroke-width", 2);
      (marks || []).forEach(m => {
        if (!isFinite(m[0])) return;
        gc.append("line").attr("x1", x(m[0])).attr("x2", x(m[0])).attr("y1", 0).attr("y2", PH)
          .attr("stroke", m[1]).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
        gc.append("text").attr("x", x(m[0]) + 3).attr("y", 9).attr("font-size", 9).attr("fill", m[1]).text(m[2]);
      });
    }
    chart(52, "smoothed intensity", iv, VC.muted, [[R0, VC.good, "R"], [lvl, VC.violet, "level set"]]);
    chart(52 + PH + 42, "‖∇I‖", mv, VC.a2, [[R0, VC.good, "R"], [gmax, VC.a2, "max"]]);
    chart(52 + 2 * (PH + 42), "∇²G ∗ I", lv, VC.accent, [[R0, VC.good, "R"], [zero, VC.accent, "zero"]]);
    const putr = FT.kv(g, RX, 52 + 3 * (PH + 42) - 6, { keyW: 168, lead: 15 });
    putr("smoothing moved the edge", VZ.fmt(lvl - R0, 4) + "  (pred " + VZ.fmt(shrink, 4) + ")", VC.violet, true);
    putr("gradient max − level set", VZ.fmt(gmax - lvl, 4), Math.abs(gmax - lvl) < 0.08 ? VC.good : VC.a2, true);
    putr("LoG zero − R", VZ.fmt(zero - R0, 4) + "  (pred " + VZ.fmt(pred, 4) + ")", VC.accent, true);
    putr("LoG zero − gradient max", VZ.fmt(zero - gmax, 4) + "  (pred " + VZ.fmt(2 * pred, 4) + ")", VC.ink, true);
    g0.gm = gmax; g0.zr = zero; g0.lvl = lvl; g0.R0 = R0; g0.pred = pred; g0.shrink = shrink;
    svg.node().__st = g0;
  }

  function say() {
    const g0 = svg.node().__st;
    if (!g0) return;
    out.innerHTML =
      `At <span class="keep">σ</span> = ${g0.sg.toFixed(1)}: <b>${g0.zc.length}</b> zero-crossing edgels survive the `
      + `gradient gate of ${g0.gate.toFixed(2)}, against <b>${FTcanny.counts(g0.st.E).total}</b> Canny edge pixels `
      + `on the scene at the left.<br>`
      + `On the clean test disc of radius <b>${g0.R0}</b>: smoothing alone moved the mid-intensity level set to `
      + `<b>${VZ.fmt(g0.lvl, 4)}</b>, a shift of <b>${VZ.fmt(g0.lvl - g0.R0, 4)}</b> against the curve-shortening `
      + `prediction <code>√(R² − σ²) − R = ${VZ.fmt(g0.shrink, 4)}</code>. The gradient maximum sits at `
      + `<b>${VZ.fmt(g0.gm, 4)}</b>, which is <b>${VZ.fmt(g0.gm - g0.lvl, 4)} px</b> from that moved edge — `
      + `non-maximum suppression finds the smoothed edge essentially exactly. The Laplacian zero crossing is at `
      + `<b>${VZ.fmt(g0.zr, 4)}</b>, <b>${VZ.fmt(g0.zr - g0.R0, 4)}</b> outside the original radius against the `
      + `prediction <code>σ²/(2R) = ${VZ.fmt(g0.pred, 4)}</code>.<br>`
      + `So both detectors are displaced on a curve, in opposite directions, and the gap between them is about `
      + `<code>σ²/R</code>. Canny inherits the smoothing's own curvature shift; the Laplacian adds a second one of `
      + `its own. Drop the gradient gate to zero and the scene fills with contours: every extremum of the smoothed `
      + `image, noise included, is ringed by a closed zero-crossing loop. Closure is a property of the operator, `
      + `not evidence about the scene.`;
  }

  function all() {
    eSv.textContent = (+eS.value).toFixed(1); eNv.textContent = (+eN.value).toFixed(2);
    eGv.textContent = (+eG.value).toFixed(2);
    document.getElementById("lg-rv").textContent = document.getElementById("lg-r").value;
    draw(); say();
  }
  eS.oninput = all; eN.oninput = all; eG.oninput = all; eO.onchange = all;
  document.getElementById("lg-r").oninput = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   12 · #hough-svg — the accumulator, both variants, and random sampling
   ══════════════════════════════════════════════════════════════════════════ */
const FThough = (function () {
  const W = 96, H = 96;
  /* four true lines, in normalised coordinates: (theta, d) with d = x·cosθ + y·sinθ */
  const LINES = [
    { th: VZ.rad(0), d: -0.42 },                  // a vertical line
    { th: VZ.rad(90), d: 0.55 },                  // a horizontal line
    { th: VZ.rad(35), d: 0.10 },
    { th: VZ.rad(-52), d: -0.20 }
  ];
  const nx = p => [2 * p[0] / (W - 1) - 1, 2 * p[1] / (H - 1) - 1];   // pixel → [−1, 1]²
  const px = q => [(q[0] + 1) * (W - 1) / 2, (q[1] + 1) * (H - 1) / 2];

  /* the scene: bright bars along the four lines, with an occluding gap cut through
     the third one, plus a little clutter so the accumulator is not artificially clean */
  function scene(gap, noise) {
    const I = VZ.zeros2(H, W);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const q = nx([x, y]);
      let v = 0.18;
      LINES.forEach((L, i) => {
        const dd = q[0] * Math.cos(L.th) + q[1] * Math.sin(L.th) - L.d;
        if (Math.abs(dd) < 0.022) {
          if (i === 2) {                          // the occluded one
            const t = -q[0] * Math.sin(L.th) + q[1] * Math.cos(L.th);
            if (Math.abs(t) < gap / (W - 1)) return;
          }
          v = 0.84;
        }
      });
      I[y][x] = v;
    }
    /* a few short clutter segments, deterministic */
    const r = VZ.rng(77);
    for (let k = 0; k < 6; k++) {
      const x0 = 6 + Math.floor(r() * (W - 20)), y0 = 6 + Math.floor(r() * (H - 20));
      const a = r() * Math.PI, len = 5 + Math.floor(r() * 7);
      for (let t = 0; t < len; t++) {
        const xx = Math.round(x0 + t * Math.cos(a)), yy = Math.round(y0 + t * Math.sin(a));
        if (xx >= 0 && yy >= 0 && xx < W && yy < H) I[yy][xx] = 0.66;
      }
    }
    const S0 = VZ.sep2(I, VZ.gauss1(0.8), "mirror");
    return noise ? FT.addNoise(S0, noise, 91) : S0;
  }
  /* edgels with orientation, from the Canny machinery of §10–§12 */
  function edgels(gap, noise) {
    const I = scene(gap, noise);
    const G = FT.grads(VZ.sep2(I, VZ.gauss1(1.0), "mirror"), 1.0, "mirror");
    const mag = VZ.zeros2(H, W);
    let mx = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      mag[y][x] = Math.hypot(G.gx[y][x], G.gy[y][x]); mx = Math.max(mx, mag[y][x]);
    }
    const out = [];
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const m = mag[y][x];
      if (m < 0.30 * mx) continue;
      const ux = G.gx[y][x] / m, uy = G.gy[y][x] / m;
      if (m < FT.bilerp(mag, y + uy, x + ux) || m < FT.bilerp(mag, y - uy, x - ux)) continue;
      out.push({ x: x, y: y, th: Math.atan2(uy, ux), m: m / mx, q: nx([x, y]) });
    }
    return { I: I, pts: out };
  }
  const DMAX = Math.SQRT2;
  /* the accumulator. mode "orient" = one vote each, "full" = a sinusoid each. */
  function accumulate(pts, nt, nd, mode) {
    const A = VZ.zeros2(nd, nt);
    let votes = 0;
    for (const p of pts) {
      if (mode === "orient") {
        const th = p.th, d = p.q[0] * Math.cos(th) + p.q[1] * Math.sin(th);
        const i = Math.floor(((th + Math.PI) / (2 * Math.PI)) * nt) % nt;
        const j = Math.floor(((d + DMAX) / (2 * DMAX)) * nd);
        if (j >= 0 && j < nd) { A[j][i] += p.m; votes++; }
      } else {
        for (let i = 0; i < nt; i++) {
          const th = -Math.PI + (i + 0.5) * 2 * Math.PI / nt;
          const d = p.q[0] * Math.cos(th) + p.q[1] * Math.sin(th);
          const j = Math.floor(((d + DMAX) / (2 * DMAX)) * nd);
          if (j >= 0 && j < nd) { A[j][i] += p.m; votes++; }
        }
      }
    }
    return { A: A, votes: votes, nt: nt, nd: nd };
  }
  /* the k strongest well-separated peaks, refined by re-fitting to their own edgels */
  function peaks(acc, k, pts) {
    const { A, nt, nd } = acc, cand = [];
    for (let j = 0; j < nd; j++) for (let i = 0; i < nt; i++) {
      const v = A[j][i];
      if (v <= 0) continue;
      let ok = true;
      for (let b = -1; b <= 1 && ok; b++) for (let a = -1; a <= 1; a++) {
        if (!a && !b) continue;
        const jj = j + b, ii = ((i + a) % nt + nt) % nt;
        if (jj < 0 || jj >= nd) continue;
        if (A[jj][ii] > v) ok = false;
      }
      if (ok) cand.push([i, j, v]);
    }
    cand.sort((p, q) => q[2] - p[2]);
    const keep = [];
    for (const c of cand) {
      if (keep.length >= k) break;
      const th = -Math.PI + (c[0] + 0.5) * 2 * Math.PI / nt;
      const d = -DMAX + (c[1] + 0.5) * 2 * DMAX / nd;
      /* re-fit: total least squares on the edgels that voted for this cell */
      const own = pts.filter(p => {
        const dd = Math.abs(p.q[0] * Math.cos(th) + p.q[1] * Math.sin(th) - d);
        const da = Math.abs(((p.th - th + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI);
        return dd < 2 * DMAX / nd && da < 2 * Math.PI / nt;
      });
      const fit = own.length >= 2 ? tls(own) : { th: th, d: d };
      if (keep.every(q => Math.abs(angDiff(q.th, fit.th)) > 0.25 || Math.abs(q.d - fit.d) > 0.10))
        keep.push({ th: fit.th, d: fit.d, votes: c[2], n: own.length, cell: [c[0], c[1]] });
    }
    return keep;
  }
  const angDiff = (a, b) => ((a - b + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  /* total least squares line through points: the direction is the small eigenvector
     of the scatter matrix, which VZ.jacobiEig supplies */
  function tls(ps) {
    let mx = 0, my = 0;
    for (const p of ps) { mx += p.q[0]; my += p.q[1]; }
    mx /= ps.length; my /= ps.length;
    let sxx = 0, sxy = 0, syy = 0;
    for (const p of ps) {
      const dx = p.q[0] - mx, dy = p.q[1] - my;
      sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
    }
    const e = VZ.jacobiEig([[sxx, sxy], [sxy, syy]]);
    const n = e.vectors[0];                        // the SMALLEST eigenvalue's vector = the normal
    const d = n[0] * mx + n[1] * my;
    return { th: Math.atan2(n[1], n[0]), d: d };
  }
  /* how many of the four true lines a set of hypotheses recovers */
  function recovered(hyp) {
    let n = 0;
    for (const L of LINES) {
      const hit = hyp.some(h => {
        const same = Math.abs(angDiff(h.th, L.th)) < 0.12 && Math.abs(h.d - L.d) < 0.06;
        const flip = Math.abs(angDiff(h.th, L.th + Math.PI)) < 0.12 && Math.abs(h.d + L.d) < 0.06;
        return same || flip;
      });
      if (hit) n++;
    }
    return n;
  }
  /* RANSAC on the same edgels: two points define a line */
  function ransac(pts, iters, tol, seed) {
    const r = VZ.rng(seed || 5);
    let best = null, hist = [];
    for (let it = 0; it < iters; it++) {
      const i = Math.floor(r() * pts.length), j = Math.floor(r() * pts.length);
      if (i === j) { hist.push(best ? best.n : 0); continue; }
      const a = pts[i].q, b = pts[j].q;
      const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy);
      if (L < 1e-6) { hist.push(best ? best.n : 0); continue; }
      const n = [-dy / L, dx / L], d = n[0] * a[0] + n[1] * a[1];
      let cnt = 0;
      for (const p of pts) if (Math.abs(p.q[0] * n[0] + p.q[1] * n[1] - d) < tol) cnt++;
      if (!best || cnt > best.n) best = { th: Math.atan2(n[1], n[0]), d: d, n: cnt, i: i, j: j, at: it };
      hist.push(best.n);
    }
    return { best: best, hist: hist };
  }
  return { W, H, LINES, nx, px, scene, edgels, accumulate, peaks, recovered, ransac, tls, DMAX, angDiff };
})();

(function () {
  const svg = d3.select("#hough-svg");
  if (svg.empty()) return;
  const T = FThough;
  const eV = document.getElementById("ho-v");
  const eT = document.getElementById("ho-nt"), eTv = document.getElementById("ho-ntv");
  const eD = document.getElementById("ho-nd"), eDv = document.getElementById("ho-ndv");
  const eK = document.getElementById("ho-k"), eKv = document.getElementById("ho-kv");
  const eG = document.getElementById("ho-g"), eGv = document.getElementById("ho-gv");
  const out = document.getElementById("hough-readout");
  const EC = new Map();
  const ed = (gap, nz) => {
    const k = gap + "|" + nz;
    if (!EC.has(k)) EC.set(k, T.edgels(gap, nz));
    return EC.get(k);
  };

  function get() {
    const gap = +eG.value, nt = +eT.value, nd = +eD.value, k = +eK.value, mode = eV.value;
    const nz = +document.getElementById("ho-n").value;
    const E = ed(gap, nz);
    const acc = T.accumulate(E.pts, nt, nd, mode);
    const pk = T.peaks(acc, k, E.pts);
    const rs = T.ransac(E.pts, 260, 0.02, 5);
    return { gap, nz, nt, nd, k, mode, E, acc, pk, rs };
  }
  /* clip a line (θ, d) in [−1, 1]² and return its two endpoints in pixels */
  function segment(th, d) {
    const n = [Math.cos(th), Math.sin(th)], t = [-n[1], n[0]];
    const p0 = VZ.scale(n, d), hits = [];
    for (const s of [-1, 1]) {
      if (Math.abs(t[0]) > 1e-9) {
        const u = (s - p0[0]) / t[0], y = p0[1] + u * t[1];
        if (Math.abs(y) <= 1.0001) hits.push([s, y]);
      }
      if (Math.abs(t[1]) > 1e-9) {
        const u = (s - p0[1]) / t[1], x = p0[0] + u * t[0];
        if (Math.abs(x) <= 1.0001) hits.push([x, s]);
      }
    }
    return hits.length >= 2 ? [T.px(hits[0]), T.px(hits[1])] : null;
  }

  function draw() {
    const st = get();
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.3, IW = T.W * cw;

    /* ---- left: the scene, its edgels, and the recovered lines -------------- */
    const B1 = FT.box(g, 14, 40, IW, IW, st.E.pts.length + " edgels, " + st.pk.length + " lines");
    FT.cells(B1, 0, 0, cw, T.W, T.H, (x, y) => FT.grey(0.2 + 0.5 * st.E.I[y][x]));
    st.E.pts.forEach(p => {
      B1.append("line").attr("x1", (p.x + 0.5) * cw).attr("y1", (p.y + 0.5) * cw)
        .attr("x2", (p.x + 0.5 + 1.6 * Math.cos(p.th)) * cw)
        .attr("y2", (p.y + 0.5 + 1.6 * Math.sin(p.th)) * cw)
        .attr("stroke", VC.muted).attr("stroke-width", 0.6).attr("stroke-opacity", 0.7);
    });
    st.pk.forEach((L, i) => {
      const s = segment(L.th, L.d);
      if (!s) return;
      B1.append("line").attr("x1", s[0][0] * cw).attr("y1", s[0][1] * cw)
        .attr("x2", s[1][0] * cw).attr("y2", s[1][1] * cw)
        .attr("stroke", SGPAL(i)).attr("stroke-width", 1.8).attr("stroke-opacity", 0.95);
    });

    /* ---- middle: the accumulator ------------------------------------------ */
    const MX = 14 + IW + 26, AW = 214, AH = 168;
    const B2 = FT.box(g, MX, 40, AW, AH, "accumulator · " + st.acc.votes.toLocaleString() + " votes");
    let mx = 1e-12;
    for (const r of st.acc.A) for (const v of r) mx = Math.max(mx, v);
    /* the accumulator is drawn at one unit per cell inside a <g> that is then scaled
       to the panel, so the cell count can change without touching the layout. Both of
       FT.cells' rendering paths — a rasterised <image> for large grids and individual
       <rect>s for small ones — sit inside that <g> and scale identically. */
    const sub = B2.append("g").attr("transform", `scale(${AW / st.nt},${AH / st.nd})`);
    FT.cells(sub, 0, 0, 1, st.nt, st.nd, (i, j) => FT.heat(st.acc.A[j][i] / mx));
    st.pk.forEach((L, i) => {
      if (!L.cell) return;
      B2.append("circle").attr("cx", (L.cell[0] + 0.5) * AW / st.nt).attr("cy", (L.cell[1] + 0.5) * AH / st.nd)
        .attr("r", 5).attr("fill", "none").attr("stroke", SGPAL(i)).attr("stroke-width", 1.8);
    });
    const sxx = d3.scaleLinear().domain([-180, 180]).range([0, AW]);
    const syy = d3.scaleLinear().domain([-T.DMAX, T.DMAX]).range([0, AH]);
    B2.append("g").attr("class", "axis").attr("transform", `translate(0,${AH})`)
      .call(d3.axisBottom(sxx).ticks(5).tickFormat(d => d + "°"));
    B2.append("g").attr("class", "axis").call(d3.axisLeft(syy).ticks(4));
    B2.append("text").attr("x", -28).attr("y", -8).attr("font-size", 10).attr("fill", VC.muted).text("d");
    B2.append("text").attr("x", AW).attr("y", AH + 30).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", VC.muted).text("θ");

    const put = FT.kv(g, MX, 40 + AH + 56, { keyW: 156, lead: 15.5 });
    put("accumulator cells", (st.nt * st.nd).toLocaleString(), VC.ink, true);
    put("votes cast", st.acc.votes.toLocaleString(), st.mode === "full" ? VC.bad : VC.good, true);
    put("votes per edgel", (st.acc.votes / st.E.pts.length).toFixed(1), VC.ink);
    put("true lines recovered", T.recovered(st.pk) + " of " + T.LINES.length,
      T.recovered(st.pk) === T.LINES.length ? VC.good : VC.bad, true);
    put("cell size (θ, d)", (360 / st.nt).toFixed(1) + "°, " + (2 * T.DMAX / st.nd).toFixed(3), VC.muted);

    /* ---- right: RANSAC on the same edgels ---------------------------------- */
    const RX = MX + AW + 40, cw2 = 2.0, IW2 = T.W * cw2;
    const B3 = FT.box(g, RX, 40, IW2, IW2, "the same edgels, by random sampling");
    FT.cells(B3, 0, 0, cw2, T.W, T.H, () => "rgb(16,18,25)");
    const b = st.rs.best;
    if (b) {
      st.E.pts.forEach(p => {
        const dd = Math.abs(p.q[0] * Math.cos(b.th) + p.q[1] * Math.sin(b.th) - b.d);
        B3.append("circle").attr("cx", (p.x + 0.5) * cw2).attr("cy", (p.y + 0.5) * cw2).attr("r", 1.2)
          .attr("fill", dd < 0.02 ? VC.good : VC.muted).attr("fill-opacity", dd < 0.02 ? 1 : 0.4);
      });
      const s = segment(b.th, b.d);
      if (s) B3.append("line").attr("x1", s[0][0] * cw2).attr("y1", s[0][1] * cw2)
        .attr("x2", s[1][0] * cw2).attr("y2", s[1][1] * cw2)
        .attr("stroke", VC.good).attr("stroke-width", 2);
      [st.E.pts[b.i], st.E.pts[b.j]].forEach(p => B3.append("circle")
        .attr("cx", (p.x + 0.5) * cw2).attr("cy", (p.y + 0.5) * cw2).attr("r", 3.4)
        .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.8));
    }
    /* the best-so-far inlier count against iteration */
    const CY = 40 + IW2 + 34, CW2 = IW2, CH2 = 74;
    const gc = g.append("g").attr("transform", `translate(${RX},${CY})`);
    const x = d3.scaleLinear().domain([0, st.rs.hist.length]).range([0, CW2]);
    const y = d3.scaleLinear().domain([0, d3.max(st.rs.hist) * 1.1 + 1]).range([CH2, 0]);
    VZ.axisB(gc, x, CH2, 4, "hypotheses tried");
    gc.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(3));
    gc.append("text").attr("x", 0).attr("y", -7).attr("font-size", 10).attr("fill", VC.good)
      .text("best inlier count so far");
    gc.append("path").attr("d", d3.line().x((d, i) => x(i)).y(d => y(d))(st.rs.hist))
      .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.8);
  }
  function SGPAL(i) { return ["#5b9cff", "#ffb454", "#4ade80", "#c084fc", "#2dd4bf", "#fb7185",
    "#a3e635", "#38bdf8", "#f0abfc", "#fbbf24"][i % 10]; }

  function say() {
    const st = get();
    const other = T.accumulate(st.E.pts, st.nt, st.nd, st.mode === "orient" ? "full" : "orient");
    let pkmax = 0;
    for (const r of st.acc.A) for (const v of r) pkmax = Math.max(pkmax, v);
    out.innerHTML =
      `<b>${st.E.pts.length}</b> edgels, an accumulator of <b>${st.nt} × ${st.nd} = `
      + `${(st.nt * st.nd).toLocaleString()}</b> cells, <b>${st.acc.votes.toLocaleString()}</b> votes cast `
      + `(${(st.acc.votes / st.E.pts.length).toFixed(1)} per edgel). The other voting scheme would cast `
      + `<b>${other.votes.toLocaleString()}</b> — a factor of `
      + `<b>${VZ.fmt(Math.max(other.votes, st.acc.votes) / Math.min(other.votes, st.acc.votes), 1)}</b>. `
      + `<b>${T.recovered(st.pk)} of ${T.LINES.length}</b> true lines recovered, and the occluded line is `
      + `spanned across a gap of ${st.gap} px that no chain-based method could bridge.<br>`
      + `Random sampling on the same edgels found <b>${st.rs.best ? st.rs.best.n : 0}</b> inliers in 260 `
      + `hypotheses, first reaching that count at hypothesis <b>${st.rs.best ? st.rs.best.at + 1 : "—"}</b>. `
      + `It needs no accumulator and no bin sizes — but it returns <i>one</i> line per run, so finding four `
      + `means four runs with the inliers removed in between, while the Hough array found all four in a single `
      + `pass.<br>`
      + `The bin sizes behave like a histogram's, and the two failure directions are not symmetric here. `
      + `<b>Coarse bins merge</b>: at 8 × 8 the accumulator recovers only 3 of the 4 lines, because two of them `
      + `fall in one cell, and that is a hard loss. <b>Fine bins scatter</b>: the tallest peak falls from about `
      + `87 at 8 × 8 to <b>${VZ.fmt(pkmax, 1)}</b> here, a measurable collapse in peak height — but on a scene `
      + `with only four lines and little clutter the peaks still stand above the background and nothing is lost. `
      + `Turn the noise up and watch the peak height fall further. The over-fine failure is a signal-to-noise `
      + `failure, not a geometric one, and it bites in cluttered images rather than in this one. Either way the `
      + `peaks are used only to <i>group</i> edgels: every line drawn on the left is re-fitted by total least `
      + `squares to the edgels that voted for it, so the bin size never limits the final accuracy.`;
  }
  function all() {
    eTv.textContent = eT.value; eDv.textContent = eD.value;
    eKv.textContent = eK.value; eGv.textContent = eG.value;
    document.getElementById("ho-nv").textContent = (+document.getElementById("ho-n").value).toFixed(2);
    draw(); say();
  }
  eV.onchange = all; eT.oninput = all; eD.oninput = all; eK.oninput = all; eG.oninput = all;
  document.getElementById("ho-n").oninput = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   the scale-space machinery — shared by figures 13, 14, 15 and 16
   ══════════════════════════════════════════════════════════════════════════ */
const FTss = (function () {
  const S = FTscene, W = S.CW, H = S.CH;
  /* the stack, built INCREMENTALLY: the semigroup makes each level cost only the
     difference in variance from the previous one, not a fresh full-width blur. */
  const CACHE = new Map();
  /* `tag` identifies the input image for the cache. It is REQUIRED to be distinct
     for distinct inputs — keying on identity alone silently returned the noiseless
     stack for a noisy image, which is exactly the kind of bug a cache introduces. */
  function stack(I, s, s0, nOct, tag) {
    const key = [tag || (I === S.C ? "C" : "A"), s, s0, nOct].join("|");
    if (CACHE.has(key)) return CACHE.get(key);
    const k = Math.pow(2, 1 / s), levels = [];
    let cur = VZ.sep2(I, VZ.gauss1(s0), "mirror"), sig = s0, cost = s0;
    levels.push({ sigma: sig, L: cur, inc: s0 });
    const n = (nOct || 1) * s + 3;
    for (let i = 1; i < n; i++) {
      const next = sig * k, inc = Math.sqrt(next * next - sig * sig);
      cur = VZ.sep2(cur, VZ.gauss1(inc), "mirror");
      sig = next; cost += inc;
      levels.push({ sigma: sig, L: cur, inc: inc });
    }
    /* differences of adjacent levels, and the scale-normalised Laplacian */
    for (let i = 0; i < levels.length; i++) {
      const lv = levels[i];
      lv.lap = normLaplacian(lv.L, lv.sigma, 1);
      if (i > 0) {
        const D = VZ.zeros2(I.length, I[0].length);
        for (let y = 0; y < I.length; y++) for (let x = 0; x < I[0].length; x++)
          D[y][x] = levels[i].L[y][x] - levels[i - 1].L[y][x];
        lv.dog = D;
      }
    }
    const r = { levels: levels, k: k, s: s, cost: cost };
    CACHE.set(key, r);
    return r;
  }
  /* σ^(2γ)·∇²L, computed with the five-point stencil on an already-smoothed level */
  function normLaplacian(L, sigma, gamma) {
    const H2 = L.length, W2 = L[0].length, O = VZ.zeros2(H2, W2);
    const g = Math.pow(sigma, 2 * (gamma === undefined ? 1 : gamma));
    for (let y = 0; y < H2; y++) for (let x = 0; x < W2; x++) {
      const c = L[y][x];
      const l = L[y][Math.max(0, x - 1)], r = L[y][Math.min(W2 - 1, x + 1)];
      const u = L[Math.max(0, y - 1)][x], d = L[Math.min(H2 - 1, y + 1)][x];
      O[y][x] = g * (l + r + u + d - 4 * c);
    }
    return O;
  }
  /* local extrema of one level, used only to demonstrate non-enhancement */
  function countExtrema(L) {
    let n = 0;
    for (let y = 1; y < L.length - 1; y++) for (let x = 1; x < L[0].length - 1; x++) {
      const v = L[y][x];
      let hi = true, lo = true;
      for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
        if (!a && !b) continue;
        if (L[y + b][x + a] >= v) hi = false;
        if (L[y + b][x + a] <= v) lo = false;
      }
      if (hi || lo) n++;
    }
    return n;
  }
  /* the 26-neighbour scale-space extremum test over a DoG stack */
  function extrema(levels, key) {
    const out = [];
    for (let i = 1; i < levels.length - 1; i++) {
      const D = levels[i][key], Dm = levels[i - 1][key], Dp = levels[i + 1][key];
      if (!D || !Dm || !Dp) continue;
      for (let y = 2; y < D.length - 2; y++) for (let x = 2; x < D[0].length - 2; x++) {
        const v = D[y][x];
        let hi = true, lo = true;
        for (const M of [Dm, D, Dp]) for (let b = -1; b <= 1 && (hi || lo); b++) for (let a = -1; a <= 1; a++) {
          if (M === D && !a && !b) continue;
          if (M[y + b][x + a] >= v) hi = false;
          if (M[y + b][x + a] <= v) lo = false;
        }
        if (hi || lo) out.push({ x: x, y: y, i: i, v: v, sigma: levels[i].sigma, sign: hi ? 1 : -1 });
      }
    }
    return out;
  }
  /* the 3-D quadratic refinement of §19 step 1 */
  function refine3(levels, key, c) {
    const D = levels[c.i][key], Dm = levels[c.i - 1][key], Dp = levels[c.i + 1][key];
    const x = c.x, y = c.y;
    const gx = (D[y][x + 1] - D[y][x - 1]) / 2, gy = (D[y + 1][x] - D[y - 1][x]) / 2;
    const gs = (Dp[y][x] - Dm[y][x]) / 2;
    const hxx = D[y][x + 1] - 2 * D[y][x] + D[y][x - 1];
    const hyy = D[y + 1][x] - 2 * D[y][x] + D[y - 1][x];
    const hss = Dp[y][x] - 2 * D[y][x] + Dm[y][x];
    const hxy = (D[y + 1][x + 1] - D[y + 1][x - 1] - D[y - 1][x + 1] + D[y - 1][x - 1]) / 4;
    const hxs = (Dp[y][x + 1] - Dp[y][x - 1] - Dm[y][x + 1] + Dm[y][x - 1]) / 4;
    const hys = (Dp[y + 1][x] - Dp[y - 1][x] - Dm[y + 1][x] + Dm[y - 1][x]) / 4;
    const Hm = [[hxx, hxy, hxs], [hxy, hyy, hys], [hxs, hys, hss]];
    let d;
    try { d = VZ.mv(VZ.invN(Hm), [-gx, -gy, -gs]); } catch (e) { d = [0, 0, 0]; }
    if (!d.every(v => isFinite(v))) d = [0, 0, 0];
    const val = D[y][x] + 0.5 * (gx * d[0] + gy * d[1] + gs * d[2]);
    return { d: d, val: val, hxx: hxx, hyy: hyy, hxy: hxy,
      tr: hxx + hyy, det: hxx * hyy - hxy * hxy,
      ok: Math.abs(d[0]) < 0.6 && Math.abs(d[1]) < 0.6 && Math.abs(d[2]) < 0.6 };
  }
  return { W, H, stack, normLaplacian, countExtrema, extrema, refine3 };
})();

/* ══════════════════════════════════════════════════════════════════════════
   13 · #scale-svg — the stack, and what survives climbing it
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#scale-svg");
  if (svg.empty()) return;
  const S = FTscene, SS = FTss;
  const eL = document.getElementById("sc-l"), eLv = document.getElementById("sc-lv");
  const eS = document.getElementById("sc-s"), eSv = document.getElementById("sc-sv");
  const eW = document.getElementById("sc-w"), eX = document.getElementById("sc-x");
  const out = document.getElementById("scale-readout");

  function get() {
    const s = +eS.value, st = SS.stack(S.A, s, 0.8, 3);
    const li = Math.min(+eL.value, st.levels.length - 1);
    return { s: s, st: st, li: li, lv: st.levels[li] };
  }
  function draw() {
    const g0 = get();
    eL.max = g0.st.levels.length - 1;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.7, IW = S.W * cw;

    const B1 = FT.box(g, 14, 40, IW, IW, "L(·; σ = " + g0.lv.sigma.toFixed(3) + ")");
    FT.cells(B1, 0, 0, cw, S.W, S.H, (x, y) => FT.grey(g0.lv.L[y][x]));
    /* thumbnails of the earlier levels */
    const tw = 0.72;
    for (let i = 0; i < Math.min(5, g0.li + 1); i++) {
      const lv = g0.st.levels[i];
      const B = FT.box(g, 14 + i * (S.W * tw + 8), 40 + IW + 22, S.W * tw, S.H * tw, null);
      FT.cells(B, 0, 0, tw, S.W, S.H, (x, y) => FT.grey(lv.L[y][x]));
      B.append("text").attr("x", 0).attr("y", S.H * tw + 10).attr("font-size", 8.5)
        .attr("fill", i === g0.li ? VC.a2 : VC.muted).text("σ " + lv.sigma.toFixed(2));
    }

    const MX = 14 + IW + 26;
    const which = eW.value;
    const M = which === "L" ? g0.lv.L : (which === "dog" ? (g0.lv.dog || VZ.zeros2(S.H, S.W)) : g0.lv.lap);
    const title = which === "L" ? "L" : (which === "dog" ? "L(kσ) − L(σ)" : "σ²∇²L");
    const B2 = FT.box(g, MX, 40, IW, IW, title);
    if (which === "L") FT.cells(B2, 0, 0, cw, S.W, S.H, (x, y) => FT.grey(M[y][x]));
    else {
      let mx = 1e-12;
      for (const r of M) for (const v of r) mx = Math.max(mx, Math.abs(v));
      FT.cells(B2, 0, 0, cw, S.W, S.H, (x, y) => FT.signed(M[y][x] / mx));
      B2.append("text").attr("x", 0).attr("y", IW + 12).attr("font-size", 9.5).attr("fill", VC.muted)
        .text("±" + mx.toExponential(2));
    }

    /* ---- right: extrema count and cost, both measured ---------------------- */
    const RX = MX + IW + 32, RW = 196, RH = 132;
    const rows = g0.st.levels.map((lv, i) => ({ i: i, sigma: lv.sigma, inc: lv.inc,
      ext: eX.checked ? SS.countExtrema(lv.L) : 0 }));
    if (eX.checked) {
      const gc = g.append("g").attr("transform", `translate(${RX},52)`);
      const x = d3.scaleLog().domain([rows[0].sigma * 0.9, rows[rows.length - 1].sigma * 1.1]).range([0, RW]);
      const y = d3.scaleLinear().domain([0, d3.max(rows, r => r.ext) * 1.1 + 1]).range([RH, 0]);
      VZ.gridY(gc, y, RW, 4);
      gc.append("g").attr("class", "axis").attr("transform", `translate(0,${RH})`)
        .call(d3.axisBottom(x).ticks(4, "~g"));
      gc.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
      gc.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.accent)
        .text("local extrema of L");
      gc.append("text").attr("x", RW).attr("y", RH + 30).attr("text-anchor", "end")
        .attr("font-size", 10).attr("fill", VC.muted).text("σ (log)");
      gc.append("path").attr("d", d3.line().x(d => x(d.sigma)).y(d => y(d.ext))(rows))
        .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
      rows.forEach(d => gc.append("circle").attr("cx", x(d.sigma)).attr("cy", y(d.ext)).attr("r", 2.4)
        .attr("fill", d.i === g0.li ? VC.a2 : VC.accent));
      let mono = true;
      for (let i = 1; i < rows.length; i++) if (rows[i].ext > rows[i - 1].ext) mono = false;
      gc.append("text").attr("x", 0).attr("y", RH + 44).attr("font-size", 10)
        .attr("fill", mono ? VC.good : VC.a2)
        .text(mono ? "non-increasing throughout" : "NOT monotone — extrema were created");
    }
    /* cost: incremental against direct */
    const gc2 = g.append("g").attr("transform", `translate(${RX},${52 + RH + 84})`);
    const x2 = d3.scaleLinear().domain([0, rows.length - 1]).range([0, RW]);
    const y2 = d3.scaleLinear().domain([0, d3.max(rows, r => r.sigma) * 1.1]).range([110, 0]);
    VZ.axisB(gc2, x2, 110, 5, "level");
    gc2.append("g").attr("class", "axis").call(d3.axisLeft(y2).ticks(4));
    gc2.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.ink)
      .text("kernel width needed at each level");
    gc2.append("path").attr("d", d3.line().x(d => x2(d.i)).y(d => y2(d.sigma))(rows))
      .attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
    gc2.append("path").attr("d", d3.line().x(d => x2(d.i)).y(d => y2(d.inc))(rows))
      .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 2);
    VZ.legend(gc2, [{ color: VC.muted, label: "direct, from the original", dash: "4 3" },
      { color: VC.good, label: "incremental (semigroup)" }], 4, 12, { gap: 13 });
  }

  function say() {
    const g0 = get(), rows = g0.st.levels;
    const e0 = SS.countExtrema(rows[0].L), e1 = SS.countExtrema(rows[rows.length - 1].L);
    let mono = true, prev = Infinity;
    for (const lv of rows) { const e = SS.countExtrema(lv.L); if (e > prev) mono = false; prev = e; }
    const totalInc = rows.reduce((a, r) => a + r.inc, 0);
    const totalDir = rows.reduce((a, r) => a + r.sigma, 0);
    out.innerHTML =
      `Level <b>${g0.li}</b> of ${rows.length - 1}, <span class="keep">σ</span> = <b>${g0.lv.sigma.toFixed(4)}</b>, `
      + `reached from the previous level by an incremental blur of <b>${g0.lv.inc.toFixed(4)}</b> rather than by `
      + `a fresh ${g0.lv.sigma.toFixed(2)}-wide one. Summed over the stack, incremental costs `
      + `<b>${totalInc.toFixed(1)}</b> against <b>${totalDir.toFixed(1)}</b> in kernel width — a factor of `
      + `<b>${VZ.fmt(totalDir / totalInc, 2)}</b>, and that is before the octave-by-octave downsampling of `
      + `<a href="image-processing.html#pyramids">Part 2 §23</a> which cuts it further.<br>`
      + `The image has <b>${e0}</b> local extrema at the bottom of the stack and <b>${e1}</b> at the top. `
      + (mono
        ? `On this stack the count never rises. `
        : `<b>The count is not monotone</b> — it rises somewhere in the middle. `)
      + `That is worth dwelling on, because the usual summary of the scale-space axioms overstates them: in `
      + `<i>one</i> dimension the number of extrema is provably non-increasing, but in <i>two</i> it is not, and `
      + `two ridges merging can manufacture a new saddle and a new maximum between them. Non-enhancement `
      + `constrains an extremum's <b>value</b>, not the <b>count</b>. The trend is still strongly downward and `
      + `the process is still irreversible, which is what makes the stack a legitimate search space.`;
  }
  function all() {
    const g0 = get();
    eLv.textContent = "σ = " + g0.lv.sigma.toFixed(2); eSv.textContent = eS.value;
    draw(); say();
  }
  eL.oninput = all; eS.oninput = all; eW.onchange = all; eX.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   14 · #select-svg — the normalised response across scale
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#select-svg");
  if (svg.empty()) return;
  const S = FTscene, SS = FTss;
  const eG = document.getElementById("se-g"), eB = document.getElementById("se-b");
  const eS = document.getElementById("se-s"), eSv = document.getElementById("se-sv");
  const out = document.getElementById("select-readout");
  const SIGS = VZ.linspace(1, 16, 46);

  const CUR = new Map();
  function curves(gamma) {
    if (CUR.has(gamma)) return CUR.get(gamma);
    const rows = S.DISCS.map(([cx, cy, R]) => ({ cx, cy, R, vals: [] }));
    SIGS.forEach(sg => {
      const L = VZ.sep2(S.C, VZ.gauss1(sg, 4), "clamp");
      const N = SS.normLaplacian(L, sg, gamma);
      rows.forEach(r => r.vals.push(Math.abs(N[r.cy][r.cx])));
    });
    rows.forEach(r => {
      let bi = 0;
      for (let i = 1; i < SIGS.length; i++) if (r.vals[i] > r.vals[bi]) bi = i;
      /* parabolic refinement of the peak in log σ, which is the natural axis */
      let sg = SIGS[bi];
      if (bi > 0 && bi < SIGS.length - 1) {
        const a = r.vals[bi - 1], b = r.vals[bi], c = r.vals[bi + 1], den = a - 2 * b + c;
        if (Math.abs(den) > 1e-15) {
          const t = 0.5 * (a - c) / den;
          sg = Math.exp(Math.log(SIGS[bi]) + t * (Math.log(SIGS[bi + 1]) - Math.log(SIGS[bi - 1])) / 2);
        }
      }
      r.peakSigma = sg; r.peakVal = r.vals[bi]; r.pred = r.R / Math.SQRT2;
    });
    CUR.set(gamma, rows);
    return rows;
  }

  function draw() {
    const gamma = +eG.value, bi = +eB.value, sg = +eS.value;
    const rows = curves(gamma);
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.4, IW = S.CW * cw, IH = S.CH * cw;

    const L = VZ.sep2(S.C, VZ.gauss1(sg, 4), "clamp");
    const N = SS.normLaplacian(L, sg, gamma);
    const B1 = FT.box(g, 14, 40, IW, IH, "the three discs");
    FT.cells(B1, 0, 0, cw, S.CW, S.CH, (x, y) => FT.grey(S.C[y][x]));
    rows.forEach((r, i) => {
      B1.append("circle").attr("cx", (r.cx + 0.5) * cw).attr("cy", (r.cy + 0.5) * cw)
        .attr("r", Math.SQRT2 * r.peakSigma * cw).attr("fill", "none")
        .attr("stroke", PAL(i)).attr("stroke-width", 1.6);
      B1.append("text").attr("x", (r.cx + 0.5) * cw).attr("y", (r.cy + 0.5) * cw - Math.SQRT2 * r.peakSigma * cw - 4)
        .attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", PAL(i))
        .text("R " + r.R + " → " + VZ.fmt(Math.SQRT2 * r.peakSigma, 2));
    });
    const B2 = FT.box(g, 14, 40 + IH + 30, IW, IH, "response at σ = " + sg.toFixed(2));
    let mx = 1e-12;
    for (const r of N) for (const v of r) mx = Math.max(mx, Math.abs(v));
    FT.cells(B2, 0, 0, cw, S.CW, S.CH, (x, y) => FT.signed(N[y][x] / mx));

    /* ---- middle: the three response curves against scale ------------------ */
    const MX = 14 + IW + 34, CW2 = 250, CH2 = 210;
    const gc = g.append("g").attr("transform", `translate(${MX},52)`);
    const x = d3.scaleLog().domain([SIGS[0], SIGS[SIGS.length - 1]]).range([0, CW2]);
    const hi = d3.max(rows, r => d3.max(r.vals));
    const y = d3.scaleLinear().domain([0, hi * 1.12]).range([CH2, 0]);
    VZ.gridY(gc, y, CW2, 5);
    gc.append("g").attr("class", "axis").attr("transform", `translate(0,${CH2})`)
      .call(d3.axisBottom(x).ticks(5, "~g"));
    gc.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    gc.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.ink)
      .text(gamma === 1 ? "|σ²∇²L| at each blob centre" : (gamma === 0 ? "|∇²L|, no normalisation" : "|σ∇²L|, γ = 0.5"));
    gc.append("text").attr("x", CW2).attr("y", CH2 + 30).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", VC.muted).text("σ (log)");
    rows.forEach((r, i) => {
      gc.append("path").attr("d", d3.line().x((d, j) => x(SIGS[j])).y(d => y(d))(r.vals))
        .attr("fill", "none").attr("stroke", PAL(i)).attr("stroke-width", i === bi ? 2.6 : 1.6)
        .attr("stroke-opacity", i === bi ? 1 : 0.6);
      gc.append("circle").attr("cx", x(r.peakSigma)).attr("cy", y(r.peakVal)).attr("r", 4)
        .attr("fill", "none").attr("stroke", PAL(i)).attr("stroke-width", 1.8);
      gc.append("line").attr("x1", x(r.pred)).attr("x2", x(r.pred)).attr("y1", CH2).attr("y2", y(r.peakVal))
        .attr("stroke", PAL(i)).attr("stroke-width", 1).attr("stroke-dasharray", "2 3");
    });
    gc.append("line").attr("x1", x(sg)).attr("x2", x(sg)).attr("y1", 0).attr("y2", CH2)
      .attr("stroke", VC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.6);
    if (gamma === 1) {
      /* the continuous ideal is 2/e for a UNIT-contrast disc; these discs have a
         contrast of 0.68, so the line is drawn at 0.68 × 2/e and labelled as such.
         Drawing it at 0.7358 would be comparing two different quantities. */
      const ideal = 0.68 * 2 / Math.E;
      gc.append("line").attr("x1", 0).attr("x2", CW2).attr("y1", y(ideal)).attr("y2", y(ideal))
        .attr("stroke", VC.good).attr("stroke-width", 1).attr("stroke-dasharray", "5 3");
      gc.append("text").attr("x", 3).attr("y", y(ideal) - 4).attr("font-size", 9.5)
        .attr("fill", VC.good).text("0.68 × 2/e = " + ideal.toFixed(4) + " — the ideal peak, any R");
    }

    /* ---- right: the numbers ----------------------------------------------- */
    const RX = MX + CW2 + 46;
    g.append("text").attr("x", RX).attr("y", 40).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("argmax over scale");
    const put = FT.kv(g, RX, 62, { keyW: 132, lead: 15.5 });
    rows.forEach((r, i) => {
      put("blob R = " + r.R, "", PAL(i), true);
      put("  peak σ", VZ.fmt(r.peakSigma, 4), PAL(i));
      put("  R/√2 (predicted)", VZ.fmt(r.pred, 4), VC.muted);
      put("  √2·σ (reported R)", VZ.fmt(Math.SQRT2 * r.peakSigma, 3), VC.ink, true);
      put("  peak value", VZ.fmt(r.peakVal, 4), VC.ink);
    });
    const hs = rows.map(r => r.peakVal);
    put("peak height spread", VZ.fmt(Math.max(...hs) / Math.min(...hs), 3) + " ×",
      Math.max(...hs) / Math.min(...hs) < 1.3 ? VC.good : VC.bad, true);
  }
  function PAL(i) { return [VC.accent, VC.a2, VC.violet][i % 3]; }

  function say() {
    const gamma = +eG.value, rows = curves(gamma);
    const hs = rows.map(r => r.peakVal);
    const spread = Math.max(...hs) / Math.min(...hs);
    out.innerHTML =
      rows.map(r => `R = ${r.R}: peak at <b>σ = ${VZ.fmt(r.peakSigma, 3)}</b> `
        + `(prediction R/√2 = ${VZ.fmt(r.pred, 3)}), reported radius <b>${VZ.fmt(Math.SQRT2 * r.peakSigma, 2)}</b>`).join("; ")
      + `.<br>`
      + (gamma === 1
        ? `The three peak <i>heights</i> differ by a factor of only <b>${VZ.fmt(spread, 3)}</b>. The continuous `
          + `ideal for a disc of <b>any</b> radius is exactly <code>2/e = 0.7358</code> per unit of contrast, so `
          + `<code>0.68 × 2/e = ${VZ.fmt(0.68 * 2 / Math.E, 4)}</code> for these discs — the measured peaks fall `
          + `a few per cent short of it because the operator here is a five-point stencil on a lattice rather `
          + `than the exact Laplacian. That is what `
          + `γ-normalisation buys: the vertical axis means the same thing at every scale, so "the maximum over `
          + `scale" is a meaningful operation and the argmax is a size <i>measurement</i>.`
        : `With γ = ${gamma} the peak heights differ by a factor of <b>${VZ.fmt(spread, 3)}</b>. `
          + `A maximum taken over an axis whose units change with the axis is not a measurement of anything — `
          + `switch back to γ = 1 and watch the three peaks line up at the same height. Note also that the peak `
          + `<i>locations</i> move: the un-normalised Laplacian peaks at σ = R/2 rather than R/√2, so even the `
          + `size estimate is wrong, by a factor of √2.`);
  }
  function all() {
    eSv.textContent = (+eS.value).toFixed(2);
    draw(); say();
  }
  eG.onchange = all; eB.onchange = all; eS.oninput = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   15 · #dog-svg — the difference of Gaussians against the true LoG
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#dog-svg");
  if (svg.empty()) return;
  const S = FTscene, SS = FTss;
  const eK = document.getElementById("dg-k"), eKv = document.getElementById("dg-kv");
  const eS = document.getElementById("dg-s"), eSv = document.getElementById("dg-sv");
  const eR = document.getElementById("dg-r"), eI = document.getElementById("dg-i");
  const out = document.getElementById("dog-readout");

  const G = (r, s) => Math.exp(-r * r / (2 * s * s)) / (2 * Math.PI * s * s);
  const LoG = (r, s) => ((r * r) / Math.pow(s, 4) - 2 / (s * s)) * G(r, s);
  const DoG = (r, s, k) => G(r, s) - G(r, k * s);
  /* the scale factor that makes the two agree AT THE CENTRE — the convention under
     which every error figure quoted on this page was measured, part 2 included */
  const cOf = (s, k) => -2 * k * k / (s * s * (k * k - 1));
  function err(s, k) {
    const c = cOf(s, k), L0 = Math.abs(LoG(0, s));
    let e = 0, at = 0;
    for (let i = 0; i <= 20000; i++) {
      const r = i * 10 * s / 20000, d = Math.abs(c * DoG(r, s, k) - LoG(r, s));
      if (d > e) { e = d; at = r; }
    }
    return { err: e / L0, at: at, c: c };
  }

  function draw() {
    const k = +eK.value, s = +eS.value, rescale = eR.checked;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const E = err(s, k), c = rescale ? E.c : 1;

    /* ---- left: the two profiles ------------------------------------------- */
    const PW = 250, PH = 190;
    const gc = g.append("g").attr("transform", "translate(48,52)");
    const rs = VZ.linspace(0, 5 * s, 240);
    const lv = rs.map(r => LoG(r, s)), dv = rs.map(r => c * DoG(r, s, k));
    const lo = Math.min(d3.min(lv), d3.min(dv)), hi = Math.max(d3.max(lv), d3.max(dv));
    const x = d3.scaleLinear().domain([0, 5 * s]).range([0, PW]);
    const y = d3.scaleLinear().domain([lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)]).range([PH, 0]);
    VZ.gridY(gc, y, PW, 5);
    VZ.axisB(gc, x, PH, 5, "r (px)");
    gc.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5, ".2e"));
    gc.append("line").attr("x1", 0).attr("x2", PW).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", VC.line);
    /* the shaded difference */
    const area = d3.area().x((d, i) => x(rs[i])).y0((d, i) => y(lv[i])).y1((d, i) => y(dv[i]));
    gc.append("path").attr("d", area(rs)).attr("fill", VC.bad).attr("fill-opacity", 0.22);
    gc.append("path").attr("d", d3.line().x((d, i) => x(rs[i])).y(d => y(d))(lv))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2.2);
    gc.append("path").attr("d", d3.line().x((d, i) => x(rs[i])).y(d => y(d))(dv))
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 2.2).attr("stroke-dasharray", "5 3");
    gc.append("line").attr("x1", x(E.at)).attr("x2", x(E.at)).attr("y1", 0).attr("y2", PH)
      .attr("stroke", VC.bad).attr("stroke-width", 1).attr("stroke-dasharray", "3 3");
    gc.append("text").attr("x", x(E.at) + 4).attr("y", 12).attr("font-size", 9.5).attr("fill", VC.bad)
      .text("worst at r = " + VZ.fmt(E.at, 2) + " = " + VZ.fmt(E.at / s, 2) + "σ");
    VZ.legend(gc, [{ color: VC.accent, label: "σ²∇²G — the true LoG" },
      { color: VC.a2, label: rescale ? "DoG, rescaled to match at r = 0" : "DoG, raw amplitude", dash: "5 3" }],
      6, 12, { gap: 14 });
    g.append("text").attr("x", 48).attr("y", 36).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("radial profiles, σ = " + s.toFixed(2) + ", k = " + k.toFixed(2));

    /* ---- middle: the error curve against k -------------------------------- */
    const MX = 48 + PW + 56, EW = 224, EH = 190;
    const ge = g.append("g").attr("transform", `translate(${MX},52)`);
    const ks = VZ.linspace(1.02, 2.6, 60);
    const es = ks.map(kk => err(s, kk).err);
    const x2 = d3.scaleLinear().domain([1, 2.6]).range([0, EW]);
    const y2 = d3.scaleLinear().domain([0, d3.max(es) * 1.1]).range([EH, 0]);
    VZ.gridY(ge, y2, EW, 5);
    VZ.axisB(ge, x2, EH, 5, "k");
    ge.append("g").attr("class", "axis").call(d3.axisLeft(y2).ticks(5, ".0%"));
    ge.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.bad)
      .text("maximum relative shape error");
    ge.append("path").attr("d", d3.line().x((d, i) => x2(ks[i])).y(d => y2(d))(es))
      .attr("fill", "none").attr("stroke", VC.bad).attr("stroke-width", 2.2);
    [[1.6, "k = 1.6"], [k, "now"]].forEach(([kk, lbl], i) => {
      const e2 = err(s, kk).err;
      ge.append("circle").attr("cx", x2(kk)).attr("cy", y2(e2)).attr("r", 4)
        .attr("fill", i ? VC.a2 : "none").attr("stroke", i ? VC.a2 : VC.ink).attr("stroke-width", 1.6);
      ge.append("text").attr("x", x2(kk) + 6).attr("y", y2(e2) + (i ? 14 : -6)).attr("font-size", 9.5)
        .attr("fill", i ? VC.a2 : VC.ink).text(lbl + " → " + (100 * e2).toFixed(2) + "%");
    });

    /* ---- right: both operators on the image, optionally -------------------- */
    const RX = MX + EW + 46;
    if (eI.checked) {
      const cw = 1.5, IW = S.CW * cw, IH = S.CH * cw;
      const Ls = VZ.sep2(S.C, VZ.gauss1(s, 4), "clamp");
      const LN = SS.normLaplacian(Ls, s, 1);
      const Lk = VZ.sep2(S.C, VZ.gauss1(k * s, 4), "clamp");
      const D = VZ.zeros2(S.CH, S.CW);
      for (let y2b = 0; y2b < S.CH; y2b++) for (let x2b = 0; x2b < S.CW; x2b++)
        D[y2b][x2b] = -(Lk[y2b][x2b] - Ls[y2b][x2b]);
      [["σ²∇²L", LN], ["L(σ) − L(kσ)", D]].forEach((pair, i) => {
        const B = FT.box(g, RX, 52 + i * (IH + 34), IW, IH, pair[0]);
        let mx = 1e-12;
        for (const r of pair[1]) for (const v of r) mx = Math.max(mx, Math.abs(v));
        FT.cells(B, 0, 0, cw, S.CW, S.CH, (xx, yy) => FT.signed(pair[1][yy][xx] / mx));
        S.DISCS.forEach(([cx2, cy2, R]) => B.append("circle").attr("cx", (cx2 + 0.5) * cw)
          .attr("cy", (cy2 + 0.5) * cw).attr("r", R * cw).attr("fill", "none")
          .attr("stroke", VC.good).attr("stroke-width", 1).attr("stroke-dasharray", "2 2"));
      });
    }
    const put = FT.kv(g, RX, eI.checked ? 300 : 60, { keyW: 168, lead: 15.5 });
    put("k", k.toFixed(2), VC.a2, true);
    put("max relative shape error", (100 * E.err).toFixed(2) + "%", E.err > 0.15 ? VC.bad : VC.a2, true);
    put("worst at r", VZ.fmt(E.at, 3) + " = " + VZ.fmt(E.at / s, 3) + "σ", VC.ink);
    put("rescale factor c", VZ.fmt(E.c, 4), VC.muted);
    put("DoG amplitude at r = 0", VZ.fmt(Math.abs(DoG(0, s, k)), 6), VC.muted);
    put("as k → 1, amplitude →", "0", VC.bad, true);
    put("error at k = 1.6", (100 * err(s, 1.6).err).toFixed(2) + "%", VC.a2, true);
  }

  function say() {
    const k = +eK.value, s = +eS.value, E = err(s, k);
    out.innerHTML =
      `At k = <b>${k.toFixed(2)}</b> the difference of Gaussians differs from the true Laplacian of Gaussian by `
      + `at most <b>${(100 * E.err).toFixed(2)}%</b> of the LoG's peak, worst at `
      + `r = <b>${VZ.fmt(E.at / s, 2)}σ</b>, after rescaling the two to agree at the centre. At the customary `
      + `k = 1.6 the figure is <b>${(100 * err(s, 1.6).err).toFixed(2)}%</b>, matching what `
      + `<a href="image-processing.html#derivatives">Part 2 §13</a> measured and quotes as 17.89%. Note that the `
      + `<span class="keep">σ</span> slider does not move this number at all: the shape error is a function of k alone.<br>`
      + `The error curve is <b>monotonically increasing</b> in k, so "a bigger k gives a better DoG" is exactly `
      + `backwards. Push k toward 1 and the two shapes converge — and the DoG's amplitude, printed on the right, `
      + `collapses toward zero, so the response is swamped by numerical noise. The approximation is exact only `
      + `where it is useless, which is why the choice of k is settled by the sub-octave layout of §16 and not by `
      + `any argument about filter shape. And it does not matter much, because the detector reads the response's `
      + `<i>argmax</i>, and an 18% smooth distortion of a unimodal function does not move its peak far.`;
  }
  function all() {
    eKv.textContent = (+eK.value).toFixed(2); eSv.textContent = (+eS.value).toFixed(2);
    draw(); say();
  }
  eK.oninput = all; eS.oninput = all; eR.onchange = all; eI.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   16 · #extrema-svg — the 26-neighbour test and the two rejections
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#extrema-svg");
  if (svg.empty()) return;
  const S = FTscene, SS = FTss;
  const eC = document.getElementById("ex-c"), eCv = document.getElementById("ex-cv");
  const eR = document.getElementById("ex-r"), eRv = document.getElementById("ex-rv");
  const eS = document.getElementById("ex-s"), eSv = document.getElementById("ex-sv");
  const eI = document.getElementById("ex-i");
  const out = document.getElementById("extrema-readout");
  let sel = 0;

  function get() {
    const s = +eS.value, cth = (+eC.value) / 100, r = +eR.value, refine = eI.checked;
    const nz = +document.getElementById("ex-n").value;
    const st = SS.stack(nz ? FT.addNoise(S.A, nz, 17) : S.A, s, 0.9, 2, "A" + nz);
    const cand = SS.extrema(st.levels, "dog");
    let vmax = 1e-12;
    for (const c of cand) vmax = Math.max(vmax, Math.abs(c.v));
    const rows = cand.map(c => {
      const R = SS.refine3(st.levels, "dog", c);
      const val = refine ? Math.abs(R.val) : Math.abs(c.v);
      const edge = R.det <= 0 ? Infinity : R.tr * R.tr / R.det;
      return Object.assign({}, c, { R: R, val: val,
        x2: refine && R.ok ? c.x + R.d[0] : c.x, y2: refine && R.ok ? c.y + R.d[1] : c.y,
        sig2: refine && R.ok ? c.sigma * Math.pow(st.k, R.d[2]) : c.sigma,
        edge: edge, failMove: refine && !R.ok, failCon: val < cth * vmax,
        failEdge: !(R.det > 0 && R.tr * R.tr / R.det < (r + 1) * (r + 1) / r) });
    });
    return { s, nz, cth, r, refine, st, rows, vmax };
  }

  function draw() {
    const g0 = get();
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.5, IW = S.W * cw;
    const kept = g0.rows.filter(d => !d.failMove && !d.failCon && !d.failEdge);
    if (sel >= kept.length) sel = 0;

    const B1 = FT.box(g, 14, 40, IW, IW, kept.length + " of " + g0.rows.length + " candidates survive");
    FT.cells(B1, 0, 0, cw, S.W, S.H, (x, y) => FT.grey(S.A[y][x]));
    g0.rows.forEach(d => {
      const dead = d.failMove || d.failCon || d.failEdge;
      const col = d.failEdge ? VC.bad : (d.failCon ? VC.muted : (d.failMove ? VC.violet : (d.sign > 0 ? VC.a2 : VC.accent)));
      B1.append("circle").attr("cx", (d.x2 + 0.5) * cw).attr("cy", (d.y2 + 0.5) * cw)
        .attr("r", Math.SQRT2 * d.sig2 * cw).attr("fill", "none").attr("stroke", col)
        .attr("stroke-width", dead ? 0.7 : 1.5).attr("stroke-opacity", dead ? 0.45 : 1);
    });
    kept.forEach((d, i) => B1.append("circle").attr("cx", (d.x2 + 0.5) * cw).attr("cy", (d.y2 + 0.5) * cw)
      .attr("r", 2).attr("fill", i === sel ? VC.good : "none").attr("stroke", VC.good)
      .attr("stroke-width", 1).style("cursor", "pointer").on("click", () => { sel = i; draw(); say(); }));
    VZ.legend(B1, [{ color: VC.a2, label: "bright blob" }, { color: VC.accent, label: "dark blob" },
      { color: VC.bad, label: "rejected: edge-like" }, { color: VC.muted, label: "rejected: low contrast" },
      { color: VC.violet, label: "rejected: fit moved" }], 4, IW - 68, { gap: 13 });

    /* ---- middle: the 26-neighbourhood of one candidate --------------------- */
    const MX = 14 + IW + 26, zc = 26;
    const d = kept[sel] || g0.rows[0];
    g.append("text").attr("x", MX).attr("y", 34).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("its 26 neighbours");
    if (d) {
      [-1, 0, 1].forEach((dz, gi) => {
        const lv = g0.st.levels[d.i + dz], D = lv.dog;
        const gx0 = MX, gy0 = 52 + gi * (3 * zc + 26);
        g.append("text").attr("x", gx0).attr("y", gy0 - 5).attr("font-size", 9.5).attr("fill", VC.muted)
          .text((dz === 0 ? "this level" : (dz < 0 ? "below" : "above")) + "  σ = " + lv.sigma.toFixed(2));
        for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
          const v = D[d.y + b][d.x + a], centre = (dz === 0 && !a && !b);
          g.append("rect").attr("x", gx0 + (a + 1) * zc).attr("y", gy0 + (b + 1) * zc)
            .attr("width", zc).attr("height", zc)
            .attr("fill", FT.signed(v / g0.vmax)).attr("stroke", centre ? VC.good : VC.line)
            .attr("stroke-width", centre ? 2 : 0.5);
          g.append("text").attr("x", gx0 + (a + 1.5) * zc).attr("y", gy0 + (b + 1.5) * zc + 3)
            .attr("text-anchor", "middle").attr("font-size", 7.5)
            .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", VC.ink)
            .text(v.toExponential(0));
        }
      });
      const put = FT.kv(g, MX + 3 * zc + 14, 62, { keyW: 66, lead: 14.5, size: 10.5 });
      put("δx", VZ.fmt(d.R.d[0], 3), Math.abs(d.R.d[0]) < 0.6 ? VC.good : VC.bad);
      put("δy", VZ.fmt(d.R.d[1], 3), Math.abs(d.R.d[1]) < 0.6 ? VC.good : VC.bad);
      put("δσ (levels)", VZ.fmt(d.R.d[2], 3), Math.abs(d.R.d[2]) < 0.6 ? VC.good : VC.bad);
      put("fit", d.R.ok ? "accepted" : "move & refit", d.R.ok ? VC.good : VC.bad, true);
      put("|D| refined", d.val.toExponential(2), VC.ink, true);
      put("tr²/det", isFinite(d.edge) ? VZ.fmt(d.edge, 3) : "saddle", d.failEdge ? VC.bad : VC.good, true);
      put("threshold", VZ.fmt((g0.r + 1) * (g0.r + 1) / g0.r, 3), VC.muted);
      put("σ, refined", VZ.fmt(d.sig2, 3), VC.a2, true);
      put("radius √2σ", VZ.fmt(Math.SQRT2 * d.sig2, 2), VC.a2, true);
    }

    /* ---- right: the curvature plane and the counts ------------------------ */
    const RX = MX + 3 * zc + 168, PS = 168;
    const gp = g.append("g").attr("transform", `translate(${RX},52)`);
    gp.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.ink)
      .text("the DoG Hessian's two curvatures");
    const es = g0.rows.map(d => {
      const e = FT.eig2(d.R.hxx, d.R.hxy, d.R.hyy);
      return { a: Math.abs(e[0]), b: Math.abs(e[1]), d: d };
    });
    const mx = d3.max(es, e => Math.max(e.a, e.b)) || 1;
    const x = d3.scaleLinear().domain([0, mx]).range([0, PS]);
    const y = d3.scaleLinear().domain([0, mx]).range([PS, 0]);
    gp.append("rect").attr("width", PS).attr("height", PS).attr("fill", VC.panel).attr("stroke", VC.line);
    /* the accepted wedge: |λ₁/λ₀| ≤ r */
    gp.append("path").attr("d", `M0,${PS} L${x(mx / g0.r)},0 L${PS},0 L${PS},${y(mx / g0.r)} Z`)
      .attr("fill", VC.good).attr("fill-opacity", 0.10);
    gp.append("line").attr("x1", 0).attr("y1", PS).attr("x2", x(mx / g0.r)).attr("y2", 0)
      .attr("stroke", VC.good).attr("stroke-dasharray", "3 3");
    gp.append("line").attr("x1", 0).attr("y1", PS).attr("x2", PS).attr("y2", y(mx / g0.r))
      .attr("stroke", VC.good).attr("stroke-dasharray", "3 3");
    es.forEach(e => gp.append("circle").attr("cx", x(e.a)).attr("cy", y(e.b)).attr("r", 2.4)
      .attr("fill", e.d.failEdge ? VC.bad : VC.good).attr("fill-opacity", 0.85));
    VZ.axisB(gp, x, PS, 3, "|λ₀|");
    gp.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(3));

    const nEdge = g0.rows.filter(d => d.failEdge).length;
    const nCon = g0.rows.filter(d => !d.failEdge && d.failCon).length;
    const nMove = g0.rows.filter(d => !d.failEdge && !d.failCon && d.failMove).length;
    const put2 = FT.kv(g, RX, 52 + PS + 44, { keyW: 168, lead: 15.5 });
    put2("26-neighbour candidates", g0.rows.length, VC.ink, true);
    put2("rejected: edge-like", nEdge, VC.bad, true);
    put2("rejected: low contrast", nCon, VC.muted, true);
    put2("rejected: fit did not settle", nMove, VC.violet, true);
    put2("surviving keypoints", kept.length, VC.good, true);
    put2("equivalent Harris k", VZ.fmt(g0.r / ((g0.r + 1) * (g0.r + 1)), 5), VC.a2, true);
  }

  function say() {
    const g0 = get();
    const kept = g0.rows.filter(d => !d.failMove && !d.failCon && !d.failEdge);
    const nEdge = g0.rows.filter(d => d.failEdge).length;
    const kEq = g0.r / ((g0.r + 1) * (g0.r + 1));
    out.innerHTML =
      `<b>${g0.rows.length}</b> candidates pass the 26-neighbour test; <b>${kept.length}</b> survive all three `
      + `rejections. The edge test alone removes <b>${nEdge}</b> of them at r = ${g0.r}, whose threshold on `
      + `<code>tr²/det</code> is <b>${VZ.fmt((g0.r + 1) * (g0.r + 1) / g0.r, 3)}</b>.<br>`
      + `That same inequality is §06's Harris positivity test with <code>k = r/(r+1)² = `
      + `${VZ.fmt(kEq, 5)}</code> — set r = 10 and you get k = 0.08264, which sits between the usual Harris `
      + `settings of 0.04 and 0.06 and the stricter 0.10. Two detectors, two matrices, one inequality. `
      + `Turn the refinement off and the reported positions snap to integers and the reported scales snap to `
      + `sub-octave levels, which is a quantisation of about ${VZ.fmt(100 * (Math.pow(2, 1 / g0.s) - 1), 0)}% in `
      + `scale — enough to matter when the descriptor's support radius is derived from it.`;
  }
  function all() {
    eCv.textContent = eC.value + "%"; eRv.textContent = eR.value; eSv.textContent = eS.value;
    document.getElementById("ex-nv").textContent = (+document.getElementById("ex-n").value).toFixed(2);
    draw(); say();
  }
  eC.oninput = all; eR.oninput = all; eS.oninput = all; eI.onchange = all;
  document.getElementById("ex-n").oninput = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   17 · #affine-svg — affine adaptation, iterated
   ══════════════════════════════════════════════════════════════════════════ */
const FTaff = (function () {
  const N = 49, C = (N - 1) / 2;
  /* a textured patch defined analytically, so warping is exact resampling of a
     continuous function rather than resampling of a raster */
  function tex(x, y) {
    /* deliberately close to ISOTROPIC: a single dominant edge would give the patch
       an elongated second-moment matrix of its own, and the figure would then be
       measuring the texture's anisotropy rather than the viewpoint's. */
    let v = 0.50;
    const blobs = [[4, -3, 26, 0.15], [-5, 4, 20, -0.14], [0, 6, 14, 0.12],
      [-6, -5, 18, 0.11], [6, 5, 16, -0.12], [1, -6, 12, -0.10]];
    for (const [bx, by, s2, a] of blobs) v += a * Math.exp(-((x - bx) * (x - bx) + (y - by) * (y - by)) / s2);
    v += 0.055 * Math.sin(0.55 * x) * Math.cos(0.52 * y);
    v += 0.045 * Math.sin(0.41 * y) * Math.cos(0.38 * x);
    return VZ.clamp(v, 0.03, 0.97);
  }
  /* sample the patch through a 2 × 2 map M applied in patch coordinates */
  function render(M) {
    const I = VZ.zeros2(N, N);
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const u = c - C, v = r - C;
      I[r][c] = tex(M[0][0] * u + M[0][1] * v, M[1][0] * u + M[1][1] * v);
    }
    return I;
  }
  /* the second-moment matrix of the rendered patch, with a Gaussian window */
  function moment(I, sw) {
    const G = FT.grads(I, 1.2, "clamp");
    let a = 0, b = 0, c = 0;
    for (let r = 0; r < N; r++) for (let cc = 0; cc < N; cc++) {
      const u = cc - C, v = r - C, w = Math.exp(-(u * u + v * v) / (2 * sw * sw));
      const gx = G.gx[r][cc], gy = G.gy[r][cc];
      a += w * gx * gx; b += w * gx * gy; c += w * gy * gy;
    }
    return [[a, b], [b, c]];
  }
  /* M^(−1/2) for a symmetric positive-definite 2 × 2, via VZ.jacobiEig */
  function invSqrt(M) {
    const e = VZ.jacobiEig(M);
    const l0 = Math.max(e.values[0], 1e-12), l1 = Math.max(e.values[1], 1e-12);
    const v0 = e.vectors[0], v1 = e.vectors[1];
    const a = 1 / Math.sqrt(l0), b = 1 / Math.sqrt(l1);
    return [[a * v0[0] * v0[0] + b * v1[0] * v1[0], a * v0[0] * v0[1] + b * v1[0] * v1[1]],
            [a * v0[0] * v0[1] + b * v1[0] * v1[1], a * v0[1] * v0[1] + b * v1[1] * v1[1]]];
  }
  const norm2 = M => {                       /* rescale so the larger singular value is 1 */
    const e = VZ.jacobiEig(VZ.mul(VZ.T(M), M));
    const s = Math.sqrt(Math.max(e.values[1], 1e-12));
    return M.map(r => r.map(v => v / s));
  };
  /* the full iteration: warp, measure, update */
  function adapt(A0, iters, sw) {
    const hist = [];
    let U = VZ.eye(2);
    for (let it = 0; it <= iters; it++) {
      const I = render(VZ.mul(A0, U));
      const M = moment(I, sw), e = VZ.jacobiEig(M);
      const ratio = e.values[0] > 1e-15 ? e.values[1] / e.values[0] : Infinity;
      hist.push({ it: it, I: I, M: M, ratio: ratio, U: U.map(r => r.slice()) });
      if (it < iters) U = norm2(VZ.mul(U, invSqrt(M)));
    }
    return hist;
  }
  /* The residual between two patches, minimised over the SIMILARITY that the rest of
     the pipeline removes anyway: a rotation (§22's orientation assignment) and an
     isotropic scale (§17's scale selection). Affine adaptation's claim is that
     nothing OTHER than a similarity is left, so this is the right thing to measure —
     minimising over rotation alone would confound the claim with the fact that the
     normalisation fixes only the SHAPE of the second-moment matrix and not its size. */
  function residual(Ia, Ib) {
    let best = Infinity, bestTh = 0, bestS = 1;
    for (let d = 0; d < 360; d += 3) {
      const th = VZ.rad(d), ct = Math.cos(th), st = Math.sin(th);
      for (let k = 0; k <= 16; k++) {
        const sc = 0.62 * Math.pow(1.65 / 0.62, k / 16);
        let s = 0, n = 0;
        for (let r = 0; r < N; r += 2) for (let c = 0; c < N; c += 2) {
          const u = c - C, v = r - C;
          if (u * u + v * v > (C - 5) * (C - 5)) continue;
          const q = FT.bilerpC(Ib, C + sc * (st * u + ct * v), C + sc * (ct * u - st * v));
          const d2 = Ia[r][c] - q;
          s += d2 * d2; n++;
        }
        const rms = Math.sqrt(s / Math.max(1, n));
        if (rms < best) { best = rms; bestTh = th; bestS = sc; }
      }
    }
    return { rms: best, theta: bestTh, scale: bestS };
  }
  /* a slant of `deg` about an axis rotated by `phi`, as a 2 × 2 */
  function slant(deg, phi) {
    const c = Math.cos(VZ.rad(deg));
    const R = [[Math.cos(phi), -Math.sin(phi)], [Math.sin(phi), Math.cos(phi)]];
    const D = [[1, 0], [0, 1 / Math.max(c, 0.2)]];
    return VZ.mul(VZ.mul(R, D), VZ.T(R));
  }
  return { N, C, tex, render, moment, invSqrt, norm2, adapt, residual, slant };
})();

(function () {
  const svg = d3.select("#affine-svg");
  if (svg.empty()) return;
  const AF = FTaff;
  const eA = document.getElementById("af-a"), eAv = document.getElementById("af-av");
  const eB = document.getElementById("af-b"), eBv = document.getElementById("af-bv");
  const eR = document.getElementById("af-r"), eRv = document.getElementById("af-rv");
  const eI = document.getElementById("af-i"), eIv = document.getElementById("af-iv");
  const out = document.getElementById("affine-readout");

  function get() {
    const a = +eA.value, b = +eB.value, rot = VZ.rad(+eR.value), n = +eI.value;
    const Aa = AF.slant(a, 0.35);
    const Rb = [[Math.cos(rot), -Math.sin(rot)], [Math.sin(rot), Math.cos(rot)]];
    const Ab = VZ.mul(AF.slant(b, 1.15), Rb);
    const ha = AF.adapt(Aa, n, 9), hb = AF.adapt(Ab, n, 9);
    const res = AF.residual(ha[n].I, hb[n].I);
    const res0 = AF.residual(ha[0].I, hb[0].I);
    return { a, b, rot, n, ha, hb, res, res0 };
  }
  function ellipse(g, M, cx, cy, R, col) {
    const e = VZ.jacobiEig(M);
    const l0 = Math.max(e.values[0], 1e-15), l1 = Math.max(e.values[1], 1e-15);
    const s = R / Math.sqrt(l0);
    const pts = [];
    for (let i = 0; i <= 72; i++) {
      const t = i / 72 * 2 * Math.PI;
      const p = VZ.add(VZ.scale(e.vectors[0], Math.cos(t) / Math.sqrt(l0)),
        VZ.scale(e.vectors[1], Math.sin(t) / Math.sqrt(l1)));
      pts.push([cx + s * p[0], cy + s * p[1]]);
    }
    VZ.poly(g, pts, { stroke: col, fill: "none", w: 1.6 });
  }

  function draw() {
    const g0 = get();
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.5, IW = AF.N * cw;

    function pane(x0, y0, I, M, title, col) {
      const B = FT.box(g, x0, y0, IW, IW, title);
      FT.cells(B, 0, 0, cw, AF.N, AF.N, (x, y) => FT.grey(I[y][x]));
      ellipse(B, M, (AF.C + 0.5) * cw, (AF.C + 0.5) * cw, 0.36 * IW, col);
      return B;
    }
    pane(14, 42, g0.ha[0].I, g0.ha[0].M, "view A, as seen", VC.accent);
    pane(14, 42 + IW + 32, g0.hb[0].I, g0.hb[0].M, "view B, as seen", VC.a2);
    const MX = 14 + IW + 26;
    pane(MX, 42, g0.ha[g0.n].I, g0.ha[g0.n].M, "A normalised, " + g0.n + " iters", VC.accent);
    pane(MX, 42 + IW + 32, g0.hb[g0.n].I, g0.hb[g0.n].M, "B normalised, " + g0.n + " iters", VC.a2);

    /* ---- right: convergence and residual ---------------------------------- */
    const RX = MX + IW + 34, CW2 = 214, CH2 = 132;
    const gc = g.append("g").attr("transform", `translate(${RX},52)`);
    const its = g0.ha.map(h => h.it);
    const x = d3.scaleLinear().domain([0, Math.max(1, g0.n)]).range([0, CW2]);
    const hi = Math.max(d3.max(g0.ha, h => isFinite(h.ratio) ? h.ratio : 1),
      d3.max(g0.hb, h => isFinite(h.ratio) ? h.ratio : 1), 2);
    const y = d3.scaleLinear().domain([1, hi * 1.05]).range([CH2, 0]);
    VZ.gridY(gc, y, CW2, 4);
    VZ.axisB(gc, x, CH2, Math.min(6, g0.n + 1), "iteration");
    gc.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
    gc.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.ink)
      .text("eigenvalue ratio λ₁/λ₀ → 1");
    gc.append("line").attr("x1", 0).attr("x2", CW2).attr("y1", y(1)).attr("y2", y(1))
      .attr("stroke", VC.good).attr("stroke-dasharray", "4 3");
    [[g0.ha, VC.accent], [g0.hb, VC.a2]].forEach(([h, col]) => {
      gc.append("path").attr("d", d3.line().x(d => x(d.it)).y(d => y(Math.min(d.ratio, hi * 1.05)))(h))
        .attr("fill", "none").attr("stroke", col).attr("stroke-width", 2);
      h.forEach(d => gc.append("circle").attr("cx", x(d.it)).attr("cy", y(Math.min(d.ratio, hi * 1.05)))
        .attr("r", 2.4).attr("fill", col));
    });
    const put = FT.kv(g, RX, 52 + CH2 + 50, { keyW: 176, lead: 15.5 });
    put("A: ratio before → after", VZ.fmt(g0.ha[0].ratio, 3) + "  →  " + VZ.fmt(g0.ha[g0.n].ratio, 3), VC.accent, true);
    put("B: ratio before → after", VZ.fmt(g0.hb[0].ratio, 3) + "  →  " + VZ.fmt(g0.hb[g0.n].ratio, 3), VC.a2, true);
    put("residual before, best rot.", VZ.fmt(g0.res0.rms, 4), VC.muted, true);
    put("residual after, best rot.", VZ.fmt(g0.res.rms, 4),
      g0.res.rms < g0.res0.rms ? VC.good : VC.bad, true);
    put("improvement", VZ.fmt(g0.res0.rms / Math.max(1e-9, g0.res.rms), 2) + " ×",
      g0.res.rms < g0.res0.rms ? VC.good : VC.bad, true);
    put("best residual rotation", VZ.fmt(VZ.deg(g0.res.theta), 1) + "°", VC.ink);
    const diverged = !isFinite(g0.ha[g0.n].ratio) || !isFinite(g0.hb[g0.n].ratio)
      || g0.ha[g0.n].ratio > 6 || g0.hb[g0.n].ratio > 6;
    if (diverged) g.append("text").attr("x", RX).attr("y", 52 + CH2 + 50 + 7 * 15.5 + 8)
      .attr("font-size", 10.5).attr("fill", VC.bad)
      .text("ratio still far from 1 — discard this point");
  }

  function say() {
    const g0 = get();
    out.innerHTML =
      `View A is slanted ${g0.a}°, view B ${g0.b}° with a further ${eR.value}° of in-plane rotation. `
      + `Their second-moment eigenvalue ratios start at <b>${VZ.fmt(g0.ha[0].ratio, 3)}</b> and `
      + `<b>${VZ.fmt(g0.hb[0].ratio, 3)}</b>; after <b>${g0.n}</b> iterations they are `
      + `<b>${VZ.fmt(g0.ha[g0.n].ratio, 3)}</b> and <b>${VZ.fmt(g0.hb[g0.n].ratio, 3)}</b>. `
      + `The residual between the two normalised patches, minimised over the similarity that the rest of the `
      + `pipeline removes anyway — a rotation and an isotropic scale — falls from `
      + `<b>${VZ.fmt(g0.res0.rms, 4)}</b> to <b>${VZ.fmt(g0.res.rms, 4)}</b>, a factor of `
      + `<b>${VZ.fmt(g0.res0.rms / Math.max(1e-9, g0.res.rms), 2)}</b>, at a rotation of `
      + `<b>${VZ.fmt(VZ.deg(g0.res.theta), 0)}°</b> and a scale of <b>${VZ.fmt(g0.res.scale, 2)}</b>. `
      + `Note that the recovered rotation tracks the in-plane rotation applied to view B, which is the `
      + `check that the residual is measuring what it claims to.<br>`
      + `That is the whole claim of §20 made measurable: after each patch is warped by the inverse square root `
      + `of its <i>own</i> second-moment matrix, what is left between the two views is a pure rotation, which `
      + `§22 already knows how to remove — and the residual is minimised over exactly that similarity, so the `
      + `improvement is a fair measurement rather than a coincidence. Set both slants equal and the improvement `
      + `collapses to about 1: with no affine difference between the views there is nothing for the `
      + `normalisation to remove, and paying 5–10× the detection cost for it is a pure loss. Note also that the `
      + `starting ratio is about 2 even at 0° slant, because the patch's own texture is not perfectly isotropic; `
      + `what the iteration removes is the <i>difference</i> between the two views, not the anisotropy of the `
      + `scene. On real data the iteration does sometimes fail to converge, and a working implementation must `
      + `test for that and discard the point rather than report a frame it does not believe.`;
  }
  function all() {
    eAv.textContent = eA.value + "°"; eBv.textContent = eB.value + "°";
    eRv.textContent = eR.value + "°"; eIv.textContent = eI.value;
    draw(); say();
  }
  eA.oninput = all; eB.oninput = all; eR.oninput = all; eI.oninput = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   the descriptor machinery — shared by figures 18, 19 and 21
   ══════════════════════════════════════════════════════════════════════════ */
const FTdesc = (function () {
  const N = 33, C = (N - 1) / 2, R = 8;          // 33 samples across ±8 patch units

  /* the reference scene, defined analytically so rotation is exact */
  function tex(x, y) {
    let v = 0.50 + 0.30 * Math.tanh(1.1 * (0.8 * x + 0.6 * y + 1.2));
    v += 0.22 * Math.exp(-((x - 3) * (x - 3) + (y + 2.5) * (y + 2.5)) / 8);
    v -= 0.18 * Math.exp(-((x + 3.5) * (x + 3.5) + (y - 3) * (y - 3)) / 6);
    v += 0.08 * Math.sin(0.9 * x) * Math.cos(0.7 * y);
    return v;
  }
  /* sample the SAME scene through a frame rotated by theta, with an intensity map */
  function sample(theta, a, b, gamma) {
    const I = VZ.zeros2(N, N), c = Math.cos(theta), s = Math.sin(theta);
    for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
      const X = -R + 2 * R * k / (N - 1), Y = -R + 2 * R * r / (N - 1);
      let v = (a === undefined ? 1 : a) * tex(c * X - s * Y, s * X + c * Y) + (b || 0);
      if (gamma && gamma !== 1) v = Math.pow(Math.max(v, 1e-6), gamma);
      I[r][k] = v;
    }
    return I;
  }
  /* central differences, matching the Python reference this page was checked against */
  function grads(P) {
    const n = P.length, gx = VZ.zeros2(n, n), gy = VZ.zeros2(n, n);
    for (let r = 0; r < n; r++) for (let k = 1; k < n - 1; k++) gx[r][k] = (P[r][k + 1] - P[r][k - 1]) / 2;
    for (let r = 1; r < n - 1; r++) for (let k = 0; k < n; k++) gy[r][k] = (P[r + 1][k] - P[r - 1][k]) / 2;
    return { gx, gy };
  }
  const SW = 0.5;                                // Gaussian weight, in normalised units
  function weights(n) {
    const W = VZ.zeros2(n, n);
    for (let r = 0; r < n; r++) for (let k = 0; k < n; k++) {
      const X = -1 + 2 * k / (n - 1), Y = -1 + 2 * r / (n - 1);
      W[r][k] = Math.exp(-(X * X + Y * Y) / (2 * SW * SW));
    }
    return W;
  }
  /* the orientation histogram of §22, with optional circular smoothing */
  function orient(P, nb, smooth, refine) {
    const n = P.length, G = grads(P), W = weights(n), h = new Float64Array(nb || 36);
    const B = nb || 36;
    for (let r = 0; r < n; r++) for (let k = 0; k < n; k++) {
      const m = Math.hypot(G.gx[r][k], G.gy[r][k]);
      if (m <= 0) continue;
      const a = Math.atan2(G.gy[r][k], G.gx[r][k]);
      const i = Math.floor(((a + Math.PI) / (2 * Math.PI)) * B) % B;
      h[(i + B) % B] += m * W[r][k];
    }
    let hs = h;
    for (let p = 0; p < (smooth === undefined ? 2 : smooth); p++) {
      const t = new Float64Array(B);
      for (let i = 0; i < B; i++) t[i] = (hs[(i - 1 + B) % B] + hs[i] + hs[(i + 1) % B]) / 3;
      hs = t;
    }
    let ki = 0;
    for (let i = 1; i < B; i++) if (hs[i] > hs[ki]) ki = i;
    let dk = 0;
    if (refine !== false) {
      const y0 = hs[(ki - 1 + B) % B], y1 = hs[ki], y2 = hs[(ki + 1) % B], den = y0 - 2 * y1 + y2;
      dk = Math.abs(den) < 1e-14 ? 0 : VZ.clamp(0.5 * (y0 - y2) / den, -0.5, 0.5);
    }
    const theta = -Math.PI + (ki + 0.5 + dk) * (2 * Math.PI / B);
    /* secondary peaks above 80% of the maximum */
    const peaks = [];
    for (let i = 0; i < B; i++) {
      if (i === ki) continue;
      if (hs[i] >= 0.8 * hs[ki] && hs[i] > hs[(i - 1 + B) % B] && hs[i] > hs[(i + 1) % B])
        peaks.push(-Math.PI + (i + 0.5) * (2 * Math.PI / B));
    }
    return { theta: theta, hist: h, smoothed: hs, peak: ki, dk: dk, peaks: peaks, B: B };
  }
  /* the 4 × 4 × 8 gradient-histogram descriptor of §23 */
  function descriptor(P, theta0, opt) {
    const o = Object.assign({ cells: 4, bins: 8, cap: 0.2, soft: true, norm: true }, opt || {});
    const n = P.length, G = grads(P), W = weights(n);
    const D = new Float64Array(o.cells * o.cells * o.bins);
    const th = theta0 || 0, ct = Math.cos(-th), st = Math.sin(-th);
    for (let r = 0; r < n; r++) for (let k = 0; k < n; k++) {
      const X = -1 + 2 * k / (n - 1), Y = -1 + 2 * r / (n - 1);
      const Xr = ct * X - st * Y, Yr = st * X + ct * Y;
      if (Math.abs(Xr) > 1 || Math.abs(Yr) > 1) continue;
      const m = Math.hypot(G.gx[r][k], G.gy[r][k]) * W[r][k];
      if (m <= 0) continue;
      const A = ((Math.atan2(G.gy[r][k], G.gx[r][k]) - th + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
      const cx = (Xr + 1) / 2 * o.cells - 0.5, cy = (Yr + 1) / 2 * o.cells - 0.5;
      const ab = (A + Math.PI) / (2 * Math.PI) * o.bins - 0.5;
      if (!o.soft) {
        const xx = VZ.clamp(Math.round(cx), 0, o.cells - 1), yy = VZ.clamp(Math.round(cy), 0, o.cells - 1);
        const aa = ((Math.round(ab) % o.bins) + o.bins) % o.bins;
        D[(yy * o.cells + xx) * o.bins + aa] += m;
        continue;
      }
      const x0 = Math.floor(cx), y0 = Math.floor(cy), a0 = Math.floor(ab);
      const fx = cx - x0, fy = cy - y0, fa = ab - a0;
      for (let dy = 0; dy <= 1; dy++) {
        const yy = y0 + dy;
        if (yy < 0 || yy >= o.cells) continue;
        const wy = dy ? fy : 1 - fy;
        for (let dx = 0; dx <= 1; dx++) {
          const xx = x0 + dx;
          if (xx < 0 || xx >= o.cells) continue;
          const wx = dx ? fx : 1 - fx;
          for (let da = 0; da <= 1; da++) {
            const aa = ((a0 + da) % o.bins + o.bins) % o.bins, wa = da ? fa : 1 - fa;
            D[(yy * o.cells + xx) * o.bins + aa] += m * wx * wy * wa;
          }
        }
      }
    }
    if (!o.norm) return D;
    let nn = 0;
    for (const v of D) nn += v * v;
    nn = Math.sqrt(nn);
    if (nn > 0) for (let i = 0; i < D.length; i++) D[i] /= nn;
    if (o.cap > 0 && o.cap < 1) {
      for (let i = 0; i < D.length; i++) D[i] = Math.min(D[i], o.cap);
      nn = 0;
      for (const v of D) nn += v * v;
      nn = Math.sqrt(nn);
      if (nn > 0) for (let i = 0; i < D.length; i++) D[i] /= nn;
    }
    return D;
  }
  const dist = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) * (a[i] - b[i]); return Math.sqrt(s); };
  return { N, C, R, tex, sample, grads, weights, orient, descriptor, dist };
})();

/* ══════════════════════════════════════════════════════════════════════════
   18 · #orient-svg — orientation assignment
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#orient-svg");
  if (svg.empty()) return;
  const D = FTdesc;
  const eR = document.getElementById("or-r"), eRv = document.getElementById("or-rv");
  const eB = document.getElementById("or-b"), eBv = document.getElementById("or-bv");
  const eS = document.getElementById("or-s"), eSv = document.getElementById("or-sv");
  const eO = document.getElementById("or-o"), eP = document.getElementById("or-p");
  const out = document.getElementById("orient-readout");

  function get() {
    const deg = +eR.value, nb = +eB.value, sm = +eS.value, cancel = eO.checked, ref = eP.checked;
    const P0 = D.sample(0), o0 = D.orient(P0, nb, sm, ref);
    const v0 = D.descriptor(P0, cancel ? o0.theta : 0);
    const P = D.sample(VZ.rad(deg)), o = D.orient(P, nb, sm, ref);
    const v = D.descriptor(P, cancel ? o.theta : 0);
    let exp = VZ.deg(o0.theta) - deg; exp = ((exp + 180) % 360 + 360) % 360 - 180;
    let est = ((VZ.deg(o.theta) + 180) % 360 + 360) % 360 - 180;
    const err = ((est - exp + 180) % 360 + 360) % 360 - 180;
    return { deg, nb, sm, cancel, ref, P0, o0, v0, P, o, v, est, exp, err, d: D.dist(v, v0) };
  }
  /* the distance-against-rotation curve, both ways, cached per (bins, smooth, refine) */
  const CV = new Map();
  function curve(nb, sm, ref) {
    const key = nb + "|" + sm + "|" + ref;
    if (CV.has(key)) return CV.get(key);
    const P0 = D.sample(0), o0 = D.orient(P0, nb, sm, ref);
    const vA0 = D.descriptor(P0, o0.theta), vB0 = D.descriptor(P0, 0);
    const rows = [];
    for (let deg = 0; deg <= 360; deg += 6) {
      const P = D.sample(VZ.rad(deg)), o = D.orient(P, nb, sm, ref);
      rows.push({ deg: deg, a: D.dist(D.descriptor(P, o.theta), vA0), b: D.dist(D.descriptor(P, 0), vB0) });
    }
    CV.set(key, rows);
    return rows;
  }

  function draw() {
    const g0 = get();
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 5.2, IW = D.N * cw;

    const B1 = FT.box(g, 14, 44, IW, IW, "the patch, rotated " + g0.deg + "°");
    FT.cells(B1, 0, 0, cw, D.N, D.N, (x, y) => FT.grey(g0.P[y][x]));
    const cx = (D.C + 0.5) * cw, cy = (D.C + 0.5) * cw;
    VZ.arrow(B1, cx, cy, cx + 0.42 * IW * Math.cos(g0.o.theta), cy + 0.42 * IW * Math.sin(g0.o.theta),
      { color: VC.a2, w: 2.4, head: 8 });
    g0.o.peaks.forEach(t => VZ.arrow(B1, cx, cy, cx + 0.34 * IW * Math.cos(t), cy + 0.34 * IW * Math.sin(t),
      { color: VC.violet, w: 1.6, head: 6, op: 0.85 }));
    B1.append("text").attr("x", 0).attr("y", IW + 14).attr("font-size", 10).attr("fill", VC.muted)
      .text("θ₀ = " + VZ.fmt(g0.est, 2) + "°" + (g0.o.peaks.length ? "  + " + g0.o.peaks.length + " secondary" : ""));

    /* ---- middle: the polar histogram -------------------------------------- */
    const MX = 14 + IW + 34, PR = 84;
    const gp = g.append("g").attr("transform", `translate(${MX + PR},${44 + PR})`);
    const hmax = Math.max(...g0.o.smoothed, ...g0.o.hist, 1e-12);
    for (let i = 0; i < g0.nb; i++) {
      const a0 = -Math.PI + i * 2 * Math.PI / g0.nb, a1 = a0 + 2 * Math.PI / g0.nb;
      const rr = PR * g0.o.hist[i] / hmax;
      gp.append("path").attr("d", d3.arc()({ innerRadius: 0, outerRadius: rr,
        startAngle: a0 + Math.PI / 2, endAngle: a1 + Math.PI / 2 }))
        .attr("fill", i === g0.o.peak ? VC.a2 : VC.accent).attr("fill-opacity", 0.35);
    }
    const line = d3.lineRadial().angle(d => d.a + Math.PI / 2).radius(d => d.r).curve(d3.curveCardinalClosed);
    gp.append("path").attr("d", line(d3.range(g0.nb).map(i => ({
      a: -Math.PI + (i + 0.5) * 2 * Math.PI / g0.nb, r: PR * g0.o.smoothed[i] / hmax }))))
      .attr("fill", "none").attr("stroke", VC.ink).attr("stroke-width", 1.8);
    gp.append("circle").attr("r", PR * 0.8 * Math.max(...g0.o.smoothed) / hmax)
      .attr("fill", "none").attr("stroke", VC.violet).attr("stroke-dasharray", "3 3");
    VZ.arrow(gp, 0, 0, PR * Math.cos(g0.o.theta), PR * Math.sin(g0.o.theta), { color: VC.a2, w: 2.2, head: 7 });
    g.append("text").attr("x", MX).attr("y", 36).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("orientation histogram");
    g.append("text").attr("x", MX).attr("y", 44 + 2 * PR + 16).attr("font-size", 10).attr("fill", VC.violet)
      .text("dashed ring = 80% of the peak");
    g.append("text").attr("x", MX).attr("y", 44 + 2 * PR + 30).attr("font-size", 10).attr("fill", VC.muted)
      .text("sub-bin offset δk = " + VZ.fmt(g0.o.dk, 3) + " of " + VZ.fmt(360 / g0.nb, 1) + "°");

    /* ---- middle-lower: the descriptor as a grid of stars ------------------ */
    const DY = 44 + 2 * PR + 46, cs = 34;
    g.append("text").attr("x", MX).attr("y", DY - 6).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("the descriptor, 4 × 4 × 8");
    const dmax = Math.max(...g0.v, 1e-12);
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
      const ox = MX + c * cs + cs / 2, oy = DY + 8 + r * cs + cs / 2;
      g.append("rect").attr("x", ox - cs / 2).attr("y", oy - cs / 2).attr("width", cs).attr("height", cs)
        .attr("fill", "none").attr("stroke", VC.line);
      for (let b = 0; b < 8; b++) {
        const a = -Math.PI + (b + 0.5) * Math.PI / 4, L = (cs / 2 - 2) * g0.v[(r * 4 + c) * 8 + b] / dmax;
        g.append("line").attr("x1", ox).attr("y1", oy).attr("x2", ox + L * Math.cos(a)).attr("y2", oy + L * Math.sin(a))
          .attr("stroke", VC.a2).attr("stroke-width", 1.4);
      }
    }

    /* ---- right: the distance curve over a full turn ----------------------- */
    const RX = MX + 4 * cs + 46, CW2 = 232, CH2 = 176;
    const rows = curve(g0.nb, g0.sm, g0.ref);
    const gc = g.append("g").attr("transform", `translate(${RX},60)`);
    const x = d3.scaleLinear().domain([0, 360]).range([0, CW2]);
    const y = d3.scaleLinear().domain([0, 1.45]).range([CH2, 0]);
    VZ.gridY(gc, y, CW2, 5);
    VZ.axisB(gc, x, CH2, 5, "rotation (°)");
    gc.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    gc.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.ink)
      .text("descriptor distance to the reference");
    gc.append("path").attr("d", d3.line().x(d => x(d.deg)).y(d => y(d.b))(rows))
      .attr("fill", "none").attr("stroke", VC.bad).attr("stroke-width", 2);
    gc.append("path").attr("d", d3.line().x(d => x(d.deg)).y(d => y(d.a))(rows))
      .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 2);
    gc.append("line").attr("x1", x(g0.deg)).attr("x2", x(g0.deg)).attr("y1", 0).attr("y2", CH2)
      .attr("stroke", VC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.6);
    VZ.legend(gc, [{ color: VC.good, label: "orientation cancelled" },
      { color: VC.bad, label: "not cancelled" }], 6, 12, { gap: 14 });
    const mA = d3.mean(rows, r => r.a), mB = d3.mean(rows, r => r.b);
    const put = FT.kv(g, RX, 60 + CH2 + 54, { keyW: 176, lead: 15.5 });
    put("θ₀ estimated", VZ.fmt(g0.est, 2) + "°", VC.a2, true);
    put("θ₀ expected", VZ.fmt(g0.exp, 2) + "°", VC.muted);
    put("tracking error", VZ.fmt(g0.err, 2) + "°", Math.abs(g0.err) < 4 ? VC.good : VC.bad, true);
    put("descriptor distance", VZ.fmt(g0.d, 4), g0.d < 0.2 ? VC.good : VC.bad, true);
    put("mean over a full turn", VZ.fmt(mA, 4) + "  vs  " + VZ.fmt(mB, 4), VC.ink, true);
    put("ratio", VZ.fmt(mB / Math.max(1e-9, mA), 1) + " ×", VC.good, true);
  }

  function say() {
    const g0 = get(), rows = curve(g0.nb, g0.sm, g0.ref);
    const mA = d3.mean(rows, r => r.a), mB = d3.mean(rows, r => r.b);
    out.innerHTML =
      `At ${g0.deg}° the histogram peak is at <b>${VZ.fmt(g0.est, 2)}°</b> against an expected `
      + `<b>${VZ.fmt(g0.exp, 2)}°</b> — a tracking error of <b>${VZ.fmt(g0.err, 2)}°</b> — and the descriptor `
      + `sits <b>${VZ.fmt(g0.d, 4)}</b> from the reference. `
      + (g0.o.peaks.length
        ? `<b>${g0.o.peaks.length}</b> secondary orientation${g0.o.peaks.length > 1 ? "s are" : " is"} above 80% of `
          + `the peak, so a real detector would emit ${g0.o.peaks.length + 1} keypoints here and let §25 choose. `
        : `No secondary peak is above 80% of the maximum, so this patch has one unambiguous orientation. `)
      + `<br>Averaged over a full turn, the descriptor distance is <b>${VZ.fmt(mA, 4)}</b> with orientation `
      + `cancellation and <b>${VZ.fmt(mB, 4)}</b> without — a factor of <b>${VZ.fmt(mB / Math.max(1e-9, mA), 1)}</b>. `
      + `The green curve is flat and near zero except where the estimate is inaccurate; the red one sweeps most `
      + `of the descriptor's available range, which is what "not rotation invariant" means quantitatively. `
      + `Reduce the number of bins and watch the green curve rise: coarser bins quantise θ₀ and every degree of `
      + `that quantisation is a degree of misalignment in the sampling frame.`;
  }
  function all() {
    eRv.textContent = eR.value + "°"; eBv.textContent = eB.value; eSv.textContent = eS.value;
    draw(); say();
  }
  eR.oninput = all; eB.oninput = all; eS.oninput = all; eO.onchange = all; eP.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   19 · #desc-svg — a gradient-histogram descriptor, built cell by cell
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#desc-svg");
  if (svg.empty()) return;
  const D = FTdesc;
  const eS = document.getElementById("de-stage");
  const eC = document.getElementById("de-c"), eCv = document.getElementById("de-cv");
  const eO = document.getElementById("de-o"), eOv = document.getElementById("de-ov");
  const eA = document.getElementById("de-a"), eAv = document.getElementById("de-av");
  const eB = document.getElementById("de-b"), eBv = document.getElementById("de-bv");
  const eG = document.getElementById("de-g"), eGv = document.getElementById("de-gv");
  const eT = document.getElementById("de-t"), eK = document.getElementById("de-k");
  const out = document.getElementById("desc-readout");

  function get() {
    const a = +eA.value, b = +eB.value, gm = +eG.value;
    const cells = +eC.value, bins = +eO.value, soft = eT.checked, cap = eK.checked ? 0.2 : 0;
    const opt = { cells, bins, soft, cap };
    const P0 = D.sample(0), o0 = D.orient(P0, 36, 2, true);
    const v0 = D.descriptor(P0, o0.theta, opt);
    const raw0 = D.descriptor(P0, o0.theta, Object.assign({}, opt, { norm: false }));
    const P = D.sample(0, a, b, gm), o = D.orient(P, 36, 2, true);
    const v = D.descriptor(P, o.theta, opt);
    const raw = D.descriptor(P, o.theta, Object.assign({}, opt, { norm: false }));
    const vNo = D.descriptor(P, o.theta, Object.assign({}, opt, { cap: 0 }));
    const v0No = D.descriptor(P0, o0.theta, Object.assign({}, opt, { cap: 0 }));
    return { a, b, gm, cells, bins, soft, cap, P0, P, o, v, v0, raw, raw0,
      d: D.dist(v, v0), dRaw: D.dist(raw, raw0), dNoCap: D.dist(vNo, v0No) };
  }

  function draw() {
    const g0 = get(), stage = +eS.value;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 5.0, IW = D.N * cw;

    const B1 = FT.box(g, 14, 44, IW, IW, ["the sampled patch", "gradients, weighted",
      "the cell grid", "the patch"][stage]);
    FT.cells(B1, 0, 0, cw, D.N, D.N, (x, y) => FT.grey(g0.P[y][x]));
    if (stage >= 1) {
      const G = D.grads(g0.P), W = D.weights(D.N);
      let mx = 1e-12;
      for (let r = 0; r < D.N; r++) for (let k = 0; k < D.N; k++)
        mx = Math.max(mx, Math.hypot(G.gx[r][k], G.gy[r][k]) * W[r][k]);
      for (let r = 1; r < D.N - 1; r += 2) for (let k = 1; k < D.N - 1; k += 2) {
        const m = Math.hypot(G.gx[r][k], G.gy[r][k]) * W[r][k];
        if (m < 0.02 * mx) continue;
        const a = Math.atan2(G.gy[r][k], G.gx[r][k]), L = 2.2 * cw * m / mx;
        B1.append("line").attr("x1", (k + 0.5) * cw).attr("y1", (r + 0.5) * cw)
          .attr("x2", (k + 0.5) * cw + L * Math.cos(a)).attr("y2", (r + 0.5) * cw + L * Math.sin(a))
          .attr("stroke", VC.a2).attr("stroke-width", 1).attr("stroke-opacity", 0.85);
      }
    }
    if (stage >= 2) {
      const th = -g0.o.theta, ct = Math.cos(th), st = Math.sin(th);
      const c0 = (D.C + 0.5) * cw, half = IW / 2;
      for (let i = 0; i <= g0.cells; i++) {
        const t = -1 + 2 * i / g0.cells;
        [[[t, -1], [t, 1]], [[-1, t], [1, t]]].forEach(seg => {
          const p = seg.map(q => [c0 + half * (ct * q[0] + st * q[1]), c0 + half * (-st * q[0] + ct * q[1])]);
          B1.append("line").attr("x1", p[0][0]).attr("y1", p[0][1]).attr("x2", p[1][0]).attr("y2", p[1][1])
            .attr("stroke", VC.good).attr("stroke-width", 1).attr("stroke-opacity", 0.8);
        });
      }
      VZ.arrow(B1, c0, c0, c0 + 0.4 * IW * Math.cos(g0.o.theta), c0 + 0.4 * IW * Math.sin(g0.o.theta),
        { color: VC.violet, w: 2, head: 7 });
    }

    /* ---- middle: the descriptor as stars, and as a bar chart -------------- */
    const MX = 14 + IW + 34, cs = Math.min(38, 160 / g0.cells);
    g.append("text").attr("x", MX).attr("y", 36).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text(g0.cells + " × " + g0.cells + " × " + g0.bins + " = "
        + (g0.cells * g0.cells * g0.bins) + " numbers");
    const dmax = Math.max(...g0.v, 1e-12);
    for (let r = 0; r < g0.cells; r++) for (let c = 0; c < g0.cells; c++) {
      const ox = MX + c * cs + cs / 2, oy = 52 + r * cs + cs / 2;
      g.append("rect").attr("x", ox - cs / 2).attr("y", oy - cs / 2).attr("width", cs).attr("height", cs)
        .attr("fill", "none").attr("stroke", VC.line);
      for (let b = 0; b < g0.bins; b++) {
        const a = -Math.PI + (b + 0.5) * 2 * Math.PI / g0.bins;
        const L = (cs / 2 - 2) * g0.v[(r * g0.cells + c) * g0.bins + b] / dmax;
        g.append("line").attr("x1", ox).attr("y1", oy).attr("x2", ox + L * Math.cos(a))
          .attr("y2", oy + L * Math.sin(a)).attr("stroke", VC.a2).attr("stroke-width", 1.3);
      }
    }
    const BY = 52 + g0.cells * cs + 40, BW = 250, BH = 96;
    const gb = g.append("g").attr("transform", `translate(${MX},${BY})`);
    const x = d3.scaleLinear().domain([0, g0.v.length]).range([0, BW]);
    const y = d3.scaleLinear().domain([0, Math.max(dmax, g0.cap || 0.25) * 1.1]).range([BH, 0]);
    VZ.axisB(gb, x, BH, 4, "component");
    gb.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
    for (let i = 0; i < g0.v.length; i++)
      gb.append("rect").attr("x", x(i)).attr("y", y(g0.v[i])).attr("width", Math.max(0.8, BW / g0.v.length - 0.3))
        .attr("height", BH - y(g0.v[i]))
        .attr("fill", g0.cap && g0.v[i] >= g0.cap * 0.999 ? VC.bad : VC.accent);
    if (g0.cap) {
      gb.append("line").attr("x1", 0).attr("x2", BW).attr("y1", y(g0.cap)).attr("y2", y(g0.cap))
        .attr("stroke", VC.bad).attr("stroke-dasharray", "4 3");
      gb.append("text").attr("x", BW).attr("y", y(g0.cap) - 4).attr("text-anchor", "end")
        .attr("font-size", 9.5).attr("fill", VC.bad).text("cap 0.2");
    }

    /* ---- right: the distances -------------------------------------------- */
    const RX = MX + BW + 46;
    const nCapped = g0.cap ? g0.v.filter(v => v >= g0.cap * 0.999).length : 0;
    const put = FT.kv(g, RX, 60, { keyW: 184, lead: 15.5 });
    put("gain a", g0.a.toFixed(2), VC.ink, true);
    put("offset b", VZ.fmt(g0.b, 2), VC.ink, true);
    put("gamma", g0.gm.toFixed(2), g0.gm === 1 ? VC.muted : VC.a2, true);
    put("d, normalised descriptor", g0.d.toFixed(6), g0.d < 1e-9 ? VC.good : VC.a2, true);
    put("d, raw histogram", VZ.fmt(g0.dRaw, 4), g0.dRaw > 0.1 ? VC.bad : VC.muted, true);
    put("components at the cap", nCapped + " of " + g0.v.length, nCapped ? VC.bad : VC.muted);
    put("d without the cap", VZ.fmt(g0.dNoCap, 6), VC.ink, true);
    put("the cap is", g0.gm === 1 && g0.d < 1e-9 ? "irrelevant here"
      : (g0.d < g0.dNoCap ? "helping" : "hurting"),
      g0.gm === 1 && g0.d < 1e-9 ? VC.muted : (g0.d < g0.dNoCap ? VC.good : VC.bad), true);
    put("soft binning", g0.soft ? "on" : "off", g0.soft ? VC.good : VC.a2, true);
  }

  function say() {
    const g0 = get();
    out.innerHTML =
      `Gain <b>${g0.a.toFixed(2)}</b>, offset <b>${VZ.fmt(g0.b, 2)}</b>, gamma <b>${g0.gm.toFixed(2)}</b>. `
      + `Normalised descriptor distance <b>${g0.d.toFixed(6)}</b>; raw histogram distance `
      + `<b>${VZ.fmt(g0.dRaw, 4)}</b>.<br>`
      + (g0.gm === 1
        ? `With gamma at 1 the intensity change is <b>affine</b>, and the normalised distance is exactly `
          + `<b>${g0.d.toFixed(6)}</b> — zero to every digit, for any gain and any offset. The offset dies in the `
          + `gradient and the gain dies in the unit normalisation. These are algebraic identities, not tuned `
          + `robustness: sweep the two sliders as far as they go and the number does not move.`
        : `With gamma at ${g0.gm.toFixed(2)} the intensity change is <b>non-linear</b>, and no amount of `
          + `normalisation removes it — the distance is <b>${VZ.fmt(g0.d, 4)}</b> with the cap and `
          + `<b>${VZ.fmt(g0.dNoCap, 4)}</b> without. On this patch the cap is `
          + `<b>${g0.d < g0.dNoCap ? "helping" : "hurting"}</b>, and over 200 random patches it hurts on average `
          + `under every non-linear map tested. It is a database-tuned heuristic, and §23 says so.`)
      + ` Turning soft binning off makes the descriptor jump discontinuously when a sample crosses a cell `
      + `boundary — visible as the stars snapping between neighbouring cells as the gamma slider changes the `
      + `gradient magnitudes and hence which samples dominate.`;
  }
  function all() {
    eCv.textContent = eC.value; eOv.textContent = eO.value;
    eAv.textContent = (+eA.value).toFixed(2); eBv.textContent = (+eB.value).toFixed(2);
    eGv.textContent = (+eG.value).toFixed(2);
    draw(); say();
  }
  eS.onchange = all; eC.oninput = all; eO.oninput = all; eA.oninput = all;
  eB.oninput = all; eG.oninput = all; eT.onchange = all; eK.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   20 · #binary-svg — a binary descriptor and Hamming matching
   ══════════════════════════════════════════════════════════════════════════ */
const FTbin = (function () {
  const D = FTdesc, N = D.N, C = D.C;
  /* the sampling pattern: pairs drawn from an isotropic Gaussian inside the patch,
     in NORMALISED coordinates so it can be rotated exactly */
  const PAIRS = (function () {
    const r = VZ.rng(9091), out = [];
    while (out.length < 256) {
      const p = [VZ.randn(r) * 0.32, VZ.randn(r) * 0.32];
      const q = [p[0] + VZ.randn(r) * 0.20, p[1] + VZ.randn(r) * 0.20];
      if (Math.hypot(...p) > 0.92 || Math.hypot(...q) > 0.92) continue;
      out.push([p, q]);
    }
    return out;
  })();
  function smoothed(P, sigma) {
    return sigma > 0.05 ? VZ.sep2(P, VZ.gauss1(sigma), "clamp") : P;
  }
  /* the bit string: one comparison per pair, with the pattern optionally rotated */
  function bits(P, n, sigma, rot) {
    const S = smoothed(P, sigma), out = new Uint8Array(n);
    const ct = Math.cos(rot || 0), st = Math.sin(rot || 0);
    for (let i = 0; i < n; i++) {
      const [p, q] = PAIRS[i];
      const pr = [ct * p[0] - st * p[1], st * p[0] + ct * p[1]];
      const qr = [ct * q[0] - st * q[1], st * q[0] + ct * q[1]];
      const a = FT.bilerpC(S, C + pr[1] * C, C + pr[0] * C);
      const b = FT.bilerpC(S, C + qr[1] * C, C + qr[0] * C);
      out[i] = a < b ? 1 : 0;
    }
    return out;
  }
  const hamming = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] ^ b[i]; return s; };
  const MAPS = {
    id: v => v, gain: v => 1.8 * v, offset: v => v + 0.2,
    gamma: v => Math.pow(Math.max(v, 1e-6), 2.0), invert: v => 1 - v
  };
  return { PAIRS, bits, hamming, MAPS, smoothed };
})();

(function () {
  const svg = d3.select("#binary-svg");
  if (svg.empty()) return;
  const D = FTdesc, B = FTbin;
  const eN = document.getElementById("bi-n"), eNv = document.getElementById("bi-nv");
  const eS = document.getElementById("bi-s"), eSv = document.getElementById("bi-sv");
  const eM = document.getElementById("bi-m"), eR = document.getElementById("bi-r");
  const eRv = document.getElementById("bi-rv"), eP = document.getElementById("bi-p");
  const out = document.getElementById("binary-readout");

  function apply(P, f) { return P.map(row => Float64Array.from(row, f)); }
  function get() {
    const n = +eN.value, sg = +eS.value, deg = +eR.value, rotPat = eP.checked;
    const f = B.MAPS[eM.value];
    const P0 = D.sample(0);
    const P = apply(D.sample(VZ.rad(deg)), f);
    const b0 = B.bits(P0, n, sg, 0);
    const b1 = B.bits(P, n, sg, rotPat ? -VZ.rad(deg) : 0);
    return { n, sg, deg, rotPat, P0, P, b0, b1, h: B.hamming(b0, b1), map: eM.value };
  }
  const CV = new Map();
  function curve(n, sg, mapk, rotPat) {
    const key = [n, sg, mapk, rotPat].join("|");
    if (CV.has(key)) return CV.get(key);
    const f = B.MAPS[mapk], P0 = D.sample(0), b0 = B.bits(P0, n, sg, 0), rows = [];
    for (let deg = 0; deg <= 360; deg += 6) {
      const P = apply(D.sample(VZ.rad(deg)), f);
      rows.push({ deg: deg,
        fix: B.hamming(b0, B.bits(P, n, sg, 0)),
        rot: B.hamming(b0, B.bits(P, n, sg, -VZ.rad(deg))) });
    }
    CV.set(key, rows);
    return rows;
  }

  function draw() {
    const g0 = get();
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 4.1, IW = D.N * cw;

    function pane(x0, P, rot, title, bits) {
      const Bx = FT.box(g, x0, 44, IW, IW, title);
      FT.cells(Bx, 0, 0, cw, D.N, D.N, (x, y) => FT.grey(VZ.clamp(P[y][x], 0, 1)));
      const ct = Math.cos(rot), st = Math.sin(rot), c0 = (D.C + 0.5) * cw;
      for (let i = 0; i < Math.min(g0.n, 96); i++) {
        const [p, q] = B.PAIRS[i];
        const pr = [ct * p[0] - st * p[1], st * p[0] + ct * p[1]];
        const qr = [ct * q[0] - st * q[1], st * q[0] + ct * q[1]];
        Bx.append("line").attr("x1", c0 + pr[0] * D.C * cw).attr("y1", c0 + pr[1] * D.C * cw)
          .attr("x2", c0 + qr[0] * D.C * cw).attr("y2", c0 + qr[1] * D.C * cw)
          .attr("stroke", bits[i] ? VC.a2 : VC.accent).attr("stroke-width", 0.9)
          .attr("stroke-opacity", 0.85);
      }
      return Bx;
    }
    pane(14, g0.P0, 0, "reference", g0.b0);
    pane(14 + IW + 22, g0.P, g0.rotPat ? -VZ.rad(g0.deg) : 0,
      g0.deg + "°" + (g0.rotPat ? ", pattern rotated" : ""), g0.b1);

    /* the two bit strings */
    const BY = 44 + IW + 34, bw = Math.min(6, 300 / g0.n * 2), perRow = Math.ceil(g0.n / 4);
    g.append("text").attr("x", 14).attr("y", BY - 6).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("the two bit strings, differing bits in red");
    for (let i = 0; i < g0.n; i++) {
      const r = Math.floor(i / perRow), c = i % perRow;
      [0, 1].forEach(k => {
        const v = k ? g0.b1[i] : g0.b0[i];
        g.append("rect").attr("x", 14 + c * bw).attr("y", BY + r * 16 + k * 7)
          .attr("width", Math.max(1, bw - 0.4)).attr("height", 6)
          .attr("fill", g0.b0[i] !== g0.b1[i] ? VC.bad : (v ? VC.a2 : VC.accent))
          .attr("fill-opacity", g0.b0[i] !== g0.b1[i] ? 1 : 0.55);
      });
    }

    /* ---- right: the distance and the curve -------------------------------- */
    const RX = 14 + 2 * IW + 46, CW2 = 232, CH2 = 150;
    const rows = curve(g0.n, g0.sg, g0.map, g0.rotPat);
    const gc = g.append("g").attr("transform", `translate(${RX},60)`);
    const x = d3.scaleLinear().domain([0, 360]).range([0, CW2]);
    const y = d3.scaleLinear().domain([0, g0.n]).range([CH2, 0]);
    VZ.gridY(gc, y, CW2, 4);
    VZ.axisB(gc, x, CH2, 5, "rotation (°)");
    gc.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
    gc.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.ink)
      .text("Hamming distance, " + g0.n + " bits");
    gc.append("line").attr("x1", 0).attr("x2", CW2).attr("y1", y(g0.n / 2)).attr("y2", y(g0.n / 2))
      .attr("stroke", VC.bad).attr("stroke-dasharray", "4 3");
    gc.append("text").attr("x", CW2).attr("y", y(g0.n / 2) - 4).attr("text-anchor", "end")
      .attr("font-size", 9.5).attr("fill", VC.bad).text("n/2 = chance");
    gc.append("path").attr("d", d3.line().x(d => x(d.deg)).y(d => y(d.fix))(rows))
      .attr("fill", "none").attr("stroke", VC.bad).attr("stroke-width", 2);
    gc.append("path").attr("d", d3.line().x(d => x(d.deg)).y(d => y(d.rot))(rows))
      .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 2);
    gc.append("line").attr("x1", x(g0.deg)).attr("x2", x(g0.deg)).attr("y1", 0).attr("y2", CH2)
      .attr("stroke", VC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.6);
    VZ.legend(gc, [{ color: VC.good, label: "pattern rotated too" },
      { color: VC.bad, label: "pattern fixed" }], 6, 12, { gap: 14 });

    const put = FT.kv(g, RX, 60 + CH2 + 54, { keyW: 176, lead: 15.5 });
    put("Hamming distance", g0.h + " of " + g0.n, g0.h < g0.n / 6 ? VC.good : VC.bad, true);
    put("as a fraction", VZ.fmt(g0.h / g0.n, 4), VC.ink, true);
    put("chance level", (g0.n / 2) + " (" + VZ.fmt(0.5, 2) + ")", VC.muted);
    put("descriptor size", (g0.n / 8) + " bytes", VC.good, true);
    put("128 floats would be", "512 bytes", VC.muted);
    put("distance cost", Math.ceil(g0.n / 64) + " XOR + " + Math.ceil(g0.n / 64) + " popcount", VC.good, true);
  }

  function say() {
    const g0 = get();
    const names = { id: "no intensity change", gain: "a gain of 1.8", offset: "an offset of +0.2",
      gamma: "a gamma of 2.0", invert: "an inversion" };
    const mono = g0.map !== "invert";
    out.innerHTML =
      `At ${g0.deg}° with ${names[g0.map]}: Hamming distance <b>${g0.h}</b> of <b>${g0.n}</b> bits `
      + `(<b>${VZ.fmt(g0.h / g0.n, 3)}</b>), against a chance level of <b>${g0.n / 2}</b>.<br>`
      + (g0.deg === 0
        ? (g0.map === "id"
          ? `With no transformation at all the distance is <b>${g0.h}</b>, as it must be. `
          : (mono
            ? `The intensity map is <b>monotonically increasing</b>, so every comparison <code>I(p) &lt; I(q)</code> `
              + `gives the same answer it did before and the distance is <b>${g0.h}</b>. That is a stronger `
              + `guarantee than a gradient histogram's: not merely affine invariance but invariance to <i>any</i> `
              + `monotonically increasing tone curve, and it is free rather than fitted. `
            : `Inversion is monotonically <b>DECREASING</b>, so every comparison flips and the distance is `
              + `<b>${g0.h}</b> of ${g0.n} — the complement. A descriptor that is invariant to monotone increasing `
              + `maps is <i>anti</i>-invariant to monotone decreasing ones, which is why polarity has to be `
              + `recorded separately (§18) rather than assumed away. `))
        : (g0.rotPat
          ? `With the sampling pattern rotated by the same angle, the distance stays low across the whole turn — `
            + `the green curve. In practice the pattern is precomputed at about 30 discrete angles, which `
            + `quantises the orientation to 12° and puts a floor under that curve. `
          : `With the pattern fixed, the distance climbs to the chance level of ${g0.n / 2} within a few tens `
            + `of degrees and then past it — the red curve. Going <i>above</i> chance is not a bug: at a `
            + `half-turn the pattern samples the patch anti-symmetrically, so the comparisons are `
            + `anti-correlated rather than merely uncorrelated. A binary descriptor has no rotation invariance `
            + `of its own; the pattern must be steered, which is what the checkbox does — and with it steered `
            + `the distance is exactly zero at every angle on this noiseless synthetic patch. `))
      + `Note the distance is an <b>integer</b>, so ties are common and the ratio test of §25 is coarser than it `
      + `is on a continuous distance — a real and often-unmentioned cost of going binary.`;
  }
  function all() {
    eNv.textContent = eN.value; eSv.textContent = (+eS.value).toFixed(2); eRv.textContent = eR.value + "°";
    draw(); say();
  }
  eN.oninput = all; eS.oninput = all; eM.onchange = all; eR.oninput = all; eP.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   21 · #ratio-svg — the ratio test against a distance threshold
   ══════════════════════════════════════════════════════════════════════════ */
const FTmatch = (function () {
  const DIM = 64, NA = 400, PM = 0.60, NCLUT = 250, NPROTO = 14, JIT = 0.12;
  /* A synthetic matching problem with KNOWN correspondences. Two ingredients make it
     realistic rather than trivial: a share of features drawn near one of a few shared
     "texture prototypes", so descriptor space has a varying density; and a per-feature
     deformation severity drawn from a log-normal, so some true matches are far apart
     while some wrong ones are close. Without both, every strategy scores 1.0. */
  const CACHE = new Map();
  function build(sevMed, prep) {
    const key = sevMed.toFixed(3) + "|" + prep.toFixed(3);
    if (CACHE.has(key)) return CACHE.get(key);
    const r = VZ.rng(5150);
    const unit = v => { const n = Math.hypot(...v); return v.map(x => x / n); };
    const proto = [];
    for (let i = 0; i < NPROTO; i++) proto.push(unit(Array.from({ length: DIM }, () => VZ.randn(r))));
    const A = [], kind = [], sev = [];
    for (let i = 0; i < NA; i++) {
      let v;
      if (r() < prep) {
        const p = proto[Math.floor(r() * NPROTO)];
        v = p.map(x => x + JIT * VZ.randn(r)); kind.push(1);
      } else { v = Array.from({ length: DIM }, () => VZ.randn(r)); kind.push(0); }
      A.push(unit(v));
      sev.push(Math.exp(Math.log(sevMed) + 0.55 * VZ.randn(r)));
    }
    const B = [], gt = new Int32Array(NA).fill(-1);
    for (let i = 0; i < NA; i++) {
      if (r() >= PM) continue;
      gt[i] = B.length;
      B.push(unit(A[i].map(x => x + sev[i] / Math.sqrt(DIM) * VZ.randn(r))));
    }
    for (let j = 0; j < NCLUT; j++) {
      let v;
      if (r() < prep) { const p = proto[Math.floor(r() * NPROTO)]; v = p.map(x => x + 0.28 * VZ.randn(r)); }
      else v = Array.from({ length: DIM }, () => VZ.randn(r));
      B.push(unit(v));
    }
    /* exhaustive nearest-neighbour search — nothing here is approximate */
    const d1 = new Float64Array(NA), d2 = new Float64Array(NA), nn = new Int32Array(NA);
    const back = new Int32Array(B.length).fill(-1), bbest = new Float64Array(B.length).fill(Infinity);
    for (let i = 0; i < NA; i++) {
      let b1 = Infinity, b2 = Infinity, bi = -1;
      for (let j = 0; j < B.length; j++) {
        let s = 0;
        for (let k = 0; k < DIM; k++) { const t = A[i][k] - B[j][k]; s += t * t; }
        s = Math.sqrt(s);
        if (s < b1) { b2 = b1; b1 = s; bi = j; } else if (s < b2) b2 = s;
        if (s < bbest[j]) { bbest[j] = s; back[j] = i; }
      }
      d1[i] = b1; d2[i] = b2; nn[i] = bi;
    }
    const correct = new Uint8Array(NA), mutual = new Uint8Array(NA);
    let P = 0;
    for (let i = 0; i < NA; i++) {
      if (gt[i] >= 0) P++;
      correct[i] = (gt[i] >= 0 && nn[i] === gt[i]) ? 1 : 0;
      mutual[i] = back[nn[i]] === i ? 1 : 0;
    }
    const res = { A, B, gt, d1, d2, nn, correct, mutual, P, kind, nB: B.length };
    CACHE.set(key, res);
    return res;
  }
  function score(M, mask) {
    let tp = 0, fp = 0;
    for (let i = 0; i < M.d1.length; i++) if (mask[i]) { if (M.correct[i]) tp++; else fp++; }
    const prec = (tp + fp) ? tp / (tp + fp) : 1, rec = tp / M.P;
    return { tp, fp, fn: M.P - tp, prec, rec, f: (prec + rec) ? 2 * prec * rec / (prec + rec) : 0 };
  }
  function sweep(M, kind, xcheck, n) {
    const rows = [];
    const hi = kind === "ratio" ? 1.0 : Math.max(...M.d1) * 1.02;
    for (let i = 0; i <= (n || 120); i++) {
      const t = hi * i / (n || 120);
      const mask = new Uint8Array(M.d1.length);
      for (let j = 0; j < M.d1.length; j++) {
        const v = kind === "ratio" ? (M.d2[j] > 0 ? M.d1[j] / M.d2[j] : 0) : M.d1[j];
        mask[j] = (v < t && (!xcheck || M.mutual[j])) ? 1 : 0;
      }
      rows.push(Object.assign({ t: t }, score(M, mask)));
    }
    return rows;
  }
  return { DIM, build, score, sweep };
})();

(function () {
  const svg = d3.select("#ratio-svg");
  if (svg.empty()) return;
  const MM = FTmatch;
  const eS = document.getElementById("rt-s");
  const eT = document.getElementById("rt-t"), eTv = document.getElementById("rt-tv");
  const eX = document.getElementById("rt-x");
  const eD = document.getElementById("rt-d"), eDv = document.getElementById("rt-dv");
  const eR = document.getElementById("rt-r"), eRv = document.getElementById("rt-rv");
  const out = document.getElementById("ratio-readout");

  function get() {
    const kind = eS.value, xc = eX.checked;
    const M = MM.build(+eD.value, +eR.value);
    const hi = kind === "ratio" ? 1.0 : Math.max(...M.d1) * 1.02;
    const t = (+eT.value) * hi;
    const mask = new Uint8Array(M.d1.length);
    for (let j = 0; j < M.d1.length; j++) {
      const v = kind === "ratio" ? (M.d2[j] > 0 ? M.d1[j] / M.d2[j] : 0) : M.d1[j];
      mask[j] = (v < t && (!xc || M.mutual[j])) ? 1 : 0;
    }
    return { kind, xc, M, t, hi, mask, sc: MM.score(M, mask),
      swR: MM.sweep(M, "ratio", xc), swD: MM.sweep(M, "dist", xc) };
  }

  function hist(g, x0, y0, W, H, vals, sel, title, t, hi) {
    const NB = 34, a = new Float64Array(NB), b = new Float64Array(NB);
    for (let i = 0; i < vals.length; i++) {
      const k = Math.min(NB - 1, Math.floor(vals[i] / hi * NB));
      if (sel[i]) a[k]++; else b[k]++;
    }
    const mx = Math.max(...a, ...b, 1);
    const gg = g.append("g").attr("transform", `translate(${x0},${y0})`);
    gg.append("text").attr("x", 0).attr("y", -7).attr("font-size", 10.5).attr("fill", VC.ink).text(title);
    const x = d3.scaleLinear().domain([0, hi]).range([0, W]);
    const y = d3.scaleLinear().domain([0, mx * 1.08]).range([H, 0]);
    VZ.axisB(gg, x, H, 4);
    for (let k = 0; k < NB; k++) {
      gg.append("rect").attr("x", x(hi * k / NB)).attr("y", y(a[k])).attr("width", W / NB - 0.6)
        .attr("height", H - y(a[k])).attr("fill", VC.good).attr("fill-opacity", 0.75);
      gg.append("rect").attr("x", x(hi * k / NB)).attr("y", y(b[k])).attr("width", W / NB - 0.6)
        .attr("height", H - y(b[k])).attr("fill", VC.bad).attr("fill-opacity", 0.55);
    }
    if (t !== null) gg.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", 0).attr("y2", H)
      .attr("stroke", VC.ink).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
    return gg;
  }

  function draw() {
    const g0 = get(), M = g0.M;
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const HW = 226, HH = 118;
    const ratios = Array.from(M.d1, (v, i) => M.d2[i] > 0 ? v / M.d2[i] : 0);
    hist(g, 44, 52, HW, HH, Array.from(M.d1), M.correct, "d₁ · correct (green) vs wrong (red)",
      g0.kind === "dist" ? g0.t : null, Math.max(...M.d1) * 1.02);
    hist(g, 44, 52 + HH + 62, HW, HH, ratios, M.correct, "d₁/d₂ — the overlap is far smaller",
      g0.kind === "ratio" ? g0.t : null, 1.0);
    VZ.legend(g, [{ color: VC.good, label: "NN is the true match" },
      { color: VC.bad, label: "NN is wrong" }], 44, 52 + 2 * HH + 92, { gap: 0, vertical: false, step: 148 });

    /* ---- middle: precision and recall against the threshold --------------- */
    const MX = 44 + HW + 56, CW2 = 208, CH2 = 148;
    const gc = g.append("g").attr("transform", `translate(${MX},52)`);
    const sw = g0.kind === "ratio" ? g0.swR : g0.swD;
    const x = d3.scaleLinear().domain([0, g0.hi]).range([0, CW2]);
    const y = d3.scaleLinear().domain([0, 1.02]).range([CH2, 0]);
    VZ.gridY(gc, y, CW2, 5);
    VZ.axisB(gc, x, CH2, 4, g0.kind === "ratio" ? "ρ" : "τ");
    gc.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5, ".0%"));
    gc.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.ink)
      .text("precision and recall");
    gc.append("path").attr("d", d3.line().x(d => x(d.t)).y(d => y(d.prec))(sw))
      .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 2);
    gc.append("path").attr("d", d3.line().x(d => x(d.t)).y(d => y(d.rec))(sw))
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 2);
    gc.append("line").attr("x1", x(g0.t)).attr("x2", x(g0.t)).attr("y1", 0).attr("y2", CH2)
      .attr("stroke", VC.ink).attr("stroke-dasharray", "3 3");
    VZ.legend(gc, [{ color: VC.accent, label: "precision" }, { color: VC.a2, label: "recall" }],
      6, 12, { gap: 14 });

    /* the PR curves for both strategies */
    const gp = g.append("g").attr("transform", `translate(${MX},${52 + CH2 + 62})`);
    const xr = d3.scaleLinear().domain([0, 1.02]).range([0, CW2]);
    const yp = d3.scaleLinear().domain([0, 1.02]).range([CH2, 0]);
    VZ.gridY(gp, yp, CW2, 5);
    VZ.axisB(gp, xr, CH2, 5, "recall", d3.format(".0%"));
    gp.append("g").attr("class", "axis").call(d3.axisLeft(yp).ticks(5, ".0%"));
    gp.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.ink)
      .text("precision–recall, both strategies");
    [[g0.swR, VC.good, "ratio"], [g0.swD, VC.bad, "distance"]].forEach(([rows, col]) => {
      gp.append("path").attr("d", d3.line().x(d => xr(d.rec)).y(d => yp(d.prec))(rows))
        .attr("fill", "none").attr("stroke", col).attr("stroke-width", 2);
    });
    gp.append("circle").attr("cx", xr(g0.sc.rec)).attr("cy", yp(g0.sc.prec)).attr("r", 5)
      .attr("fill", "none").attr("stroke", VC.ink).attr("stroke-width", 2);
    VZ.legend(gp, [{ color: VC.good, label: "ratio test" }, { color: VC.bad, label: "distance threshold" }],
      6, 12, { gap: 14 });

    /* ---- right: the counts ------------------------------------------------ */
    const RX = MX + CW2 + 48;
    const bestR = g0.swR.reduce((a, b) => b.f > a.f ? b : a);
    const bestD = g0.swD.reduce((a, b) => b.f > a.f ? b : a);
    const put = FT.kv(g, RX, 60, { keyW: 172, lead: 15.5 });
    put("features in A", M.d1.length, VC.ink, true);
    put("matchable", M.P, VC.ink, true);
    put("NN is the true match for", Array.from(M.correct).reduce((a, b) => a + b, 0)
      + "  (" + VZ.fmt(100 * Array.from(M.correct).reduce((a, b) => a + b, 0) / M.P, 1) + "%)", VC.good, true);
    put("", "", VC.ink);
    put("threshold", VZ.fmt(g0.t, 3) + (g0.xc ? "  + cross-check" : ""), VC.a2, true);
    put("true positives", g0.sc.tp, VC.good, true);
    put("false positives", g0.sc.fp, g0.sc.fp ? VC.bad : VC.good, true);
    put("false negatives", g0.sc.fn, VC.muted, true);
    put("precision", VZ.fmt(g0.sc.prec, 4), g0.sc.prec > 0.95 ? VC.good : VC.bad, true);
    put("recall", VZ.fmt(g0.sc.rec, 4), VC.a2, true);
    put("F", VZ.fmt(g0.sc.f, 4), VC.ink, true);
    put("", "", VC.ink);
    put("best F, ratio test", VZ.fmt(bestR.f, 4) + " at ρ = " + VZ.fmt(bestR.t, 3), VC.good, true);
    put("best F, distance", VZ.fmt(bestD.f, 4) + " at τ = " + VZ.fmt(bestD.t, 3), VC.bad, true);
  }

  function say() {
    const g0 = get(), M = g0.M;
    const bestR = g0.swR.reduce((a, b) => b.f > a.f ? b : a);
    const bestD = g0.swD.reduce((a, b) => b.f > a.f ? b : a);
    /* the fair comparison: the other strategy at the SAME recall */
    const other = g0.kind === "ratio" ? g0.swD : g0.swR;
    let mate = other[other.length - 1];
    for (const r of other) if (r.rec >= g0.sc.rec) { mate = r; break; }
    out.innerHTML =
      `<b>${M.d1.length}</b> features, <b>${M.P}</b> of them matchable, and the nearest neighbour is the true `
      + `match for <b>${Array.from(M.correct).reduce((a, b) => a + b, 0)}</b> of those. At the current threshold: `
      + `TP <b>${g0.sc.tp}</b>, FP <b>${g0.sc.fp}</b>, FN <b>${g0.sc.fn}</b> — precision `
      + `<b>${VZ.fmt(g0.sc.prec, 4)}</b>, recall <b>${VZ.fmt(g0.sc.rec, 4)}</b>.<br>`
      + `At the same recall the <b>other</b> strategy gives precision <b>${VZ.fmt(mate.prec, 4)}</b> with `
      + `<b>${mate.fp}</b> false positives against this one's <b>${g0.sc.fp}</b>. Over the whole sweep the best `
      + `F is <b>${VZ.fmt(bestR.f, 4)}</b> for the ratio test and <b>${VZ.fmt(bestD.f, 4)}</b> for the distance `
      + `threshold.<br>`
      + `The mechanism is in the two histograms on the left. The <code>d₁</code> distributions for correct and `
      + `wrong nearest neighbours <i>overlap</i>, because a badly deformed correct match can be farther away `
      + `than a lucky wrong one — so no vertical line separates them. Dividing by <code>d₂</code> normalises `
      + `each feature by the local density of descriptor space around it, and the two distributions pull apart. `
      + `Drag the repetitive fraction to zero and watch the advantage shrink: with no near-duplicate structure, `
      + `descriptor space has a uniform density and there is nothing for <code>d₂</code> to correct for.`;
  }
  function all() {
    eTv.textContent = (+eT.value).toFixed(2); eDv.textContent = (+eD.value).toFixed(2);
    eRv.textContent = (+eR.value).toFixed(2);
    draw(); say();
  }
  eS.onchange = all; eT.oninput = all; eX.onchange = all; eD.oninput = all; eR.oninput = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   22 · #ransac-svg — RANSAC, with the iteration formula measured
   ══════════════════════════════════════════════════════════════════════════ */
const FTransac = (function () {
  const NP = 100;
  const TRUE = { n: VZ.unit([-0.6, 1.0]), c: 0.1 / Math.hypot(-0.6, 1.0) };
  /* the data: w·NP inliers on the true line with Gaussian noise, the rest uniform */
  function data(w, seed, sigma) {
    const r = VZ.rng(seed), pts = [], ni = Math.round(w * NP);
    for (let i = 0; i < ni; i++) {
      const x = r() * 2 - 1, y = 0.6 * x + 0.1 + (sigma === undefined ? 0.02 : sigma) * VZ.randn(r);
      pts.push({ x: x, y: y, in: true });
    }
    for (let i = ni; i < NP; i++) pts.push({ x: r() * 2 - 1, y: r() * 2 - 1, in: false });
    return { pts: pts, ni: ni };
  }
  const Nform = (p, w, s) => Math.log(1 - p) / Math.log(1 - Math.pow(w, s));
  /* one complete run, keeping the whole history so the figure can step through it */
  function run(D, iters, tol, seed, adaptive, p) {
    const r = VZ.rng(seed), hist = [];
    let best = null, sawClean = false, need = iters;
    for (let it = 0; it < iters; it++) {
      if (adaptive && it >= need) break;
      const i = Math.floor(r() * D.pts.length);
      let j = Math.floor(r() * D.pts.length);
      if (j === i) j = (j + 1) % D.pts.length;
      const a = D.pts[i], b = D.pts[j];
      const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy);
      if (L < 1e-9) { hist.push(null); continue; }
      const n = [-dy / L, dx / L], c = n[0] * a.x + n[1] * a.y;
      let cnt = 0;
      for (const q of D.pts) if (Math.abs(q.x * n[0] + q.y * n[1] - c) < tol) cnt++;
      const clean = a.in && b.in;
      if (clean) sawClean = true;
      const rec = { it: it, i: i, j: j, n: n, c: c, cnt: cnt, clean: clean };
      if (!best || cnt > best.cnt) best = rec;
      rec.best = best.cnt;
      if (adaptive) {
        const wh = VZ.clamp(best.cnt / D.pts.length, 0.05, 0.999);
        need = Math.min(iters, Math.ceil(Nform(p || 0.99, wh, 2)));
      }
      rec.need = need;
      hist.push(rec);
    }
    return { hist: hist, best: best, sawClean: sawClean, used: hist.length };
  }
  const angDiff = (a, b) => Math.abs(((a - b + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI);
  function correct(best) {
    if (!best) return false;
    const cosang = Math.abs(best.n[0] * TRUE.n[0] + best.n[1] * TRUE.n[1]);
    return cosang > Math.cos(VZ.rad(3)) && Math.abs(Math.abs(best.c) - Math.abs(TRUE.c)) < 0.05;
  }
  /* the Monte-Carlo check: many independent complete runs at these settings */
  const MC = new Map();
  function measure(w, tol, N, trials, p) {
    const key = [w, tol, N, trials, p].join("|");
    if (MC.has(key)) return MC.get(key);
    let clean = 0, ok = 0, okGivenClean = 0, meanIn = 0;
    for (let t = 0; t < trials; t++) {
      const D = data(w, 1000 + t * 7);
      const R = run(D, N, tol, 50000 + t * 13, false, p);
      const c = correct(R.best);
      clean += R.sawClean ? 1 : 0;
      ok += c ? 1 : 0;
      okGivenClean += (c && R.sawClean) ? 1 : 0;
      meanIn += R.best ? R.best.cnt : 0;
    }
    const out = { pClean: clean / trials, pOK: ok / trials,
      pOKgiven: clean ? okGivenClean / clean : 0, meanIn: meanIn / trials, trials: trials };
    MC.set(key, out);
    return out;
  }
  return { NP, TRUE, data, Nform, run, correct, measure };
})();

(function () {
  const svg = d3.select("#ransac-svg");
  if (svg.empty()) return;
  const RN = FTransac;
  const eW = document.getElementById("rn-w"), eWv = document.getElementById("rn-wv");
  const eI = document.getElementById("rn-i"), eIv = document.getElementById("rn-iv");
  const eT = document.getElementById("rn-t"), eTv = document.getElementById("rn-tv");
  const eP = document.getElementById("rn-p"), ePv = document.getElementById("rn-pv");
  const eA = document.getElementById("rn-a");
  const out = document.getElementById("ransac-readout");

  function get() {
    const w = +eW.value, tol = +eT.value, p = +eP.value, adapt = eA.checked;
    const Nreq = Math.ceil(RN.Nform(p, w, 2));
    const D = RN.data(w, 4242);
    const R = RN.run(D, 120, tol, 777, adapt, p);
    const it = Math.min(+eI.value, R.hist.length) - 1;
    return { w, tol, p, adapt, Nreq, D, R, it: Math.max(0, it) };
  }
  function seg(n, c, S) {                        // clip a line to [−1, 1]²
    const t = [-n[1], n[0]], p0 = [n[0] * c, n[1] * c], hits = [];
    for (const s of [-1, 1]) {
      if (Math.abs(t[0]) > 1e-9) { const u = (s - p0[0]) / t[0], y = p0[1] + u * t[1]; if (Math.abs(y) <= 1.001) hits.push([s, y]); }
      if (Math.abs(t[1]) > 1e-9) { const u = (s - p0[1]) / t[1], x = p0[0] + u * t[0]; if (Math.abs(x) <= 1.001) hits.push([x, s]); }
    }
    return hits.length >= 2 ? hits.slice(0, 2).map(S) : null;
  }

  function draw() {
    const g0 = get();
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const PS = 240;
    const sx = d3.scaleLinear().domain([-1, 1]).range([0, PS]);
    const sy = d3.scaleLinear().domain([-1, 1]).range([PS, 0]);
    const S = q => [sx(q[0]), sy(q[1])];
    const B1 = FT.box(g, 44, 52, PS, PS, "iteration " + (g0.it + 1) + " of " + g0.R.used);
    const cur = g0.R.hist[g0.it];
    /* the tolerance band of the current hypothesis */
    if (cur) {
      [-1, 1].forEach(sgn => {
        const s2 = seg(cur.n, cur.c + sgn * g0.tol, S);
        if (s2) B1.append("line").attr("x1", s2[0][0]).attr("y1", s2[0][1])
          .attr("x2", s2[1][0]).attr("y2", s2[1][1])
          .attr("stroke", VC.accent).attr("stroke-width", 1).attr("stroke-dasharray", "3 3");
      });
    }
    g0.D.pts.forEach(q => {
      const inCur = cur && Math.abs(q.x * cur.n[0] + q.y * cur.n[1] - cur.c) < g0.tol;
      B1.append("circle").attr("cx", sx(q.x)).attr("cy", sy(q.y)).attr("r", 2.6)
        .attr("fill", inCur ? VC.accent : (q.in ? VC.good : VC.bad))
        .attr("fill-opacity", inCur ? 1 : 0.45);
    });
    const st = seg(RN.TRUE.n, RN.TRUE.c, S);
    if (st) B1.append("line").attr("x1", st[0][0]).attr("y1", st[0][1]).attr("x2", st[1][0]).attr("y2", st[1][1])
      .attr("stroke", VC.good).attr("stroke-width", 1.2).attr("stroke-dasharray", "6 3");
    if (cur) {
      const sc = seg(cur.n, cur.c, S);
      if (sc) B1.append("line").attr("x1", sc[0][0]).attr("y1", sc[0][1]).attr("x2", sc[1][0]).attr("y2", sc[1][1])
        .attr("stroke", VC.accent).attr("stroke-width", 2);
      [g0.D.pts[cur.i], g0.D.pts[cur.j]].forEach(q => B1.append("circle").attr("cx", sx(q.x)).attr("cy", sy(q.y))
        .attr("r", 5).attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 2));
    }
    const bs = g0.R.best ? seg(g0.R.best.n, g0.R.best.c, S) : null;
    if (bs) B1.append("line").attr("x1", bs[0][0]).attr("y1", bs[0][1]).attr("x2", bs[1][0]).attr("y2", bs[1][1])
      .attr("stroke", VC.violet).attr("stroke-width", 2.4).attr("stroke-opacity", 0.9);
    VZ.legend(B1, [{ color: VC.good, label: "true inlier" }, { color: VC.bad, label: "outlier" },
      { color: VC.accent, label: "in this hypothesis" }, { color: VC.violet, label: "best so far" }],
      6, PS - 56, { gap: 13 });

    /* ---- middle: consensus counts against iteration ----------------------- */
    const MX = 44 + PS + 42, CW2 = 216, CH2 = 200;
    const gc = g.append("g").attr("transform", `translate(${MX},52)`);
    const x = d3.scaleLinear().domain([0, Math.max(g0.R.used, g0.Nreq * 1.05)]).range([0, CW2]);
    const y = d3.scaleLinear().domain([0, RN.NP]).range([CH2, 0]);
    VZ.gridY(gc, y, CW2, 5);
    VZ.axisB(gc, x, CH2, 5, "iteration");
    gc.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    gc.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.ink)
      .text("consensus size");
    gc.append("line").attr("x1", 0).attr("x2", CW2).attr("y1", y(g0.D.ni)).attr("y2", y(g0.D.ni))
      .attr("stroke", VC.good).attr("stroke-dasharray", "4 3");
    gc.append("text").attr("x", 3).attr("y", y(g0.D.ni) - 4).attr("font-size", 9.5).attr("fill", VC.good)
      .text("true inliers: " + g0.D.ni);
    g0.R.hist.forEach(h => { if (!h) return;
      gc.append("circle").attr("cx", x(h.it)).attr("cy", y(h.cnt)).attr("r", 2)
        .attr("fill", h.clean ? VC.good : VC.muted).attr("fill-opacity", 0.8); });
    gc.append("path").attr("d", d3.line().x(d => x(d.it)).y(d => y(d.best)).curve(d3.curveStepAfter)(
      g0.R.hist.filter(Boolean))).attr("fill", "none").attr("stroke", VC.violet).attr("stroke-width", 2);
    gc.append("line").attr("x1", x(g0.Nreq)).attr("x2", x(g0.Nreq)).attr("y1", 0).attr("y2", CH2)
      .attr("stroke", VC.a2).attr("stroke-width", 1.6).attr("stroke-dasharray", "5 3");
    gc.append("text").attr("x", x(g0.Nreq) + 4).attr("y", 12).attr("font-size", 9.5).attr("fill", VC.a2)
      .text("N = " + g0.Nreq);
    if (g0.adapt) {
      gc.append("path").attr("d", d3.line().x(d => x(d.it)).y(d => Math.max(0, Math.min(CH2, y(0) - (CH2 * d.need / Math.max(g0.R.used, g0.Nreq * 1.05)))))
        .curve(d3.curveStepAfter)(g0.R.hist.filter(Boolean)))
        .attr("fill", "none").attr("stroke", VC.teal).attr("stroke-width", 1.6).attr("stroke-dasharray", "3 3");
      gc.append("text").attr("x", 3).attr("y", CH2 - 6).attr("font-size", 9.5).attr("fill", VC.teal)
        .text("adaptive N (scaled to the axis)");
    }
    gc.append("line").attr("x1", x(g0.it)).attr("x2", x(g0.it)).attr("y1", 0).attr("y2", CH2)
      .attr("stroke", VC.ink).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.6);

    /* ---- right: the formula, measured ------------------------------------- */
    const RX = MX + CW2 + 46;
    const M = RN.measure(g0.w, g0.tol, g0.Nreq, 400, g0.p);
    const pred = 1 - Math.pow(1 - g0.w * g0.w, g0.Nreq);
    g.append("text").attr("x", RX).attr("y", 40).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("the formula, measured over 400 runs");
    const put = FT.kv(g, RX, 62, { keyW: 182, lead: 15.5 });
    put("w  (inlier fraction)", g0.w.toFixed(2) + "  (" + g0.D.ni + " of " + RN.NP + ")", VC.good, true);
    put("s  (minimal sample)", "2", VC.ink);
    put("target p", g0.p.toFixed(3), VC.ink);
    put("N = log(1−p)/log(1−wˢ)", VZ.fmt(RN.Nform(g0.p, g0.w, 2), 2) + " → " + g0.Nreq, VC.a2, true);
    put("", "", VC.ink);
    put("formula: P(clean sample)", VZ.fmt(pred, 4), VC.a2, true);
    put("measured P(clean sample)", VZ.fmt(M.pClean, 4), Math.abs(M.pClean - pred) < 0.02 ? VC.good : VC.bad, true);
    put("difference", VZ.fmt(M.pClean - pred, 4), VC.muted, true);
    put("", "", VC.ink);
    put("measured P(right model)", VZ.fmt(M.pOK, 4), VC.violet, true);
    put("shortfall vs the formula", VZ.fmt(pred - M.pOK, 4), pred - M.pOK > 0.01 ? VC.bad : VC.good, true);
    put("P(right | clean sample)", VZ.fmt(M.pOKgiven, 4), VC.violet, true);
    put("mean consensus found", VZ.fmt(M.meanIn, 1) + " of " + g0.D.ni + " true", VC.ink, true);
    /* a small bar comparison */
    const BY = 62 + 13 * 15.5 + 16, BW = 190;
    [["formula p", pred, VC.a2], ["clean, measured", M.pClean, VC.good], ["correct, measured", M.pOK, VC.violet]]
      .forEach((row, i) => {
        g.append("text").attr("x", RX).attr("y", BY + i * 20).attr("font-size", 10).attr("fill", VC.muted).text(row[0]);
        g.append("rect").attr("x", RX + 104).attr("y", BY + i * 20 - 9).attr("width", BW * row[1] * 0 + Math.max(1, (BW - 104) * ((row[1] - 0.9) / 0.1)))
          .attr("height", 11).attr("fill", row[2]).attr("fill-opacity", 0.85);
        g.append("text").attr("x", RX + 104 + Math.max(1, (BW - 104) * ((row[1] - 0.9) / 0.1)) + 5)
          .attr("y", BY + i * 20).attr("font-size", 9.5).attr("fill", row[2]).text(VZ.fmt(row[1], 4));
      });
    g.append("text").attr("x", RX).attr("y", BY + 3 * 20 + 6).attr("font-size", 9)
      .attr("fill", VC.muted).text("bars are stretched over [0.90, 1.00]");
  }

  function say() {
    const g0 = get();
    const M = RN.measure(g0.w, g0.tol, g0.Nreq, 400, g0.p);
    const pred = 1 - Math.pow(1 - g0.w * g0.w, g0.Nreq);
    const cur = g0.R.hist[g0.it];
    out.innerHTML =
      `<b>${g0.D.ni}</b> inliers among <b>${RN.NP}</b> points, so <b>${(100 * (1 - g0.w)).toFixed(0)}%</b> of the `
      + `data is nonsense. The formula asks for <code>N = log(1−${g0.p.toFixed(3)})/log(1−${g0.w.toFixed(2)}²) = `
      + `${VZ.fmt(RN.Nform(g0.p, g0.w, 2), 2)}</code>, so <b>${g0.Nreq}</b> iterations. `
      + (cur ? `The current hypothesis was drawn from ${cur.clean ? "<b>two inliers</b>" : "at least one outlier"} `
        + `and collects <b>${cur.cnt}</b> points; the best so far collects <b>${g0.R.best.cnt}</b>.` : "")
      + `<br>Over <b>400</b> independent complete runs at these settings: the formula predicts `
      + `<b>${VZ.fmt(pred, 4)}</b> for the probability that some minimal sample is all-inlier, and the measured `
      + `value is <b>${VZ.fmt(M.pClean, 4)}</b> — agreement to `
      + `<b>${VZ.fmt(Math.abs(M.pClean - pred), 4)}</b>, with the small deficit coming from sampling <i>without</i> `
      + `replacement, which <code>wˢ</code> ignores. But the probability of actually recovering the right line is `
      + `<b>${VZ.fmt(M.pOK, 4)}</b>, <b>${VZ.fmt(pred - M.pOK, 4)}</b> lower, and given a clean sample it is still `
      + `only <b>${VZ.fmt(M.pOKgiven, 4)}</b>. Two inliers that happen to lie close together define a badly `
      + `conditioned line, and no value of N repairs that. <b>The formula bounds the sampling, not the estimate.</b> `
      + `Slide w upward and the shortfall <i>grows</i> rather than shrinking — it is largest where N is smallest, `
      + `because a handful of iterations gives few chances for a well-conditioned clean sample to turn up.`
      + (g0.adapt
        ? ` With adaptive updating on, the run stopped after <b>${g0.R.used}</b> iterations rather than the `
          + `${g0.Nreq} the pessimistic w demanded.`
        : ` Turn on adaptive N and watch the demand collapse once a good consensus is found.`);
  }
  function all() {
    eWv.textContent = (+eW.value).toFixed(2); eIv.textContent = eI.value;
    eTv.textContent = (+eT.value).toFixed(3); ePv.textContent = (+eP.value).toFixed(3);
    draw(); say();
  }
  eW.oninput = all; eI.oninput = all; eT.oninput = all; eP.oninput = all; eA.onchange = all;
  all();
})();

/* ══════════════════════════════════════════════════════════════════════════
   23 · #eval-svg — repeatability under a transformation
   ══════════════════════════════════════════════════════════════════════════ */
const FTeval = (function () {
  const S = FTscene, W = S.W, H = S.H;
  const MAGS = {
    rot: [0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90],
    scale: [1.0, 1.05, 1.1, 1.2, 1.3, 1.45, 1.6, 1.8, 2.0, 2.3, 2.6],
    noise: [0, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20],
    blur: [0, 0.4, 0.7, 1.0, 1.4, 1.8, 2.3, 2.8, 3.4, 4.0, 4.8]
  };
  const LABEL = { rot: v => v + "°", scale: v => "×" + v.toFixed(2),
    noise: v => v.toFixed(3), blur: v => "σ " + v.toFixed(1) };
  function transform(kind, m) {
    const c = (W - 1) / 2;
    let Hm = VZ.eye(3);
    if (kind === "rot") Hm = VZ.mul(VZ.mul(VZ.T2(c, c), VZ.R2(VZ.rad(m), 0, 0)), VZ.T2(-c, -c));
    if (kind === "scale") Hm = VZ.mul(VZ.mul(VZ.T2(c, c), VZ.S2(m, 0, 0, 0)), VZ.T2(-c, -c));
    const Hi = VZ.inv3(Hm);
    let src = S.A;
    if (kind === "blur" && m > 0.05) src = VZ.sep2(S.A, VZ.gauss1(m), "mirror");
    let J = VZ.zeros2(H, W);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const p = VZ.applyH(Hi, [x, y]);
      J[y][x] = FT.bilerpC(src, p[1], p[0]);
    }
    if (kind === "noise" && m > 0) J = FT.addNoise(J, m, 31);
    return { J: J, Hm: Hm };
  }
  /* three detectors on the same image, each returning at most n points */
  function detect(I, which, n, tag) {
    if (which === "dog") {
      /* the cache tag must be DETERMINISTIC and distinct per image: a random tag
         defeats the cache and grows it without bound, which is a slow leak rather
         than a visible bug and therefore worth being explicit about. */
      const st = FTss.stack(I, 3, 0.9, 2, "ev|" + (tag || "ref"));
      const c = FTss.extrema(st.levels, "dog");
      c.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
      const keep = [];
      for (const p of c) {
        if (keep.length >= n) break;
        if (keep.every(q => Math.hypot(q.x - p.x, q.y - p.y) > 4)) keep.push({ x: p.x, y: p.y, s: p.sigma });
      }
      return keep;
    }
    const T = FT.structure(I, 1.0, 2.0, "mirror"), R = VZ.zeros2(H, W);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const a = T.A11[y][x], b = T.A12[y][x], c = T.A22[y][x];
      R[y][x] = which === "shi" ? FT.eig2(a, b, c)[0] : (a * c - b * b) - 0.04 * (a + c) * (a + c);
    }
    const cand = [];
    for (let y = 3; y < H - 3; y++) for (let x = 3; x < W - 3; x++) {
      const v = R[y][x];
      if (v <= 0) continue;
      let ok = true;
      for (let b = -1; b <= 1 && ok; b++) for (let a = -1; a <= 1; a++)
        if ((a || b) && R[y + b][x + a] > v) { ok = false; break; }
      if (ok) cand.push({ x: x, y: y, v: v, s: 2 });
    }
    cand.sort((a, b) => b.v - a.v);
    const keep = [];
    for (const p of cand) {
      if (keep.length >= n) break;
      if (keep.every(q => Math.hypot(q.x - p.x, q.y - p.y) > 4)) keep.push(p);
    }
    return keep;
  }
  const inside = p => p[0] >= 4 && p[1] >= 4 && p[0] < W - 4 && p[1] < H - 4;
  /* repeatability, with the denominator restricted to the overlap */
  function repeatability(which, kind, m, eps, n) {
    const T = transform(kind, m);
    const K0 = detect(S.A, which, n, "ref");
    const K1 = detect(T.J, which, n, kind + "|" + m);
    let denom = 0, rep = 0;
    const marks = [];
    for (const k of K0) {
      const p = VZ.applyH(T.Hm, [k.x, k.y]);
      if (!inside(p)) continue;
      denom++;
      const hit = K1.some(q => Math.hypot(q.x - p[0], q.y - p[1]) < eps);
      if (hit) rep++;
      marks.push({ p: p, hit: hit });
    }
    return { K0, K1, T, denom, rep, marks, r: denom ? rep / denom : 0 };
  }
  const CACHE = new Map();
  function curve(which, kind, eps, n) {
    const key = [which, kind, eps, n].join("|");
    if (CACHE.has(key)) return CACHE.get(key);
    const rows = MAGS[kind].map(m => ({ m: m, r: repeatability(which, kind, m, eps, n).r }));
    CACHE.set(key, rows);
    return rows;
  }
  return { MAGS, LABEL, transform, detect, repeatability, curve, W, H };
})();

(function () {
  const svg = d3.select("#eval-svg");
  if (svg.empty()) return;
  const EV = FTeval, S = FTscene;
  const eT = document.getElementById("ev-t");
  const eM = document.getElementById("ev-m"), eMv = document.getElementById("ev-mv");
  const eE = document.getElementById("ev-e"), eEv = document.getElementById("ev-ev");
  const eN = document.getElementById("ev-n"), eNv = document.getElementById("ev-nv");
  const out = document.getElementById("eval-readout");
  const DETS = [["harris", "Harris", VC.a2], ["shi", "Shi–Tomasi", VC.accent], ["dog", "DoG blobs", VC.violet]];

  function get() {
    const kind = eT.value, mi = +eM.value, eps = +eE.value, n = +eN.value;
    const m = EV.MAGS[kind][mi];
    return { kind, mi, m, eps, n,
      per: DETS.map(d => ({ d: d, R: EV.repeatability(d[0], kind, m, eps, n),
        curve: EV.curve(d[0], kind, eps, n) })) };
  }

  function draw() {
    const g0 = get();
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const cw = 2.1, IW = S.W * cw;
    const main = g0.per[0];

    const B1 = FT.box(g, 14, 44, IW, IW, "reference · Harris");
    FT.cells(B1, 0, 0, cw, S.W, S.H, (x, y) => FT.grey(S.A[y][x]));
    main.R.K0.forEach(k => B1.append("circle").attr("cx", (k.x + 0.5) * cw).attr("cy", (k.y + 0.5) * cw)
      .attr("r", 2.6).attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.2));
    const B2 = FT.box(g, 14 + IW + 20, 44, IW, IW, "transformed · " + EV.LABEL[g0.kind](g0.m));
    FT.cells(B2, 0, 0, cw, S.W, S.H, (x, y) => FT.grey(main.R.T.J[y][x]));
    main.R.K1.forEach(k => B2.append("circle").attr("cx", (k.x + 0.5) * cw).attr("cy", (k.y + 0.5) * cw)
      .attr("r", 2.2).attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 0.9));
    main.R.marks.forEach(mk => B2.append("circle").attr("cx", (mk.p[0] + 0.5) * cw).attr("cy", (mk.p[1] + 0.5) * cw)
      .attr("r", g0.eps * cw).attr("fill", "none").attr("stroke", mk.hit ? VC.good : VC.bad)
      .attr("stroke-width", 1.3));
    B2.append("text").attr("x", 0).attr("y", IW + 13).attr("font-size", 10).attr("fill", VC.muted)
      .text("circles of radius ε around each mapped reference point");

    /* ---- middle: the three curves ----------------------------------------- */
    const MX = 14 + 2 * IW + 44, CW2 = 214, CH2 = 190;
    const gc = g.append("g").attr("transform", `translate(${MX},52)`);
    const mags = EV.MAGS[g0.kind];
    const x = d3.scaleLinear().domain([mags[0], mags[mags.length - 1]]).range([0, CW2]);
    const y = d3.scaleLinear().domain([0, 1.02]).range([CH2, 0]);
    VZ.gridY(gc, y, CW2, 5);
    VZ.axisB(gc, x, CH2, 5, { rot: "rotation (°)", scale: "scale factor",
      noise: "noise σ", blur: "blur σ" }[g0.kind]);
    gc.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5, ".0%"));
    gc.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10.5).attr("fill", VC.ink)
      .text("repeatability at ε = " + g0.eps.toFixed(2) + " px");
    g0.per.forEach(p => {
      gc.append("path").attr("d", d3.line().x(d => x(d.m)).y(d => y(d.r))(p.curve))
        .attr("fill", "none").attr("stroke", p.d[2]).attr("stroke-width", 2);
      p.curve.forEach(d => gc.append("circle").attr("cx", x(d.m)).attr("cy", y(d.r)).attr("r", 2.2)
        .attr("fill", p.d[2]));
    });
    gc.append("line").attr("x1", x(g0.m)).attr("x2", x(g0.m)).attr("y1", 0).attr("y2", CH2)
      .attr("stroke", VC.ink).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.6);
    VZ.legend(gc, DETS.map(d => ({ color: d[2], label: d[1] })), 6, 12, { gap: 14 });

    /* ---- right: the numbers ----------------------------------------------- */
    const RX = MX + CW2 + 46;
    const put = FT.kv(g, RX, 60, { keyW: 168, lead: 15.5 });
    g0.per.forEach(p => {
      put(p.d[1], "", p.d[2], true);
      put("  detections, both images", p.R.K0.length + " / " + p.R.K1.length, VC.muted);
      put("  in the overlap", p.R.denom, VC.ink);
      put("  repeated within ε", p.R.rep, VC.good);
      put("  repeatability", VZ.fmt(100 * p.R.r, 1) + "%", p.R.r > 0.5 ? VC.good : VC.bad, true);
    });
  }

  function say() {
    const g0 = get();
    const best = g0.per.reduce((a, b) => b.R.r > a.R.r ? b : a);
    const at0 = g0.per.map(p => p.curve[0].r);
    out.innerHTML =
      `At ${EV.LABEL[g0.kind](g0.m)} with <span class="keep">ε</span> = ${g0.eps.toFixed(2)} px: `
      + g0.per.map(p => `${p.d[1]} <b>${VZ.fmt(100 * p.R.r, 1)}%</b> (${p.R.rep} of ${p.R.denom})`).join(", ")
      + `. Best here: <b>${best.d[1]}</b>.<br>`
      + `Three things about that number are worth stating. The denominator counts only reference detections `
      + `whose mapped position lands <i>inside</i> the transformed image — otherwise a detector is punished for `
      + `points that left the frame. Every detector is capped at <b>${g0.n}</b> points, because repeatability `
      + `rises trivially with detection count and an uncapped comparison measures density rather than quality. `
      + `And <span class="keep">ε</span> is a free parameter: raise it and every curve rises together, which is why `
      + `a repeatability figure without its <span class="keep">ε</span> is not a measurement. `
      + `At zero magnitude the three detectors score ${at0.map(v => VZ.fmt(100 * v, 0) + "%").join(", ")} — `
      + `the identity transform is a resample by an identity map, so nothing moves and every point is repeated. `
      + `That is the correct ceiling, and it is what makes the rest of each curve readable as a loss rather than `
      + `as an unexplained level.`;
  }
  function all() {
    eMv.textContent = EV.LABEL[eT.value](EV.MAGS[eT.value][+eM.value]);
    eEv.textContent = (+eE.value).toFixed(2); eNv.textContent = eN.value;
    draw(); say();
  }
  eT.onchange = all; eM.oninput = all; eE.oninput = all; eN.oninput = all;
  all();
})();
