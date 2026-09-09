/* stereo-sfm.viz.js — the twenty-two visualizations on vision/stereo-sfm.html.
   Loaded after ../data.js → ../notes.js → vision-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

   All linear algebra that vision-viz.js already owns goes through VZ (mul, mv, T,
   inv3, invN, skew, rodrigues, K, camera, lookAt, view3, jacobiEig, …). Nothing
   here forks those. The ONE thing the shared library does not have is a singular
   value decomposition, and this page cannot be written without one — the eight-point
   algorithm, the rank-2 enforcement, linear triangulation, factorization and the
   epipole extraction are all "smallest singular vector" steps. So SS.svd below is a
   page-local one-sided Jacobi SVD. One-sided Jacobi is used rather than the
   textbook "eigen-decompose AᵀA" shortcut on purpose: forming AᵀA squares the
   condition number, and §08's whole point is to MEASURE a condition number of about
   1.4e5, which squaring would push past what a double can represent honestly.

     1  #epi-svg      two cameras, a draggable baseline, the epipolar plane sweeping
     2  #pencil-svg   the pencil of epipolar lines through the epipole, and the
                      focus of expansion under forward motion
     3  #emat-svg     E = [t]×R built factor by factor, with the constraint measured
     4  #fmat-svg     F transferring a point to a line, and the 1-D search on it
     5  #rank-svg     the singular values of E and F, and the degree-of-freedom count
     6  #eight-svg    the eight-point algorithm, normalised against not, with both
                      condition numbers and both errors measured over many trials
     7  #enforce-svg  rank-2 enforcement by SVD, and what it does to the pencil
     8  #degen-svg    degenerate configurations: the null space of A opens up
     9  #decomp-svg   the four (R, t) candidates from E, three killed by cheirality
    10  #tri-svg      mid-point vs linear vs optimal triangulation, both errors
    11  #rect-svg     rectification turning epipolar lines into scanlines
    12  #depth-svg    Z = fB/d, and depth uncertainty growing as Z², measured
    13  #cost-svg     matching costs under gain and bias changes
    14  #dsi-svg      the cost volume, its winner-take-all disparity, and two failures
    15  #agg-svg      aggregation window size: the bias–variance trade in stereo
    16  #sgm-svg      scanline optimisation and semi-global aggregation
    17  #sub-svg      sub-pixel parabola fitting, its bias, and the variance formula
    18  #fail-svg     the classic stereo failure modes, side by side
    19  #fact-svg     affine factorization: the rank-3 measurement matrix
    20  #ba-svg       bundle adjustment converging, reprojection error falling
    21  #spar-svg     the Jacobian sparsity pattern and the Schur complement
    22  #drift-svg    drift accumulating, and loop closure correcting it

   Every number these print is recomputed from the data they draw.

   Three page-local modules, in this order:
     SS  the geometry kit — a one-sided Jacobi SVD plus the two-view machinery that
         needs it (eight-point, rank-2 projection, epipoles, the three triangulators)
     ST  the synthetic rectified stereo pair used by §19-§24, RENDERED from three
         surfaces into two views rather than warped, so occlusion is real
     BA  the small bundle-adjustment problem used by §27 and §28                     */

/* ══════════ page-local helpers ══════════════════════════════════════════ */
const SS = (function () {

  /* ---- one-sided Jacobi SVD ---------------------------------------------
     A (m × n, m ≥ n) → {U (m × n), s (n, descending), V (n × n)}.
     Columns of A are orthogonalised in place by plane rotations; the column
     norms that survive are the singular values and the accumulated rotations
     are V. Accurate to working precision even when A is badly scaled, which
     is exactly the case §08 needs to report honestly.                       */
  function svd(Ain, sweeps) {
    const m = Ain.length, n = Ain[0].length;
    const A = Ain.map(r => r.slice());
    const V = VZ.eye(n);
    const NS = sweeps || 60;
    for (let sweep = 0; sweep < NS; sweep++) {
      let off = 0;
      for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
        let a = 0, b = 0, g = 0;
        for (let i = 0; i < m; i++) { a += A[i][p] * A[i][p]; b += A[i][q] * A[i][q]; g += A[i][p] * A[i][q]; }
        if (g === 0) continue;
        const scale = Math.sqrt(a * b);
        if (scale === 0 || Math.abs(g) < 1e-16 * scale) continue;
        off += (g * g) / (a * b);
        const z = (b - a) / (2 * g);
        const t = (z >= 0 ? 1 : -1) / (Math.abs(z) + Math.sqrt(1 + z * z));
        const c = 1 / Math.sqrt(1 + t * t), s = c * t;
        for (let i = 0; i < m; i++) { const ap = A[i][p], aq = A[i][q]; A[i][p] = c * ap - s * aq; A[i][q] = s * ap + c * aq; }
        for (let i = 0; i < n; i++) { const vp = V[i][p], vq = V[i][q]; V[i][p] = c * vp - s * vq; V[i][q] = s * vp + c * vq; }
      }
      if (off < 1e-30) break;
    }
    /* column norms are the singular values; sort descending */
    const cols = [];
    for (let j = 0; j < n; j++) {
      let nn = 0; for (let i = 0; i < m; i++) nn += A[i][j] * A[i][j];
      cols.push({ j: j, s: Math.sqrt(nn) });
    }
    cols.sort((p, q) => q.s - p.s);
    const s = cols.map(c => c.s);
    const U = [], Vs = [];
    for (let i = 0; i < m; i++) U.push(new Array(n).fill(0));
    for (let i = 0; i < n; i++) Vs.push(new Array(n).fill(0));
    const tol = 1e-13 * (s[0] || 1);
    for (let k = 0; k < n; k++) {
      const j = cols[k].j;
      for (let i = 0; i < n; i++) Vs[i][k] = V[i][j];
      if (s[k] > tol) for (let i = 0; i < m; i++) U[i][k] = A[i][j] / s[k];
    }
    /* complete U for any exactly-null directions, by Gram–Schmidt against the
       standard basis. Only ever needed for a square matrix of exact rank < n. */
    for (let k = 0; k < n; k++) {
      if (s[k] > tol) continue;
      for (let cand = 0; cand < m; cand++) {
        const v = new Array(m).fill(0); v[cand] = 1;
        for (let k2 = 0; k2 < n; k2++) {
          if (k2 === k) continue;
          let d = 0; for (let i = 0; i < m; i++) d += U[i][k2] * v[i];
          for (let i = 0; i < m; i++) v[i] -= d * U[i][k2];
        }
        let nn = 0; for (let i = 0; i < m; i++) nn += v[i] * v[i];
        if (nn > 1e-12) { nn = Math.sqrt(nn); for (let i = 0; i < m; i++) U[i][k] = v[i] / nn; break; }
      }
    }
    return { U: U, s: s, V: Vs };
  }

  const col = (M, j) => M.map(r => r[j]);
  /* the right and left null directions — the two epipoles, for an F */
  const rightNull = A => col(svd(A).V, A[0].length - 1);
  const leftNull = A => col(svd(VZ.T(A)).V, A.length - 1);

  /* nearest rank-2 matrix in Frobenius norm: zero the smallest singular value.
     Optimality is Eckart–Young, proved on the linear-algebra SVD page. */
  function rank2(F) {
    const { U, s, V } = svd(F);
    const S = [[s[0], 0, 0], [0, s[1], 0], [0, 0, 0]];
    return { F: VZ.mul(VZ.mul(U, S), VZ.T(V)), s: s };
  }

  /* ---- small geometry ---------------------------------------------------- */
  const skew = VZ.skew;
  const hom = p => [p[0], p[1], 1];
  function normLine(l) { const n = Math.hypot(l[0], l[1]); return n < 1e-15 ? l.slice() : l.map(v => v / n); }
  /* signed point-line distance in pixels, for a line normalised as above */
  const ptLine = (l, p) => (l[0] * p[0] + l[1] * p[1] + l[2]) / Math.hypot(l[0], l[1]);

  /* clip the line l to the rectangle [0,W] × [0,H]; null if it misses */
  function clipLine(l, W, H) {
    const pts = [];
    const push = (x, y) => { if (x >= -1e-7 && x <= W + 1e-7 && y >= -1e-7 && y <= H + 1e-7) pts.push([x, y]); };
    if (Math.abs(l[1]) > 1e-12) { push(0, -(l[2]) / l[1]); push(W, -(l[0] * W + l[2]) / l[1]); }
    if (Math.abs(l[0]) > 1e-12) { push(-(l[2]) / l[0], 0); push(-(l[1] * H + l[2]) / l[0], H); }
    if (pts.length < 2) return null;
    /* keep the two furthest apart */
    let best = null, bd = -1;
    for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
      const d = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
      if (d > bd) { bd = d; best = [pts[i], pts[j]]; }
    }
    return bd < 1e-9 ? null : best;
  }

  /* ---- the running two-view rig ------------------------------------------
     Camera 0 at the origin in canonical orientation; camera 1 at centre c with
     orientation R₁. Returns everything the figures keep asking for.           */
  function rig(o) {
    const K0 = o.K0 || VZ.K({ f: 500, cx: 320, cy: 240 });
    const K1 = o.K1 || VZ.K({ f: 520, cx: 316, cy: 244 });
    const R1 = o.R1 || VZ.eye(3);
    const c1 = o.c1 || [1, 0, 0];
    const t = VZ.neg(VZ.mv(R1, c1));
    const E = VZ.mul(skew(t), R1);
    const F = VZ.mul(VZ.mul(VZ.T(VZ.inv3(K1)), E), VZ.inv3(K0));
    const cam0 = VZ.camera({ K: K0, R: VZ.eye(3), t: [0, 0, 0] });
    const cam1 = VZ.camera({ K: K1, R: R1, t: t });
    return { K0: K0, K1: K1, R: R1, t: t, c1: c1, E: E, F: F, cam0: cam0, cam1: cam1 };
  }

  /* the two epipoles of an F, in pixels; w ≈ 0 means "at infinity" */
  function epipoles(F) {
    const a = rightNull(F), b = leftNull(F);
    const fin = v => Math.abs(v[2]) > 1e-12 ? [v[0] / v[2], v[1] / v[2]] : null;
    return { e0: fin(a), e1: fin(b), e0h: a, e1h: b };
  }

  /* ---- normalisation, exactly as §08 states it -------------------------- */
  function normaliseT(pts) {
    let mx = 0, my = 0;
    pts.forEach(p => { mx += p[0]; my += p[1]; });
    mx /= pts.length; my /= pts.length;
    let d = 0;
    pts.forEach(p => { d += Math.hypot(p[0] - mx, p[1] - my); });
    d /= pts.length;
    const s = d > 1e-12 ? Math.SQRT2 / d : 1;
    return [[s, 0, -s * mx], [0, s, -s * my], [0, 0, 1]];
  }

  /* the eight-point algorithm. norm = Hartley pre-conditioning on/off,
     enf = rank-2 enforcement on/off. Returns F, the design-matrix singular
     values, and the condition number σ₁/σ₈ (the last direction is the answer). */
  function eightPoint(p0, p1, norm, enf) {
    let T0 = VZ.eye(3), T1 = VZ.eye(3), a = p0, b = p1;
    if (norm) {
      T0 = normaliseT(p0); T1 = normaliseT(p1);
      a = p0.map(p => VZ.applyH(T0, p));
      b = p1.map(p => VZ.applyH(T1, p));
    }
    const A = a.map((p, i) => {
      const [u, v] = p, [up, vp] = b[i];
      return [up * u, up * v, up, vp * u, vp * v, vp, u, v, 1];
    });
    const { s, V } = svd(A);
    const f = col(V, 8);
    let F = [[f[0], f[1], f[2]], [f[3], f[4], f[5]], [f[6], f[7], f[8]]];
    let sv = null;
    if (enf !== false) { const r = rank2(F); F = r.F; sv = r.s; }
    F = VZ.mul(VZ.mul(VZ.T(T1), F), T0);
    /* scale so the numbers are comparable between runs */
    let mx = 0; F.forEach(r => r.forEach(v => { mx = Math.max(mx, Math.abs(v)); }));
    if (mx > 0) F = F.map(r => r.map(v => v / mx));
    return { F: F, designS: s, cond: s[7] > 0 ? s[0] / s[7] : Infinity, fSV: sv };
  }

  /* symmetric epipolar distance in pixels — the error every F figure reports */
  function symEpi(F, p0, p1) {
    let acc = 0;
    for (let i = 0; i < p0.length; i++) {
      const x0 = hom(p0[i]), x1 = hom(p1[i]);
      const l1 = VZ.mv(F, x0), l0 = VZ.mv(VZ.T(F), x1);
      const r = VZ.dot(x1, l1);
      const d1 = Math.abs(r) / Math.hypot(l1[0], l1[1]);
      const d0 = Math.abs(r) / Math.hypot(l0[0], l0[1]);
      acc += Math.sqrt((d0 * d0 + d1 * d1) / 2);
    }
    return acc / p0.length;
  }

  /* ---- triangulation ------------------------------------------------------ */
  /* mid-point: the world point closest to both rays, in the least-squares sense */
  function triMid(camA, camB, uA, uB) {
    const dirA = VZ.unit(VZ.mv(VZ.T(camA.R), VZ.mv(VZ.inv3(camA.K), hom(uA))));
    const dirB = VZ.unit(VZ.mv(VZ.T(camB.R), VZ.mv(VZ.inv3(camB.K), hom(uB))));
    let A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
    [[camA.C, dirA], [camB.C, dirB]].forEach(([c, v]) => {
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        const M = (i === j ? 1 : 0) - v[i] * v[j];
        A[i][j] += M; b[i] += M * c[j];
      }
    });
    const Ai = VZ.inv3(A);
    return Ai ? VZ.mv(Ai, b) : null;
  }
  /* linear DLT: stack the two "cross product with the projected point" rows per
     view and take the smallest right singular vector of the 4 × 4 system */
  function triDLT(camA, camB, uA, uB) {
    const rows = [];
    [[camA, uA], [camB, uB]].forEach(([c, u]) => {
      const P = c.P;
      rows.push(P[2].map((v, k) => u[0] * v - P[0][k]));
      rows.push(P[2].map((v, k) => u[1] * v - P[1][k]));
    });
    const X = col(svd(rows).V, 3);
    return Math.abs(X[3]) < 1e-14 ? null : [X[0] / X[3], X[1] / X[3], X[2] / X[3]];
  }
  /* the optimal two-view correction. The objective is the one the closed-form
     method solves — minimise d(x₀,l₀(θ))² + d(x₁,l₁(θ))² over the pencil of
     epipolar planes — but it is minimised here by a dense sweep of the pencil
     followed by a golden-section refinement, rather than by rooting the
     degree-six polynomial. Same optimum, no polynomial solver on the page.    */
  function triOptimal(camA, camB, uA, uB, F) {
    const ep = epipoles(F);
    if (!ep.e0 || !ep.e1) return triDLT(camA, camB, uA, uB);
    const cost = th => {
      /* the epipolar line in image 0 through e₀ at angle θ */
      const d0 = [Math.cos(th), Math.sin(th)];
      const l0 = normLine([-d0[1], d0[0], d0[1] * ep.e0[0] - d0[0] * ep.e0[1]]);
      /* its partner in image 1: transfer a point of l0 through F */
      const q = [ep.e0[0] + 40 * d0[0], ep.e0[1] + 40 * d0[1]];
      const l1 = normLine(VZ.mv(F, hom(q)));
      const a = ptLine(l0, uA), b = ptLine(l1, uB);
      return { c: a * a + b * b, l0: l0, l1: l1 };
    };
    let bt = 0, bc = Infinity;
    const N = 720;
    for (let i = 0; i < N; i++) { const th = Math.PI * i / N; const c = cost(th).c; if (c < bc) { bc = c; bt = th; } }
    let lo = bt - Math.PI / N, hi = bt + Math.PI / N;
    const gr = (Math.sqrt(5) - 1) / 2;
    for (let it = 0; it < 80; it++) {
      const m1 = hi - gr * (hi - lo), m2 = lo + gr * (hi - lo);
      if (cost(m1).c < cost(m2).c) hi = m2; else lo = m1;
    }
    const best = cost((lo + hi) / 2);
    const foot = (l, p) => { const d = ptLine(l, p); return [p[0] - d * l[0], p[1] - d * l[1]]; };
    return triDLT(camA, camB, foot(best.l0, uA), foot(best.l1, uB));
  }

  /* ---- misc ---------------------------------------------------------------- */
  const grey = v => { const c = VZ.clamp(Math.round(255 * v), 0, 255); return `rgb(${c},${c},${c})`; };
  function heat(t) {
    const u = VZ.clamp(t, 0, 1);
    return u < 0.5 ? d3.interpolateRgb(VC.accent, VC.panel2)(u * 2)
      : d3.interpolateRgb(VC.panel2, VC.a2)((u - 0.5) * 2);
  }
  /* a seeded, mildly band-limited 1-D texture — the scanline every dense-stereo
     figure matches against. Amplitude and bandwidth are arguments so that a
     figure can ask for a textureless or a repeating one. */
  function scanline(n, seed, opt) {
    const o = Object.assign({ kmin: 3, kmax: 16, amp: 40, mean: 128 }, opt || {});
    const r = VZ.rng(seed), a = new Float64Array(n);
    for (let k = o.kmin; k <= o.kmax; k++) {
      const ph = r() * 2 * Math.PI, w = VZ.randn(r);
      for (let i = 0; i < n; i++) a[i] += w * Math.sin(2 * Math.PI * k * i / n + ph);
    }
    let s = 0, m = 0;
    for (let i = 0; i < n; i++) m += a[i]; m /= n;
    for (let i = 0; i < n; i++) s += (a[i] - m) * (a[i] - m); s = Math.sqrt(s / n) || 1;
    for (let i = 0; i < n; i++) a[i] = o.mean + o.amp * (a[i] - m) / s;
    return a;
  }
  const lerpAt = (a, t) => {
    const i = VZ.clamp(Math.floor(t), 0, a.length - 2), f = t - i;
    return a[i] * (1 - f) + a[i + 1] * f;
  };

  return {
    svd: svd, col: col, rightNull: rightNull, leftNull: leftNull, rank2: rank2,
    hom: hom, normLine: normLine, ptLine: ptLine, clipLine: clipLine,
    rig: rig, epipoles: epipoles, normaliseT: normaliseT, eightPoint: eightPoint,
    symEpi: symEpi, triMid: triMid, triDLT: triDLT, triOptimal: triOptimal,
    grey: grey, heat: heat, scanline: scanline, lerpAt: lerpAt
  };
})();

/* ══════════ 1 · #epi-svg — the epipolar plane, swept ═════════════════════ */
(function () {
  const svg = d3.select("#epi-svg");
  if (svg.empty()) return;
  const W = 760, H = 486, IW = 640, IH = 480;
  const ro = d3.select("#epi-readout");
  const el = id => document.getElementById(id);

  /* image-panel geometry: 640 × 480 pixels drawn at scale 0.40625 */
  const PS = 0.40625, PW = IW * PS, PH = IH * PS;
  const P0 = { x: 400, y: 34 }, P1 = { x: 400, y: 258 };

  const K0 = VZ.K({ f: 500, cx: 320, cy: 240 });
  const K1 = VZ.K({ f: 520, cx: 316, cy: 244 });

  let drag = null;                       // pixel the reader dragged to, in image 0

  function build() {
    const B = +el("ep-b").value, verg = VZ.rad(+el("ep-v").value);
    let Z = +el("ep-z").value, X = +el("ep-x").value;
    /* display world: X right, Y forward (depth), Z up — view3's convention */
    const c0 = [-B / 2, 0, 0], c1 = [B / 2, 0, 0];
    const d0 = [Math.sin(verg / 2), Math.cos(verg / 2), 0];
    const d1 = [-Math.sin(verg / 2), Math.cos(verg / 2), 0];
    const R0 = VZ.lookAt(c0, VZ.add(c0, d0), [0, 0, 1]);
    const R1 = VZ.lookAt(c1, VZ.add(c1, d1), [0, 0, 1]);
    const cam0 = VZ.camera({ K: K0, R: R0, t: VZ.neg(VZ.mv(R0, c0)) });
    const cam1 = VZ.camera({ K: K1, R: R1, t: VZ.neg(VZ.mv(R1, c1)) });
    /* relative pose of 1 with respect to 0 */
    const Rrel = VZ.mul(R1, VZ.T(R0));
    const trel = VZ.sub(cam1.t, VZ.mv(Rrel, cam0.t));
    const E = VZ.mul(VZ.skew(trel), Rrel);
    let F = VZ.mul(VZ.mul(VZ.T(VZ.inv3(K1)), E), VZ.inv3(K0));
    let mx = 0; F.forEach(r => r.forEach(v => { mx = Math.max(mx, Math.abs(v)); }));
    F = F.map(r => r.map(v => v / mx));

    /* the scene point. If the reader dragged in image 0, back-project that pixel
       to the current depth along camera 0's optical axis; otherwise use X and a
       fixed height so the sliders alone determine it. */
    let p;
    if (drag) {
      const ray = VZ.mv(VZ.T(R0), VZ.mv(VZ.inv3(K0), SS.hom(drag)));
      const along = VZ.dot(ray, d0);
      p = VZ.add(c0, VZ.scale(ray, Z / Math.max(along, 1e-6)));
      X = p[0];
    } else {
      p = [X, Z * Math.cos(verg / 2) - B / 2 * Math.sin(verg / 2) * 0, 0.12];
      p[1] = Z;                          // depth measured forward from the rig
    }
    return { B: B, verg: verg, Z: Z, X: X, c0: c0, c1: c1, R0: R0, R1: R1, cam0: cam0, cam1: cam1, F: F, p: p, E: E, trel: trel };
  }

  /* pixel (u,v) of camera j → a world point on its image plane, at display scale */
  function planePt(cam, R, c, u, v, sc) {
    const ray = VZ.mv(VZ.T(R), VZ.mv(VZ.inv3(cam.K), [u, v, 1]));
    return VZ.add(c, VZ.scale(ray, sc));
  }

  function draw() {
    const S = build();
    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── 3-D panel ───────────────────────────────────────────────────────── */
    const gp = VZ.panelBox(g, 12, 30, 366, 442, "the rig in 3-D  ·  epipolar plane hinged on the baseline");
    const view = VZ.view3({
      target: [0, Math.min(S.Z, 8) * 0.5, 0], yaw: VZ.rad(-108), pitch: VZ.rad(20),
      dist: 9 + 0.55 * S.Z, f: 330, cx: 183, cy: 232
    });
    const pj = q => view.project(q);
    const inP = q => q.x > 2 && q.x < 364 && q.y > 2 && q.y < 440;

    VZ.gridPlane(view, 12, 0.8, -1.1).forEach(sgm => {
      if (!inP(sgm[0]) || !inP(sgm[1])) return;
      gp.append("line").attr("x1", sgm[0].x).attr("y1", sgm[0].y).attr("x2", sgm[1].x).attr("y2", sgm[1].y)
        .attr("stroke", VC.grid).attr("stroke-width", 0.7);
    });

    const sc = 0.55 + 0.05 * S.Z;                      // display size of the image planes
    /* the epipolar plane, as the quad c0 – c1 – (rays to p, extended) */
    if (el("ep-plane").checked) {
      const k = 1.32;
      const q = [S.c0, S.c1, VZ.add(S.c1, VZ.scale(VZ.sub(S.p, S.c1), k)), VZ.add(S.c0, VZ.scale(VZ.sub(S.p, S.c0), k))]
        .map(pj).map(o => [o.x, o.y]);
      VZ.poly(gp, q, { fill: VC.violet, fillOp: 0.13, stroke: VC.violet, w: 1, op: 0.5 });
    }

    /* the two cameras */
    [[S.cam0, S.R0, S.c0, VC.accent, "cam 0"], [S.cam1, S.R1, S.c1, VC.a2, "cam 1"]].forEach(([cam, R, c, colr, lab]) => {
      const corners = [[0, 0], [IW, 0], [IW, IH], [0, IH]].map(u => planePt(cam, R, c, u[0], u[1], sc));
      const cs = corners.map(pj).map(o => [o.x, o.y]);
      const oc = pj(c);
      VZ.poly(gp, cs, { fill: colr, fillOp: 0.07, stroke: colr, w: 1.2, op: 0.85 });
      cs.forEach(q => gp.append("line").attr("x1", oc.x).attr("y1", oc.y).attr("x2", q[0]).attr("y2", q[1])
        .attr("stroke", colr).attr("stroke-width", 0.7).attr("stroke-opacity", 0.55));
      gp.append("circle").attr("cx", oc.x).attr("cy", oc.y).attr("r", 3.4).attr("fill", colr);
      gp.append("text").attr("x", oc.x - 8).attr("y", oc.y + 18).attr("font-size", 10).attr("fill", colr).text(lab);
    });

    /* the baseline */
    const b0 = pj(S.c0), b1 = pj(S.c1);
    gp.append("line").attr("x1", b0.x).attr("y1", b0.y).attr("x2", b1.x).attr("y2", b1.y)
      .attr("stroke", VC.good).attr("stroke-width", 2.2);
    gp.append("text").attr("x", (b0.x + b1.x) / 2 - 20).attr("y", (b0.y + b1.y) / 2 + 16)
      .attr("font-size", 10).attr("fill", VC.good).text("baseline");

    /* the two rays and the scene point */
    const pp = pj(S.p);
    [[S.c0, VC.accent], [S.c1, VC.a2]].forEach(([c, colr]) => {
      const o = pj(c);
      gp.append("line").attr("x1", o.x).attr("y1", o.y).attr("x2", pp.x).attr("y2", pp.y)
        .attr("stroke", colr).attr("stroke-width", 1.3).attr("stroke-dasharray", "3 3");
    });
    gp.append("circle").attr("cx", pp.x).attr("cy", pp.y).attr("r", 4.4).attr("fill", VC.violet);
    gp.append("text").attr("x", pp.x + 7).attr("y", pp.y - 5).attr("font-size", 10.5).attr("fill", VC.violet).text("p");

    /* the epipolar lines, drawn on the image planes in 3-D */
    const x0 = S.cam0.project(S.p), x1 = S.cam1.project(S.p);
    const l1 = VZ.mv(S.F, SS.hom([x0[0], x0[1]]));
    const l0 = VZ.mv(VZ.T(S.F), SS.hom([x1[0], x1[1]]));
    [[l0, S.cam0, S.R0, S.c0, VC.accent], [l1, S.cam1, S.R1, S.c1, VC.a2]].forEach(([l, cam, R, c, colr]) => {
      const seg = SS.clipLine(l, IW, IH);
      if (!seg) return;
      const w = seg.map(u => pj(planePt(cam, R, c, u[0], u[1], sc)));
      gp.append("line").attr("x1", w[0].x).attr("y1", w[0].y).attr("x2", w[1].x).attr("y2", w[1].y)
        .attr("stroke", VC.violet).attr("stroke-width", 2);
    });

    /* ── the two image panels ───────────────────────────────────────────── */
    const ep = SS.epipoles(S.F);
    function imgPanel(o, title, colr, pt, line, isFirst) {
      const gg = VZ.panelBox(g, o.x, o.y, PW, PH, title, { fill: VC.bg });
      const seg = SS.clipLine(line, IW, IH);
      if (seg) gg.append("line").attr("x1", seg[0][0] * PS).attr("y1", seg[0][1] * PS)
        .attr("x2", seg[1][0] * PS).attr("y2", seg[1][1] * PS)
        .attr("stroke", VC.violet).attr("stroke-width", 1.8);
      /* the 1-D search segment: where the point would land for Z from 2 to 12 m */
      if (!isFirst && el("ep-band").checked) {
        const ray = VZ.mv(VZ.T(S.R0), VZ.mv(VZ.inv3(K0), SS.hom([pt.x0[0], pt.x0[1]])));
        const along = VZ.dot(ray, VZ.mv(VZ.T(S.R0), [0, 0, 1]));
        const ends = [2, 12].map(z => {
          const q = VZ.add(S.c0, VZ.scale(ray, z / Math.max(along, 1e-6)));
          const u = S.cam1.project(q); return [u[0] * PS, u[1] * PS];
        });
        gg.append("line").attr("x1", ends[0][0]).attr("y1", ends[0][1])
          .attr("x2", ends[1][0]).attr("y2", ends[1][1])
          .attr("stroke", VC.good).attr("stroke-width", 4.5).attr("stroke-opacity", 0.42)
          .attr("stroke-linecap", "round");
      }
      const px = pt.u[0] * PS, py = pt.u[1] * PS;
      if (px > -20 && px < PW + 20 && py > -20 && py < PH + 20) {
        gg.append("circle").attr("cx", px).attr("cy", py).attr("r", 5).attr("fill", "none")
          .attr("stroke", colr).attr("stroke-width", 1.6);
        gg.append("circle").attr("cx", px).attr("cy", py).attr("r", 2.4).attr("fill", colr);
      }
      /* the epipole, or an arrow towards it */
      const e = pt.e;
      if (e) {
        const ex = e[0] * PS, ey = e[1] * PS;
        if (ex > 0 && ex < PW && ey > 0 && ey < PH) {
          gg.append("circle").attr("cx", ex).attr("cy", ey).attr("r", 4).attr("fill", VC.bad);
          gg.append("text").attr("x", ex + 6).attr("y", ey - 5).attr("font-size", 10).attr("fill", VC.bad).text("e");
        } else {
          const cx = PW / 2, cy = PH / 2;
          const dx = ex - cx, dy = ey - cy, n = Math.hypot(dx, dy) || 1;
          VZ.arrow(gg, cx + 0.62 * PW * dx / n / 2 * 2 * 0.5, cy + 0.62 * PH * dy / n / 2 * 2 * 0.5,
            cx + dx / n * Math.min(PW, PH) * 0.44, cy + dy / n * Math.min(PW, PH) * 0.44,
            { color: VC.bad, w: 1.4, head: 6 });
          gg.append("text").attr("x", cx + dx / n * Math.min(PW, PH) * 0.44 + (dx > 0 ? -34 : 6))
            .attr("y", cy + dy / n * Math.min(PW, PH) * 0.44 - 6)
            .attr("font-size", 9.5).attr("fill", VC.bad).text("→ e (off frame)");
        }
      }
      return gg;
    }
    const g0 = imgPanel(P0, "image 0  ·  640 × 480 px", VC.accent,
      { u: [x0[0], x0[1]], e: ep.e0, x0: [x0[0], x0[1]] }, l0, true);
    imgPanel(P1, "image 1  ·  the line is the transfer of x₀ through F", VC.a2,
      { u: [x1[0], x1[1]], e: ep.e1, x0: [x0[0], x0[1]] }, l1, false);

    /* drag target on image 0 */
    g0.append("rect").attr("width", PW).attr("height", PH).attr("fill", "transparent")
      .style("cursor", "crosshair")
      .call(d3.drag().on("start drag", function (ev) {
        const m = d3.pointer(ev, this);
        drag = [VZ.clamp(m[0] / PS, 4, IW - 4), VZ.clamp(m[1] / PS, 4, IH - 4)];
        draw();
      }));

    /* ── readout ────────────────────────────────────────────────────────── */
    const resid = VZ.dot(SS.hom([x1[0], x1[1]]), VZ.mv(S.F, SS.hom([x0[0], x0[1]])));
    const ln = SS.normLine(l1);
    const dpx = Math.abs(VZ.dot(SS.hom([x1[0], x1[1]]), ln));
    const seg = SS.clipLine(l1, IW, IH);
    const segLen = seg ? Math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]) : 0;
    const inside = e => e && e[0] > 0 && e[0] < IW && e[1] > 0 && e[1] < IH;
    ro.html(
      `baseline <b>${VZ.fmt(S.B, 2)} m</b> · vergence <b>${VZ.fmt(VZ.deg(S.verg), 1)}°</b> · point at ` +
      `(${VZ.fmt(S.p[0], 2)}, ${VZ.fmt(S.p[1], 2)}, ${VZ.fmt(S.p[2], 2)}) m<br>` +
      `x₀ = (${VZ.fmt(x0[0], 1)}, ${VZ.fmt(x0[1], 1)}) px &nbsp; x₁ = (${VZ.fmt(x1[0], 1)}, ${VZ.fmt(x1[1], 1)}) px &nbsp; ` +
      `epipolar residual x₁ᵀFx₀ = <b>${resid.toExponential(2)}</b>, i.e. x₁ lies <b>${dpx.toExponential(2)} px</b> off its line<br>` +
      `e₀ = ${ep.e0 ? "(" + VZ.fmt(ep.e0[0], 0) + ", " + VZ.fmt(ep.e0[1], 0) + ") px" : "at infinity"} ` +
      `${inside(ep.e0) ? "(inside the frame)" : "(outside the frame)"} &nbsp;·&nbsp; ` +
      `e₁ = ${ep.e1 ? "(" + VZ.fmt(ep.e1[0], 0) + ", " + VZ.fmt(ep.e1[1], 0) + ") px" : "at infinity"} ` +
      `${inside(ep.e1) ? "(inside the frame)" : "(outside the frame)"}<br>` +
      `search: <b>${(IW * IH).toLocaleString()}</b> pixels unconstrained → <b>${Math.round(segLen)}</b> pixels on this epipolar line ` +
      `→ <b>${(IW * IH / Math.max(segLen, 1)).toFixed(0)}×</b> fewer candidates`
    );
  }

  ["ep-b", "ep-v", "ep-z", "ep-x"].forEach(id => {
    const f = () => {
      const v = +el(id).value;
      el(id + "v").textContent = id === "ep-v" ? VZ.fmt(v, 1) + "°"
        : (id === "ep-b" ? VZ.fmt(v, 2) + " m" : VZ.fmt(v, id === "ep-z" ? 1 : 2) + " m");
      if (id === "ep-x") drag = null;
      draw();
    };
    el(id).addEventListener("input", f);
    const v = +el(id).value;
    el(id + "v").textContent = id === "ep-v" ? VZ.fmt(v, 1) + "°"
      : (id === "ep-b" ? VZ.fmt(v, 2) + " m" : VZ.fmt(v, id === "ep-z" ? 1 : 2) + " m");
  });
  ["ep-plane", "ep-band"].forEach(id => el(id).addEventListener("change", draw));
  draw();
})();

/* ══════════ 2 · #pencil-svg — the pencil, and the epipole's flight ═══════ */
(function () {
  const svg = d3.select("#pencil-svg");
  if (svg.empty()) return;
  const W = 760, H = 482, IW = 640, IH = 480;
  const ro = d3.select("#pencil-readout");
  const el = id => document.getElementById(id);
  const PS = 0.47, PW = IW * PS, PH = IH * PS;         // 300.8 × 225.6
  const A = { x: 24, y: 34 }, B = { x: 424, y: 34 };
  const K0 = VZ.K({ f: 500, cx: 320, cy: 240 });
  const K1 = VZ.K({ f: 520, cx: 316, cy: 244 });

  /* a fixed cloud of scene points, in camera-0 coordinates (z forward) */
  const cloud = (function () {
    const r = VZ.rng(21), out = [];
    for (let i = 0; i < 26; i++) out.push([-1.6 + 3.2 * r(), -1.2 + 2.4 * r(), 3.5 + 6 * r()]);
    return out;
  })();

  function setup() {
    const a = VZ.rad(+el("pn-a").value), b = VZ.rad(+el("pn-e").value);
    const pan = VZ.rad(+el("pn-r").value);
    const pure = el("pn-rot").checked;
    const dir = [Math.cos(a) * Math.cos(b), Math.sin(b), Math.sin(a) * Math.cos(b)];
    const Bl = pure ? 0 : 0.9;
    const c1 = VZ.scale(dir, Bl);
    const R1 = VZ.Ry(pan);
    const t = VZ.neg(VZ.mv(R1, c1));
    const E = VZ.mul(VZ.skew(t), R1);
    let F = VZ.mul(VZ.mul(VZ.T(VZ.inv3(K1)), E), VZ.inv3(K0));
    let mx = 0; F.forEach(r => r.forEach(v => { mx = Math.max(mx, Math.abs(v)); }));
    if (mx > 1e-12) F = F.map(r => r.map(v => v / mx)); else F = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const cam0 = VZ.camera({ K: K0, R: VZ.eye(3), t: [0, 0, 0] });
    const cam1 = VZ.camera({ K: K1, R: R1, t: t });
    return { F: F, cam0: cam0, cam1: cam1, dir: dir, pure: pure, deg: +el("pn-a").value, mx: mx };
  }

  /* |e − principal point| for a given travel direction, no rotation, camera 0 */
  function epiDist(adeg, bdeg) {
    const a = VZ.rad(adeg), b = VZ.rad(bdeg);
    const d = [Math.cos(a) * Math.cos(b), Math.sin(b), Math.sin(a) * Math.cos(b)];
    if (Math.abs(d[2]) < 1e-9) return Infinity;
    return Math.hypot(500 * d[0] / d[2], 500 * d[1] / d[2]);
  }

  function draw() {
    const S = setup();
    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    const x0 = cloud.map(p => S.cam0.project(p));
    const x1 = cloud.map(p => S.cam1.project(p));
    const ep = SS.epipoles(S.F);

    function imgPanel(o, title, which) {
      const gg = VZ.panelBox(g, o.x, o.y, PW, PH, title, { fill: VC.bg });
      const cid = "pn-clip-" + which;
      const cp = VZ.clip(svg, cid, 0, 0, PW, PH);
      const inner = gg.append("g").attr("clip-path", cp);
      const pts = which === 0 ? x0 : x1;
      const other = which === 0 ? x1 : x0;
      const e = which === 0 ? ep.e0 : ep.e1;
      if (!S.pure) {
        pts.forEach((u, i) => {
          const src = which === 0 ? [other[i][0], other[i][1]] : [pts[i][0], pts[i][1]];
          const l = which === 0 ? VZ.mv(VZ.T(S.F), SS.hom([other[i][0], other[i][1]]))
            : VZ.mv(S.F, SS.hom([other[i][0], other[i][1]]));
          const seg = SS.clipLine(l, IW, IH);
          if (!seg) return;
          inner.append("line").attr("x1", seg[0][0] * PS).attr("y1", seg[0][1] * PS)
            .attr("x2", seg[1][0] * PS).attr("y2", seg[1][1] * PS)
            .attr("stroke", VC.violet).attr("stroke-width", 0.9).attr("stroke-opacity", 0.55);
        });
      }
      if (which === 1 && el("pn-mf").checked) {
        pts.forEach((u, i) => {
          const p = [x0[i][0] * PS, x0[i][1] * PS], q = [u[0] * PS, u[1] * PS];
          if (Math.hypot(q[0] - p[0], q[1] - p[1]) < 2.5) return;
          VZ.arrow(inner, p[0], p[1], q[0], q[1], { color: VC.good, w: 1.1, head: 4.5, op: 0.9 });
        });
      }
      pts.forEach(u => inner.append("circle").attr("cx", u[0] * PS).attr("cy", u[1] * PS).attr("r", 2)
        .attr("fill", which === 0 ? VC.accent : VC.a2));
      if (e) {
        const ex = e[0] * PS, ey = e[1] * PS;
        if (ex > -4 && ex < PW + 4 && ey > -4 && ey < PH + 4) {
          inner.append("circle").attr("cx", ex).attr("cy", ey).attr("r", 5.5).attr("fill", "none")
            .attr("stroke", VC.bad).attr("stroke-width", 1.6);
          inner.append("circle").attr("cx", ex).attr("cy", ey).attr("r", 2.6).attr("fill", VC.bad);
          gg.append("text").attr("x", VZ.clamp(ex + 8, 4, PW - 60)).attr("y", VZ.clamp(ey - 8, 12, PH - 6))
            .attr("font-size", 10).attr("fill", VC.bad).text("epipole e" + which);
        } else {
          const cx = PW / 2, cy = PH / 2, dx = ex - cx, dy = ey - cy, n = Math.hypot(dx, dy) || 1;
          VZ.arrow(gg, cx, cy, cx + dx / n * PW * 0.4, cy + dy / n * PW * 0.4, { color: VC.bad, w: 1.3, head: 6 });
          gg.append("text").attr("x", 6).attr("y", PH - 8).attr("font-size", 9.5).attr("fill", VC.bad)
            .text("e" + which + " off frame, " + (Math.hypot(e[0] - (which ? 316 : 320), e[1] - (which ? 244 : 240)) / 1).toFixed(0) + " px out");
        }
      } else {
        gg.append("text").attr("x", 6).attr("y", PH - 8).attr("font-size", 9.5).attr("fill", VC.bad)
          .text(S.pure ? "no epipole — F is identically zero" : "e" + which + " at infinity: the lines are parallel");
      }
      return gg;
    }
    imgPanel(A, "image 0  ·  the pencil of epipolar lines", 0);
    imgPanel(B, "image 1  ·  pencil + apparent motion", 1);

    /* ── the epipole-distance curve ──────────────────────────────────────── */
    const gc = VZ.panelBox(g, 60, 320, 660, 130, "distance from the principal point to the epipole, against the direction of travel");
    const bdeg = +el("pn-e").value;
    const xs = d3.scaleLinear().domain([0, 90]).range([0, 660]);
    const ys = d3.scaleLog().domain([80, 3e5]).range([130, 0]).clamp(true);
    gc.append("g").attr("transform", "translate(0,130)").attr("class", "axis")
      .call(d3.axisBottom(xs).ticks(10).tickFormat(d => d + "°"));
    gc.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(4, "~s"));
    gc.append("text").attr("x", 660).attr("y", 124).attr("text-anchor", "end")
      .attr("font-size", 10.5).attr("fill", VC.muted).text("0° = pure sideways · 90° = straight ahead");
    gc.append("text").attr("x", 2).attr("y", -6).attr("font-size", 10.5).attr("fill", VC.muted).text("px from principal point (log)");
    /* the frame half-diagonal: inside this, the epipole is in the picture */
    const halfDiag = Math.hypot(IW / 2, IH / 2);
    gc.append("line").attr("x1", 0).attr("x2", 660).attr("y1", ys(halfDiag)).attr("y2", ys(halfDiag))
      .attr("stroke", VC.good).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.2);
    gc.append("text").attr("x", 664).attr("y", ys(halfDiag) + 4).attr("font-size", 9.5).attr("fill", VC.good).text("frame");
    const path = [];
    for (let a = 0.35; a <= 90; a += 0.35) { const dv = epiDist(a, bdeg); if (isFinite(dv)) path.push([xs(a), ys(dv)]); }
    VZ.poly(gc, path, { stroke: VC.accent, w: 2, close: false });
    const cur = epiDist(S.deg, bdeg);
    if (isFinite(cur)) {
      gc.append("circle").attr("cx", xs(S.deg)).attr("cy", ys(cur)).attr("r", 4.5).attr("fill", VC.a2);
      gc.append("text").attr("x", xs(S.deg) + 7).attr("y", ys(cur) - 6).attr("font-size", 10).attr("fill", VC.a2)
        .text(VZ.fmt(cur, 0) + " px");
    }

    /* ── readout ─────────────────────────────────────────────────────────── */
    const inside = e => e && Math.abs(e[0] - 320) < IW / 2 && Math.abs(e[1] - 240) < IH / 2;
    let resid = 0;
    for (let i = 0; i < cloud.length; i++)
      resid = Math.max(resid, Math.abs(VZ.dot(SS.hom([x1[i][0], x1[i][1]]), VZ.mv(S.F, SS.hom([x0[i][0], x0[i][1]])))));
    if (S.pure) {
      ro.html(`<b>Pure rotation.</b> t = 0, so E = [t]<sub>×</sub>R = 0 and F = 0 exactly — the largest entry of the unscaled ` +
        `F is <b>${S.mx.toExponential(1)}</b>. There is no baseline, no epipolar plane, no epipolar line and no epipole. ` +
        `The apparent motion is still perfectly explained, but by a homography (part 6), not by this page's machinery. ` +
        `Any eight-point solve run on this data returns a meaningless answer — §11 measures how meaningless.`);
    } else {
      ro.html(
        `travel direction <b>${VZ.fmt(S.deg, 0)}°</b> from sideways, elevation <b>${VZ.fmt(bdeg, 0)}°</b>, camera-1 pan <b>${VZ.fmt(+el("pn-r").value, 1)}°</b><br>` +
        `e₀ = ${ep.e0 ? "(" + VZ.fmt(ep.e0[0], 0) + ", " + VZ.fmt(ep.e0[1], 0) + ") px, " + (inside(ep.e0) ? "<b>inside</b>" : "outside") + " the frame" : "<b>at infinity</b> — the pencil is a parallel family"}` +
        ` &nbsp;·&nbsp; e₁ = ${ep.e1 ? "(" + VZ.fmt(ep.e1[0], 0) + ", " + VZ.fmt(ep.e1[1], 0) + ") px, " + (inside(ep.e1) ? "<b>inside</b>" : "outside") + " the frame" : "<b>at infinity</b>"}<br>` +
        `every one of the ${cloud.length} correspondences satisfies x₁ᵀFx₀ = 0 to <b>${resid.toExponential(2)}</b> — the lines are not drawn through the points, they are <i>derived</i> and the points land on them`
      );
    }
  }

  [["pn-a", v => VZ.fmt(v, 0) + "°"], ["pn-e", v => VZ.fmt(v, 0) + "°"], ["pn-r", v => VZ.fmt(v, 1) + "°"]]
    .forEach(([id, f]) => {
      const upd = () => { el(id + "v").textContent = f(+el(id).value); draw(); };
      el(id).addEventListener("input", upd);
      el(id + "v").textContent = f(+el(id).value);
    });
  ["pn-mf", "pn-rot"].forEach(id => el(id).addEventListener("change", draw));
  draw();
})();

/* ══════════ 3 · #emat-svg — E = [t]×R, factor by factor ══════════════════ */
(function () {
  const svg = d3.select("#emat-svg");
  if (svg.empty()) return;
  const W = 760, H = 392;
  const ro = d3.select("#emat-readout");
  const el = id => document.getElementById(id);
  const K0 = VZ.K({ f: 500, cx: 320, cy: 240 });
  const K1 = VZ.K({ f: 520, cx: 316, cy: 244 });
  const cloud = (function () {
    const r = VZ.rng(7), o = [];
    for (let i = 0; i < 40; i++) o.push([-1.2 + 2.4 * r(), -0.9 + 1.8 * r(), 4 + 5 * r()]);
    return o;
  })();

  function draw() {
    const yaw = VZ.rad(+el("em-y").value), pit = VZ.rad(+el("em-p").value), rol = VZ.rad(+el("em-r").value);
    const cz = +el("em-a").value, cx = +el("em-b").value;
    const R = VZ.mul(VZ.mul(VZ.Rz(yaw), VZ.Ry(pit)), VZ.Rx(rol));
    /* the camera-1 centre, in camera 0's frame. Defaults are the running rig's
       c₁ = (1.00, 0.05, 0.10) m exactly, so the figure reproduces §04's printed E. */
    const c1 = [cx, 0.05, cz];
    const t = VZ.neg(VZ.mv(R, c1));
    const Tx = VZ.skew(t), E = VZ.mul(Tx, R);
    const sv = SS.svd(E).s;
    const cam0 = VZ.camera({ K: K0, R: VZ.eye(3), t: [0, 0, 0] });
    const cam1 = VZ.camera({ K: K1, R: R, t: t });

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: the three matrices ───────────────────────────────────────────── */
    const ga = VZ.panelBox(g, 16, 34, 262, 330, "the factors, and the product");
    VZ.matText(ga, Tx, 12, 34, { dp: 6, pad: 10, size: 10, label: "[t]×   (skew, rank 2, null space t)" });
    VZ.matText(ga, R, 12, 132, { dp: 6, pad: 10, size: 10, label: "R   (orthonormal, det = +1)" });
    VZ.matText(ga, E, 12, 230, { dp: 6, pad: 10, size: 10, label: "E = [t]×R", fill: VC.a2 });
    const kv = VZ.kv(ga, 12, 282, { keyW: 132, size: 10.5 });
    kv("‖t‖ (baseline)", VZ.fmt(VZ.norm(t), 6) + " m", VC.good);
    kv("det E", VZ.det3(E).toExponential(2), VC.ink);
    kv("‖Eᵀt‖", VZ.norm(VZ.mv(VZ.T(E), t)).toExponential(2), VC.ink);

    /* ── B: the singular values ──────────────────────────────────────────── */
    const gb = VZ.panelBox(g, 300, 34, 190, 330, "singular values of E");
    const yb = d3.scaleLinear().domain([0, Math.max(1e-9, sv[0]) * 1.25]).range([250, 0]);
    gb.append("g").attr("class", "axis").attr("transform", "translate(38,20)").call(d3.axisLeft(yb).ticks(5));
    const bw = 34;
    /* the perturbed comparison: E plus a small random matrix is no longer essential */
    const rr = VZ.rng(99);
    const Ep = E.map(row => row.map(v => v + 0.055 * sv[0] * VZ.randn(rr)));
    const svp = SS.svd(Ep).s;
    sv.forEach((s, i) => {
      const x = 56 + i * 46;
      gb.append("rect").attr("x", x).attr("y", 20 + yb(s)).attr("width", bw).attr("height", Math.max(0.7, 250 - yb(s)))
        .attr("fill", i < 2 ? VC.a2 : VC.muted).attr("fill-opacity", 0.85);
      gb.append("rect").attr("x", x + 9).attr("y", 20 + yb(svp[i])).attr("width", bw - 18).attr("height", Math.max(0.7, 250 - yb(svp[i])))
        .attr("fill", "none").attr("stroke", VC.bad).attr("stroke-width", 1.4).attr("stroke-dasharray", "3 2");
      gb.append("text").attr("x", x + bw / 2).attr("y", 284).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", VC.muted).text("σ" + (i + 1));
      gb.append("text").attr("x", x + bw / 2).attr("y", 12 + yb(s)).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", VC.ink).text(s < 1e-9 ? s.toExponential(0) : VZ.fmt(s, 4));
    });
    gb.append("text").attr("x", 8).attr("y", 306).attr("font-size", 10).attr("fill", VC.a2)
      .text("σ₁ = σ₂ = ‖t‖ exactly, σ₃ = 0 exactly");
    gb.append("text").attr("x", 8).attr("y", 320).attr("font-size", 10).attr("fill", VC.bad)
      .text("dashed: E + 5.5% noise — neither holds");

    /* ── C: the residuals, rays against pixels ───────────────────────────── */
    const gc = VZ.panelBox(g, 512, 34, 232, 330, "|constraint residual| per correspondence");
    const xs = d3.scaleLinear().domain([0, cloud.length - 1]).range([0, 190]);
    const ys = d3.scaleLog().domain([1e-18, 1e7]).range([300, 0]).clamp(true);
    gc.append("g").attr("class", "axis").attr("transform", "translate(38,14)").call(d3.axisLeft(ys).ticks(6, "~e"));
    gc.append("text").attr("x", 40).attr("y", 328).attr("font-size", 10).attr("fill", VC.muted).text("correspondence index");
    let mr = 0, mp = 0;
    cloud.forEach((p, i) => {
      const a = cam0.project(p), b = cam1.project(p);
      const h0 = VZ.mv(VZ.inv3(K0), SS.hom([a[0], a[1]])), h1 = VZ.mv(VZ.inv3(K1), SS.hom([b[0], b[1]]));
      const rray = Math.abs(VZ.dot(h1, VZ.mv(E, h0)));
      const rpix = Math.abs(VZ.dot(SS.hom([b[0], b[1]]), VZ.mv(E, SS.hom([a[0], a[1]]))));
      mr = Math.max(mr, rray); mp = Math.max(mp, rpix);
      gc.append("circle").attr("cx", 38 + xs(i)).attr("cy", 14 + ys(Math.max(rray, 1e-18))).attr("r", 2.6).attr("fill", VC.good);
      gc.append("circle").attr("cx", 38 + xs(i)).attr("cy", 14 + ys(Math.max(rpix, 1e-18))).attr("r", 2.6).attr("fill", VC.bad);
    });
    VZ.legend(gc, [{ color: VC.good, label: "normalised rays x̂ = K⁻¹x̄" }, { color: VC.bad, label: "raw pixels — a category error" }], 44, 26, { gap: 14, font: 9.5 });

    ro.html(
      `E singular values <b>(${VZ.fmt(sv[0], 6)}, ${VZ.fmt(sv[1], 6)}, ${sv[2].toExponential(2)})</b> — ` +
      `σ₁ − σ₂ = <b>${(sv[0] - sv[1]).toExponential(2)}</b>, and σ₁ = ‖t‖ = ${VZ.fmt(VZ.norm(t), 6)} m to ` +
      `<b>${Math.abs(sv[0] - VZ.norm(t)).toExponential(1)}</b><br>` +
      `perturbed by 5.5%: singular values (${VZ.fmt(svp[0], 4)}, ${VZ.fmt(svp[1], 4)}, ${VZ.fmt(svp[2], 4)}) — ` +
      `σ₃ is no longer zero and σ₁ ≠ σ₂, so the perturbed matrix is <b>not</b> an essential matrix at all<br>` +
      `residual over ${cloud.length} correspondences: <b>${mr.toExponential(2)}</b> on normalised rays, ` +
      `<b>${mp.toExponential(2)}</b> on raw pixels — a factor of <b>${(mp / Math.max(mr, 1e-300)).toExponential(1)}</b>`
    );
  }

  [["em-y", 1, "°"], ["em-p", 1, "°"], ["em-r", 1, "°"], ["em-a", 2, " m"], ["em-b", 2, " m"]].forEach(([id, dp, u]) => {
    const upd = () => { el(id + "v").textContent = VZ.fmt(+el(id).value, dp) + u; draw(); };
    el(id).addEventListener("input", upd);
    el(id + "v").textContent = VZ.fmt(+el(id).value, dp) + u;
  });
  draw();
})();

/* ══════════ 4 · #fmat-svg — transfer through F, and three residuals ══════ */
(function () {
  const svg = d3.select("#fmat-svg");
  if (svg.empty()) return;
  const W = 760, H = 440, IW = 640, IH = 480;
  const ro = d3.select("#fmat-readout");
  const el = id => document.getElementById(id);
  const PS = 0.5, PW = IW * PS, PH = IH * PS;          // 320 × 240
  const A = { x: 24, y: 34 }, B = { x: 404, y: 34 };

  /* the running rig, verbatim */
  const RIG = SS.rig({
    K0: VZ.K({ f: 500, cx: 320, cy: 240 }), K1: VZ.K({ f: 520, cx: 316, cy: 244 }),
    R1: VZ.mul(VZ.mul(VZ.Rz(VZ.rad(2)), VZ.Ry(VZ.rad(-8))), VZ.Rx(VZ.rad(3))), c1: [1, 0.05, 0.10]
  });
  const F1 = (function () { let m = 0; RIG.F.forEach(r => r.forEach(v => { m = Math.max(m, Math.abs(v)); })); return RIG.F.map(r => r.map(v => v / m)); })();

  /* back-project a pixel of image 0 to depth 5 m and forward-project it: the
     "true" match, which is what the residual is measured against */
  function trueMatch(u) {
    const ray = VZ.mv(VZ.inv3(RIG.K0), SS.hom(u));
    const p = VZ.scale(ray, 5.0 / ray[2]);
    const q = RIG.cam1.project(p);
    return { x1: [q[0], q[1]], p: p };
  }

  function draw() {
    const u0 = [+el("fm-u").value, +el("fm-v").value];
    const off = +el("fm-d").value, sc = +el("fm-s").value;
    const F = F1.map(r => r.map(v => v * sc));
    const tm = trueMatch(u0);
    const l1 = SS.normLine(VZ.mv(F, SS.hom(u0)));
    const x1 = [tm.x1[0] + off * l1[0], tm.x1[1] + off * l1[1]];
    const l0 = VZ.mv(VZ.T(F), SS.hom(x1));

    const meas = o => {
      const q = [tm.x1[0] + o * l1[0], tm.x1[1] + o * l1[1]];
      const L1 = VZ.mv(F, SS.hom(u0)), L0 = VZ.mv(VZ.T(F), SS.hom(q));
      const r = VZ.dot(SS.hom(q), L1);
      const n1 = Math.hypot(L1[0], L1[1]), n0 = Math.hypot(L0[0], L0[1]);
      const d1 = Math.abs(r) / n1, d0 = Math.abs(r) / n0;
      return { r: r, alg: Math.abs(r), sym: Math.sqrt((d0 * d0 + d1 * d1) / 2), sam: Math.abs(r) / Math.sqrt(n0 * n0 + n1 * n1) };
    };
    const cur = meas(off);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    function panel(o, title, line, marks, id) {
      const gg = VZ.panelBox(g, o.x, o.y, PW, PH, title, { fill: VC.bg });
      const cp = VZ.clip(svg, "fm-clip-" + id, 0, 0, PW, PH);
      const inner = gg.append("g").attr("clip-path", cp);
      const seg = SS.clipLine(line, IW, IH);
      if (seg) inner.append("line").attr("x1", seg[0][0] * PS).attr("y1", seg[0][1] * PS)
        .attr("x2", seg[1][0] * PS).attr("y2", seg[1][1] * PS)
        .attr("stroke", VC.violet).attr("stroke-width", 1.8);
      marks.forEach(m => {
        inner.append("circle").attr("cx", m.p[0] * PS).attr("cy", m.p[1] * PS).attr("r", m.r || 5)
          .attr("fill", m.fill || "none").attr("stroke", m.c).attr("stroke-width", 1.7)
          .attr("stroke-dasharray", m.dash || null);
        if (m.lab) inner.append("text").attr("x", m.p[0] * PS + 8).attr("y", m.p[1] * PS - 6)
          .attr("font-size", 10).attr("fill", m.c).text(m.lab);
      });
      return { gg: gg, inner: inner };
    }

    const pa = panel(A, "image 0  ·  line = Fᵀx₁, transferred back", l0,
      [{ p: u0, c: VC.accent, lab: "x₀", fill: VC.accent, r: 3.4 }], "a");
    panel(B, "image 1  ·  line = Fx₀, and the match pushed off it", l1,
      [{ p: tm.x1, c: VC.good, lab: "true match", fill: VC.good, r: 3.4 },
      { p: x1, c: VC.bad, lab: Math.abs(off) > 0.01 ? "displaced" : "", r: 6, dash: "3 2" }], "b");

    pa.gg.append("rect").attr("width", PW).attr("height", PH).attr("fill", "transparent")
      .style("cursor", "crosshair")
      .call(d3.drag().on("start drag", function (ev) {
        const m = d3.pointer(ev, this);
        el("fm-u").value = Math.round(VZ.clamp(m[0] / PS, 40, 600));
        el("fm-v").value = Math.round(VZ.clamp(m[1] / PS, 40, 440));
        el("fm-uv").textContent = el("fm-u").value; el("fm-vv").textContent = el("fm-v").value;
        draw();
      }));

    /* ── the residual curves ─────────────────────────────────────────────── */
    const gc = VZ.panelBox(g, 60, 320, 660, 100, "residual against perpendicular displacement of the match, in pixels");
    const xs = d3.scaleLinear().domain([-30, 30]).range([0, 660]);
    const ys = d3.scaleLinear().domain([0, 32]).range([100, 0]);
    const ysA = d3.scaleLinear().domain([0, Math.max(1e-9, meas(30).alg * 1.05)]).range([100, 0]);
    VZ.gridY(gc, ys, 660, 4);
    gc.append("g").attr("class", "axis").attr("transform", "translate(0,100)").call(d3.axisBottom(xs).ticks(9));
    gc.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(4));
    gc.append("text").attr("x", 2).attr("y", -6).attr("font-size", 10).attr("fill", VC.muted).text("px (left axis) · algebraic residual rescaled to fit");
    const cs = [], cy = [], ca = [];
    for (let o = -30; o <= 30.001; o += 0.5) {
      const m = meas(o);
      cs.push([xs(o), ys(m.sym)]); cy.push([xs(o), ys(m.sam)]); ca.push([xs(o), ysA(m.alg)]);
    }
    VZ.poly(gc, ca, { stroke: VC.bad, w: 1.6, close: false, dash: "4 3" });
    VZ.poly(gc, cs, { stroke: VC.accent, w: 2, close: false });
    VZ.poly(gc, cy, { stroke: VC.good, w: 2, close: false });
    gc.append("line").attr("x1", xs(off)).attr("x2", xs(off)).attr("y1", 0).attr("y2", 100)
      .attr("stroke", VC.a2).attr("stroke-width", 1.3).attr("stroke-dasharray", "3 3");
    VZ.legend(gc, [
      { color: VC.accent, label: "symmetric epipolar distance (px)" },
      { color: VC.good, label: "Sampson distance (px)" },
      { color: VC.bad, label: "algebraic |x₁ᵀFx₀| (rescaled)", dash: "4 3" }
    ], 16, 16, { gap: 14, font: 10 });

    ro.html(
      `x₀ = (${u0[0]}, ${u0[1]}) px, its true match at depth 5 m is (${VZ.fmt(tm.x1[0], 1)}, ${VZ.fmt(tm.x1[1], 1)}) px, ` +
      `displaced by <b>${VZ.fmt(off, 1)} px</b> perpendicular to the line<br>` +
      `algebraic |x₁ᵀFx₀| = <b>${cur.alg.toExponential(3)}</b> &nbsp;·&nbsp; symmetric epipolar = <b>${VZ.fmt(cur.sym, 3)} px</b> ` +
      `&nbsp;·&nbsp; Sampson = <b>${VZ.fmt(cur.sam, 3)} px</b><br>` +
      (sc === 1
        ? `F is at its estimated scale (largest entry 1). Change the scale selector: the algebraic residual will move with it, ` +
        `the two geometric measures will not — which is the whole reason the algebraic one is unusable as a threshold.`
        : `F has been scaled by <b>×${sc}</b>. The algebraic residual scaled with it; the symmetric epipolar and Sampson ` +
        `distances are unchanged, because both divide by the same scale twice over. Only they can carry a fixed pixel threshold.`)
    );
  }

  [["fm-u", v => "" + v], ["fm-v", v => "" + v], ["fm-d", v => VZ.fmt(v, 1) + " px"]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  el("fm-s").addEventListener("change", draw);
  draw();
})();

/* ══════════ 5 · #rank-svg — rank 2 is concurrency ═══════════════════════ */
(function () {
  const svg = d3.select("#rank-svg");
  if (svg.empty()) return;
  const W = 760, H = 392, IW = 640, IH = 480;
  const ro = d3.select("#rank-readout");
  const el = id => document.getElementById(id);
  const PS = 0.6, PW = IW * PS, PH = IH * PS;          // 384 × 288
  const RIG = SS.rig({
    K0: VZ.K({ f: 500, cx: 320, cy: 240 }), K1: VZ.K({ f: 520, cx: 316, cy: 244 }),
    R1: VZ.mul(VZ.mul(VZ.Rz(VZ.rad(2)), VZ.Ry(VZ.rad(-8))), VZ.Rx(VZ.rad(3))), c1: [1, 0.05, 0.10]
  });
  const unit = M => { let m = 0; M.forEach(r => r.forEach(v => { m = Math.max(m, Math.abs(v)); })); return M.map(r => r.map(v => v / m)); };
  const F0 = unit(RIG.F), E0 = unit(RIG.E);
  /* one fixed perturbation direction, so the slider is a clean one-parameter family */
  const NOISE = (function () { const r = VZ.rng(4), M = []; for (let i = 0; i < 3; i++) M.push([VZ.randn(r), VZ.randn(r), VZ.randn(r)]); return M; })();

  function draw() {
    const eps = +el("rk-p").value, n = +el("rk-n").value, which = el("rk-m").value;
    const base = which === "F" ? F0 : E0;
    const M = base.map((r, i) => r.map((v, j) => v + eps * NOISE[i][j]));
    const sv = SS.svd(M).s;
    /* the lines: transfer a spread of image-0 points (or rays) through M */
    const src = [];
    for (let i = 0; i < n; i++) {
      const u = 60 + (IW - 120) * i / (n - 1), v = 70 + (IH - 140) * ((i * 7) % n) / (n - 1);
      src.push(which === "F" ? [u, v] : VZ.dehom(VZ.mv(VZ.inv3(RIG.K0), [u, v, 1])));
    }
    const lines = src.map(p => VZ.mv(M, SS.hom(p)));
    /* draw everything in PIXELS of image 1: for E, push the line back to pixels
       with l_pix = K₁⁻ᵀ l_ray, which is how a ray-space line looks on the sensor */
    const toPix = l => which === "F" ? l : VZ.mv(VZ.T(VZ.inv3(RIG.K1)), l);
    const lp = lines.map(toPix);
    /* pairwise intersections */
    const ints = [];
    for (let i = 0; i < lp.length; i++) for (let j = i + 1; j < lp.length; j++) {
      const x = VZ.cross(lp[i], lp[j]);
      if (Math.abs(x[2]) > 1e-12) ints.push([x[0] / x[2], x[1] / x[2]]);
    }
    const cx = d3.mean(ints, p => p[0]) || 0, cy = d3.mean(ints, p => p[1]) || 0;
    const rad = ints.length ? d3.max(ints, p => Math.hypot(p[0] - cx, p[1] - cy)) : 0;

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: the lines ────────────────────────────────────────────────────── */
    const ga = VZ.panelBox(g, 16, 34, PW, PH, "image 1 · the pencil, and where the lines actually meet", { fill: VC.bg });
    const cp = VZ.clip(svg, "rk-clip", 0, 0, PW, PH);
    const inner = ga.append("g").attr("clip-path", cp);
    lp.forEach(l => {
      const seg = SS.clipLine(l, IW, IH);
      if (!seg) return;
      inner.append("line").attr("x1", seg[0][0] * PS).attr("y1", seg[0][1] * PS)
        .attr("x2", seg[1][0] * PS).attr("y2", seg[1][1] * PS)
        .attr("stroke", VC.violet).attr("stroke-width", 1).attr("stroke-opacity", 0.6);
    });
    /* the intersection cloud, drawn in a zoomed inset so it is visible at all */
    const iw = 118, zx = PW - iw - 8, zy = 8;
    const gi = ga.append("g").attr("transform", `translate(${zx},${zy})`);
    gi.append("rect").attr("width", iw).attr("height", iw).attr("fill", VC.panel2).attr("stroke", VC.line);
    const zoomR = Math.max(rad * 1.35, 1e-3);
    const zs = (iw / 2) / zoomR;
    ints.forEach(p => gi.append("circle").attr("cx", iw / 2 + (p[0] - cx) * zs).attr("cy", iw / 2 + (p[1] - cy) * zs)
      .attr("r", 1.7).attr("fill", VC.a2).attr("fill-opacity", 0.8));
    gi.append("circle").attr("cx", iw / 2).attr("cy", iw / 2).attr("r", iw / 2 - 2)
      .attr("fill", "none").attr("stroke", eps === 0 ? VC.good : VC.bad).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 2");
    gi.append("text").attr("x", 4).attr("y", 11).attr("font-size", 9).attr("fill", VC.muted).text("intersections, zoomed");
    gi.append("text").attr("x", 4).attr("y", iw - 4).attr("font-size", 9).attr("fill", eps === 0 ? VC.good : VC.bad)
      .text("spread " + (rad < 1e-6 ? rad.toExponential(1) : VZ.fmt(rad, 2)) + " px");

    /* ── B: singular values ──────────────────────────────────────────────── */
    const gb = VZ.panelBox(g, 424, 34, 128, 288, "singular values");
    const yb = d3.scaleLinear().domain([0, sv[0] * 1.2]).range([236, 0]);
    gb.append("g").attr("class", "axis").attr("transform", "translate(34,20)").call(d3.axisLeft(yb).ticks(5));
    sv.forEach((s, i) => {
      const x = 46 + i * 28;
      gb.append("rect").attr("x", x).attr("y", 20 + yb(s)).attr("width", 20).attr("height", Math.max(0.6, 236 - yb(s)))
        .attr("fill", i === 2 ? (eps === 0 ? VC.muted : VC.bad) : VC.accent).attr("fill-opacity", 0.85);
      gb.append("text").attr("x", x + 10).attr("y", 272).attr("text-anchor", "middle").attr("font-size", 10)
        .attr("fill", VC.muted).text("σ" + (i + 1));
    });
    gb.append("text").attr("x", 6).attr("y", 288).attr("font-size", 10).attr("fill", eps === 0 ? VC.good : VC.bad)
      .text("σ₃/σ₁ = " + (sv[2] / sv[0]).toExponential(2));

    /* ── C: the two measures against the perturbation ────────────────────── */
    const gc = VZ.panelBox(g, 600, 34, 144, 288, "spread & σ₃/σ₁ vs perturbation");
    const xs = d3.scaleLinear().domain([0, 0.06]).range([0, 144]);
    const ys = d3.scaleLog().domain([1e-14, 3e3]).range([288, 0]).clamp(true);
    gc.append("g").attr("class", "axis").attr("transform", "translate(0,288)").call(d3.axisBottom(xs).ticks(3));
    const c1 = [], c2 = [];
    for (let e = 0; e <= 0.0601; e += 0.002) {
      const Mm = base.map((r, i) => r.map((v, j) => v + e * NOISE[i][j]));
      const s2 = SS.svd(Mm).s;
      const lps = src.map(p => toPix(VZ.mv(Mm, SS.hom(p))));
      const ii = [];
      for (let i = 0; i < lps.length; i++) for (let j = i + 1; j < lps.length; j++) {
        const x = VZ.cross(lps[i], lps[j]);
        if (Math.abs(x[2]) > 1e-12) ii.push([x[0] / x[2], x[1] / x[2]]);
      }
      const mx = d3.mean(ii, p => p[0]), my = d3.mean(ii, p => p[1]);
      const rr = ii.length ? d3.max(ii, p => Math.hypot(p[0] - mx, p[1] - my)) : 0;
      c1.push([xs(e), ys(Math.max(rr, 1e-14))]);
      c2.push([xs(e), ys(Math.max(s2[2] / s2[0], 1e-14))]);
    }
    VZ.poly(gc, c1, { stroke: VC.a2, w: 1.8, close: false });
    VZ.poly(gc, c2, { stroke: VC.accent, w: 1.8, close: false });
    gc.append("line").attr("x1", xs(eps)).attr("x2", xs(eps)).attr("y1", 0).attr("y2", 288)
      .attr("stroke", VC.good).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
    VZ.legend(gc, [{ color: VC.a2, label: "intersection spread (px)" }, { color: VC.accent, label: "σ₃/σ₁" }], 8, 14, { gap: 13, font: 9.5 });

    ro.html(eps === 0
      ? `perturbation <b>0</b>: σ₃/σ₁ = <b>${(sv[2] / sv[0]).toExponential(2)}</b> — numerically zero — and all ` +
      `<b>${ints.length}</b> pairwise intersections of the ${n} lines coincide to within <b>${rad.toExponential(2)} px</b>. ` +
      `That is what rank 2 means, drawn: one point, shared by every line in the pencil.`
      : `perturbation <b>${VZ.fmt(eps, 3)}</b>: σ₃/σ₁ = <b>${(sv[2] / sv[0]).toExponential(2)}</b>, and the ${ints.length} pairwise ` +
      `intersections now scatter over a disc of radius <b>${VZ.fmt(rad, 2)} px</b>. There is no epipole any more — only a cloud ` +
      `of near-misses — and every step downstream that asks for "the" epipole has to pick one.`);
  }

  [["rk-p", v => VZ.fmt(v, 3)], ["rk-n", v => "" + v]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  el("rk-m").addEventListener("change", draw);
  draw();
})();

/* ══════════ 6 · #eight-svg — normalisation, measured ════════════════════ */
(function () {
  const svg = d3.select("#eight-svg");
  if (svg.empty()) return;
  const W = 760, H = 452;
  const ro = d3.select("#eight-readout");
  const el = id => document.getElementById(id);
  const RIG = SS.rig({
    K0: VZ.K({ f: 500, cx: 320, cy: 240 }), K1: VZ.K({ f: 520, cx: 316, cy: 244 }),
    R1: VZ.mul(VZ.mul(VZ.Rz(VZ.rad(2)), VZ.Ry(VZ.rad(-8))), VZ.Rx(VZ.rad(3))), c1: [1, 0.05, 0.10]
  });
  /* one fixed cloud, drawn from a fixed seed so the picture is reproducible */
  const CLOUD = (function () {
    const r = VZ.rng(7), o = [];
    for (let i = 0; i < 80; i++) o.push([-1.2 + 2.4 * r(), -0.9 + 1.8 * r(), 4 + 5 * r()]);
    return o;
  })();
  const designRow = (a, b) => [b[0] * a[0], b[0] * a[1], b[0], b[1] * a[0], b[1] * a[1], b[1], a[0], a[1], 1];

  function draw() {
    const sig = +el("e8-s").value, N = +el("e8-n").value, TR = +el("e8-t").value;
    const enf = el("e8-r").checked;
    const pts = CLOUD.slice(0, N);
    const p0 = pts.map(p => RIG.cam0.project(p).slice(0, 2));
    const p1 = pts.map(p => RIG.cam1.project(p).slice(0, 2));

    /* one representative trial, for the column and singular-value panels */
    const r1 = VZ.rng(1234);
    const q0 = p0.map(p => [p[0] + sig * VZ.randn(r1), p[1] + sig * VZ.randn(r1)]);
    const q1 = p1.map(p => [p[0] + sig * VZ.randn(r1), p[1] + sig * VZ.randn(r1)]);
    const T0 = SS.normaliseT(q0), T1 = SS.normaliseT(q1);
    const n0 = q0.map(p => VZ.applyH(T0, p)), n1 = q1.map(p => VZ.applyH(T1, p));
    const Araw = q0.map((p, i) => designRow(p, q1[i]));
    const Anrm = n0.map((p, i) => designRow(p, n1[i]));
    const rms = A => d3.range(9).map(j => Math.sqrt(d3.mean(A, row => row[j] * row[j])));
    const cRaw = rms(Araw), cNrm = rms(Anrm);
    const sRaw = SS.svd(Araw).s, sNrm = SS.svd(Anrm).s;
    const condRaw = sRaw[0] / sRaw[7], condNrm = sNrm[0] / sNrm[7];

    /* the trials */
    const rt = VZ.rng(11);
    const eN = [], eU = [];
    let cN = 0, cU = 0;
    for (let k = 0; k < TR; k++) {
      const a0 = p0.map(p => [p[0] + sig * VZ.randn(rt), p[1] + sig * VZ.randn(rt)]);
      const a1 = p1.map(p => [p[0] + sig * VZ.randn(rt), p[1] + sig * VZ.randn(rt)]);
      const fn = SS.eightPoint(a0, a1, true, enf), fu = SS.eightPoint(a0, a1, false, enf);
      eN.push(Math.max(SS.symEpi(fn.F, p0, p1), 1e-16));
      eU.push(Math.max(SS.symEpi(fu.F, p0, p1), 1e-16));
      cN += fn.cond; cU += fu.cond;
    }
    cN /= TR; cU /= TR;
    const mN = d3.mean(eN), mU = d3.mean(eU);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: column magnitudes ───────────────────────────────────────────── */
    const ga = VZ.panelBox(g, 46, 40, 214, 360, "RMS magnitude of each design column");
    const ya = d3.scaleLog().domain([1e-1, 1e6]).range([360, 0]).clamp(true);
    ga.append("g").attr("class", "axis").call(d3.axisLeft(ya).ticks(8, "~e"));
    const xa = d3.scaleBand().domain(d3.range(9)).range([0, 214]).padding(0.22);
    cRaw.forEach((v, j) => {
      ga.append("rect").attr("x", xa(j)).attr("y", ya(Math.max(v, 1e-1))).attr("width", xa.bandwidth())
        .attr("height", 360 - ya(Math.max(v, 1e-1))).attr("fill", VC.bad).attr("fill-opacity", 0.6);
      ga.append("rect").attr("x", xa(j) + 2).attr("y", ya(Math.max(cNrm[j], 1e-1))).attr("width", xa.bandwidth() - 4)
        .attr("height", 360 - ya(Math.max(cNrm[j], 1e-1))).attr("fill", VC.good).attr("fill-opacity", 0.85);
    });
    ["u₁u₀", "u₁v₀", "u₁", "v₁u₀", "v₁v₀", "v₁", "u₀", "v₀", "1"].forEach((lb, j) =>
      ga.append("text").attr("x", xa(j) + xa.bandwidth() / 2).attr("y", 374).attr("text-anchor", "middle")
        .attr("font-size", 8.5).attr("fill", VC.muted).text(lb));
    VZ.legend(ga, [{ color: VC.bad, label: "raw pixels" }, { color: VC.good, label: "normalised" }], 8, 14, { gap: 13, font: 10 });
    ga.append("text").attr("x", 6).attr("y", 390).attr("font-size", 9.5).attr("fill", VC.muted)
      .text("spread: raw " + (d3.max(cRaw) / d3.min(cRaw)).toExponential(1) + "×, normalised " + VZ.fmt(d3.max(cNrm) / d3.min(cNrm), 1) + "×");

    /* ── B: singular values of A ────────────────────────────────────────── */
    const gb = VZ.panelBox(g, 300, 40, 190, 360, "singular values of A (log)");
    const yb = d3.scaleLog().domain([1e-3, 1e7]).range([360, 0]).clamp(true);
    gb.append("g").attr("class", "axis").call(d3.axisLeft(yb).ticks(8, "~e"));
    const xb = d3.scaleLinear().domain([0, 8]).range([16, 174]);
    gb.append("g").attr("class", "axis").attr("transform", "translate(0,360)").call(d3.axisBottom(xb).ticks(9).tickFormat(d => d + 1));
    [[sRaw, VC.bad], [sNrm, VC.good]].forEach(([s, c]) => {
      VZ.poly(gb, s.map((v, j) => [xb(j), yb(Math.max(v, 1e-3))]), { stroke: c, w: 1.8, close: false });
      s.forEach((v, j) => gb.append("circle").attr("cx", xb(j)).attr("cy", yb(Math.max(v, 1e-3))).attr("r", 2.6).attr("fill", c));
    });
    gb.append("line").attr("x1", xb(7)).attr("x2", xb(7)).attr("y1", 0).attr("y2", 360)
      .attr("stroke", VC.line).attr("stroke-dasharray", "2 3");
    gb.append("text").attr("x", 4).attr("y", 376).attr("font-size", 9.5).attr("fill", VC.bad)
      .text("cond σ₁/σ₈ raw: " + condRaw.toExponential(2));
    gb.append("text").attr("x", 4).attr("y", 388).attr("font-size", 9.5).attr("fill", VC.good)
      .text("cond σ₁/σ₈ normalised: " + VZ.fmt(condNrm, 2));

    /* ── C: the error distributions ─────────────────────────────────────── */
    const gc = VZ.panelBox(g, 530, 40, 214, 360, "symmetric epipolar error over " + TR + " trials");
    const lo = 1e-4, hi = 1e3;
    const xc = d3.scaleLog().domain([lo, hi]).range([0, 214]).clamp(true);
    gc.append("g").attr("class", "axis").attr("transform", "translate(0,360)").call(d3.axisBottom(xc).ticks(6, "~e"));
    const bins = d3.range(31).map(i => lo * Math.pow(hi / lo, i / 30));
    const hist = arr => { const h = new Array(30).fill(0); arr.forEach(v => { const k = Math.floor(30 * Math.log(VZ.clamp(v, lo, hi) / lo) / Math.log(hi / lo)); h[VZ.clamp(k, 0, 29)]++; }); return h; };
    const hN = hist(eN), hU = hist(eU);
    const mxh = Math.max(d3.max(hN), d3.max(hU), 1);
    const yc = d3.scaleLinear().domain([0, mxh]).range([340, 20]);
    [[hU, VC.bad, 0.55], [hN, VC.good, 0.8]].forEach(([h, c, op]) => {
      h.forEach((v, i) => {
        if (!v) return;
        gc.append("rect").attr("x", xc(bins[i])).attr("y", yc(v))
          .attr("width", Math.max(1.5, xc(bins[i + 1]) - xc(bins[i]))).attr("height", 340 - yc(v))
          .attr("fill", c).attr("fill-opacity", op);
      });
    });
    [[mU, VC.bad, "unnormalised"], [mN, VC.good, "normalised"]].forEach(([m, c, lb]) => {
      gc.append("line").attr("x1", xc(m)).attr("x2", xc(m)).attr("y1", 14).attr("y2", 344)
        .attr("stroke", c).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
      gc.append("text").attr("x", VZ.clamp(xc(m) - 30, 0, 130)).attr("y", lb === "normalised" ? 12 : 24)
        .attr("font-size", 9.5).attr("fill", c).text(lb + " " + VZ.fmt(m, m < 1 ? 3 : 2) + " px");
    });
    gc.append("text").attr("x", 4).attr("y", 380).attr("font-size", 9.5).attr("fill", VC.muted).text("px, log axis");

    ro.html(
      `N = <b>${N}</b> correspondences, pixel noise <b>${VZ.fmt(sig, 2)} px</b>, <b>${TR}</b> independent trials, ` +
      `rank-2 enforcement <b>${enf ? "on" : "off"}</b> for both<br>` +
      `<b>unnormalised</b>: cond(A) = σ₁/σ₈ = <b>${cU.toExponential(3)}</b>, mean symmetric epipolar error = <b>${VZ.fmt(mU, 4)} px</b> ` +
      `(median ${VZ.fmt(d3.median(eU), 4)})<br>` +
      `<b>normalised</b>:&nbsp;&nbsp; cond(A) = σ₁/σ₈ = <b>${VZ.fmt(cN, 2)}</b>, mean symmetric epipolar error = <b>${VZ.fmt(mN, 4)} px</b> ` +
      `(median ${VZ.fmt(d3.median(eN), 4)})<br>` +
      `conditioning improved <b>${(cU / cN).toExponential(2)}×</b>; error improved <b>${VZ.fmt(mU / mN, 1)}×</b>. ` +
      `At <span class="keep">σ</span> = 0 both are exact — the columns above are still five decades apart, but with no noise there is nothing for bad conditioning to amplify.`
    );
  }

  [["e8-s", v => VZ.fmt(v, 2) + " px"], ["e8-n", v => "" + v], ["e8-t", v => "" + v]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  el("e8-r").addEventListener("change", draw);
  draw();
})();

/* ══════════ 7 · #enforce-svg — the rank-2 projection ════════════════════ */
(function () {
  const svg = d3.select("#enforce-svg");
  if (svg.empty()) return;
  const W = 760, H = 424, IW = 640, IH = 480;
  const ro = d3.select("#enforce-readout");
  const el = id => document.getElementById(id);
  const RIG = SS.rig({
    K0: VZ.K({ f: 500, cx: 320, cy: 240 }), K1: VZ.K({ f: 520, cx: 316, cy: 244 }),
    R1: VZ.mul(VZ.mul(VZ.Rz(VZ.rad(2)), VZ.Ry(VZ.rad(-8))), VZ.Rx(VZ.rad(3))), c1: [1, 0.05, 0.10]
  });
  const trueE = SS.epipoles(RIG.F);
  const CLOUD = (function () { const r = VZ.rng(7), o = []; for (let i = 0; i < 80; i++) o.push([-1.2 + 2.4 * r(), -0.9 + 1.8 * r(), 4 + 5 * r()]); return o; })();
  const SRC = d3.range(12).map(i => [60 + 520 * i / 11, 70 + 340 * ((i * 7) % 12) / 11]);

  function pencil(F) {
    const lp = SRC.map(p => VZ.mv(F, SS.hom(p)));
    const ii = [];
    for (let i = 0; i < lp.length; i++) for (let j = i + 1; j < lp.length; j++) {
      const x = VZ.cross(lp[i], lp[j]);
      if (Math.abs(x[2]) > 1e-12) ii.push([x[0] / x[2], x[1] / x[2]]);
    }
    return { lines: lp, ints: ii };
  }

  function draw() {
    const sig = +el("en-s").value, N = +el("en-n").value, TR = +el("en-t").value;
    const pts = CLOUD.slice(0, N);
    const p0 = pts.map(p => RIG.cam0.project(p).slice(0, 2));
    const p1 = pts.map(p => RIG.cam1.project(p).slice(0, 2));
    const r1 = VZ.rng(11);
    const a0 = p0.map(p => [p[0] + sig * VZ.randn(r1), p[1] + sig * VZ.randn(r1)]);
    const a1 = p1.map(p => [p[0] + sig * VZ.randn(r1), p[1] + sig * VZ.randn(r1)]);
    const f3 = SS.eightPoint(a0, a1, true, false), f2 = SS.eightPoint(a0, a1, true, true);
    const s3 = SS.svd(f3.F).s, s2 = SS.svd(f2.F).s;
    const P3 = pencil(f3.F), P2 = pencil(f2.F);
    const cx = d3.mean(P3.ints, p => p[0]), cy = d3.mean(P3.ints, p => p[1]);
    const dists = P3.ints.map(p => Math.hypot(p[0] - cx, p[1] - cy)).sort(d3.ascending);
    const medSpread = d3.quantile(dists, 0.5);
    const e2 = SS.epipoles(f2.F);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: singular values ─────────────────────────────────────────────── */
    const ga = VZ.panelBox(g, 46, 40, 150, 340, "singular values of F");
    const ya = d3.scaleLog().domain([1e-22, 3]).range([340, 0]).clamp(true);
    ga.append("g").attr("class", "axis").call(d3.axisLeft(ya).ticks(6, "~e"));
    const xa = d3.scaleBand().domain([0, 1, 2]).range([0, 150]).padding(0.3);
    [0, 1, 2].forEach(i => {
      ga.append("rect").attr("x", xa(i)).attr("y", ya(Math.max(s3[i], 1e-22))).attr("width", xa.bandwidth())
        .attr("height", 340 - ya(Math.max(s3[i], 1e-22))).attr("fill", VC.bad).attr("fill-opacity", 0.55);
      ga.append("rect").attr("x", xa(i) + 4).attr("y", ya(Math.max(s2[i], 1e-22))).attr("width", xa.bandwidth() - 8)
        .attr("height", 340 - ya(Math.max(s2[i], 1e-22))).attr("fill", VC.good).attr("fill-opacity", 0.9);
      ga.append("text").attr("x", xa(i) + xa.bandwidth() / 2).attr("y", 354).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", VC.muted).text("σ" + (i + 1));
    });
    VZ.legend(ga, [{ color: VC.bad, label: "raw (rank 3)" }, { color: VC.good, label: "enforced" }], 6, 14, { gap: 13, font: 9.5 });
    ga.append("text").attr("x", 0).attr("y", 372).attr("font-size", 9.5).attr("fill", VC.bad)
      .text("σ₃/σ₂ before = " + (s3[2] / s3[1]).toExponential(2));

    /* ── B: the two pencils, inside the frame ───────────────────────────── */
    const PS = 0.36, PW = IW * PS, PH = IH * PS;       // 230.4 × 172.8
    const gb = VZ.panelBox(g, 244, 40, PW, PH, "image 1 · both pencils, indistinguishable here", { fill: VC.bg });
    const cp = VZ.clip(svg, "en-clip", 0, 0, PW, PH);
    const inb = gb.append("g").attr("clip-path", cp);
    [[P3.lines, VC.bad, 1.4], [P2.lines, VC.violet, 0.9]].forEach(([ls, c, w]) => {
      ls.forEach(l => {
        const s = SS.clipLine(l, IW, IH);
        if (!s) return;
        inb.append("line").attr("x1", s[0][0] * PS).attr("y1", s[0][1] * PS)
          .attr("x2", s[1][0] * PS).attr("y2", s[1][1] * PS)
          .attr("stroke", c).attr("stroke-width", w).attr("stroke-opacity", 0.75);
      });
    });

    /* ── C: the epipole neighbourhood ───────────────────────────────────── */
    const R = Math.max(d3.max(dists) * 1.15, 40);
    const gc = VZ.panelBox(g, 244, 258, PW, 122, "the epipole neighbourhood, ±" + VZ.fmt(R, 0) + " px, centred " + VZ.fmt(Math.hypot(cx - 316, cy - 244), 0) + " px outside the frame", { fill: VC.bg });
    const sx = d3.scaleLinear().domain([cx - R, cx + R]).range([0, PW]);
    const sy = d3.scaleLinear().domain([cy - R * 122 / PW, cy + R * 122 / PW]).range([0, 122]);
    const cpc = VZ.clip(svg, "en-clip2", 0, 0, PW, 122);
    const inc = gc.append("g").attr("clip-path", cpc);
    P3.ints.forEach(p => inc.append("circle").attr("cx", sx(p[0])).attr("cy", sy(p[1])).attr("r", 2)
      .attr("fill", VC.bad).attr("fill-opacity", 0.75));
    if (e2.e1) {
      inc.append("circle").attr("cx", sx(e2.e1[0])).attr("cy", sy(e2.e1[1])).attr("r", 5.5)
        .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 2);
      inc.append("circle").attr("cx", sx(e2.e1[0])).attr("cy", sy(e2.e1[1])).attr("r", 2).attr("fill", VC.good);
    }
    if (trueE.e1) {
      inc.append("path").attr("d", `M${sx(trueE.e1[0]) - 6},${sy(trueE.e1[1])} h12 M${sx(trueE.e1[0])},${sy(trueE.e1[1]) - 6} v12`)
        .attr("stroke", VC.a2).attr("stroke-width", 1.6);
    }
    VZ.legend(gc, [{ color: VC.bad, label: "rank-3 pairwise meetings" }, { color: VC.good, label: "rank-2: one point" },
    { color: VC.a2, label: "true epipole" }], 6, 12, { gap: 12, font: 9 });

    /* ── D: error against noise, both variants ──────────────────────────── */
    const gd = VZ.panelBox(g, 534, 40, 196, 340, "mean symmetric epipolar error vs noise");
    const xs = d3.scaleLinear().domain([0, 2]).range([0, 196]);
    const ys = d3.scaleLinear().domain([0, 1.15]).range([340, 0]);
    VZ.gridY(gd, ys, 196, 5);
    gd.append("g").attr("class", "axis").attr("transform", "translate(0,340)").call(d3.axisBottom(xs).ticks(5));
    gd.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5));
    gd.append("text").attr("x", 196).attr("y", 334).attr("text-anchor", "end").attr("font-size", 9.5)
      .attr("fill", VC.muted).text("pixel noise σ");
    const c3 = [], c2 = [];
    let curA = 0, curB = 0;
    for (let s = 0; s <= 2.001; s += 0.2) {
      const rt = VZ.rng(31);
      let acc3 = 0, acc2 = 0;
      const M = Math.max(12, Math.round(TR / 6));
      for (let k = 0; k < M; k++) {
        const b0 = p0.map(p => [p[0] + s * VZ.randn(rt), p[1] + s * VZ.randn(rt)]);
        const b1 = p1.map(p => [p[0] + s * VZ.randn(rt), p[1] + s * VZ.randn(rt)]);
        acc3 += SS.symEpi(SS.eightPoint(b0, b1, true, false).F, p0, p1);
        acc2 += SS.symEpi(SS.eightPoint(b0, b1, true, true).F, p0, p1);
      }
      c3.push([xs(s), ys(acc3 / M)]); c2.push([xs(s), ys(acc2 / M)]);
    }
    VZ.poly(gd, c3, { stroke: VC.bad, w: 1.9, close: false });
    VZ.poly(gd, c2, { stroke: VC.good, w: 1.9, close: false, dash: "5 3" });
    gd.append("line").attr("x1", xs(sig)).attr("x2", xs(sig)).attr("y1", 0).attr("y2", 340)
      .attr("stroke", VC.a2).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
    VZ.legend(gd, [{ color: VC.bad, label: "no enforcement (rank 3)" }, { color: VC.good, label: "rank 2 enforced", dash: "5 3" }], 10, 16, { gap: 13, font: 9.5 });

    /* the measured cost, at the current noise */
    const rt2 = VZ.rng(77);
    let e3m = 0, e2m = 0;
    for (let k = 0; k < TR; k++) {
      const b0 = p0.map(p => [p[0] + sig * VZ.randn(rt2), p[1] + sig * VZ.randn(rt2)]);
      const b1 = p1.map(p => [p[0] + sig * VZ.randn(rt2), p[1] + sig * VZ.randn(rt2)]);
      e3m += SS.symEpi(SS.eightPoint(b0, b1, true, false).F, p0, p1);
      e2m += SS.symEpi(SS.eightPoint(b0, b1, true, true).F, p0, p1);
    }
    e3m /= TR; e2m /= TR;

    ro.html(
      `N = <b>${N}</b>, noise <b>${VZ.fmt(sig, 2)} px</b>, <b>${TR}</b> trials.&nbsp; ` +
      `σ₃/σ₂ before enforcement <b>${(s3[2] / s3[1]).toExponential(2)}</b> → after <b>${(s2[2] / Math.max(s2[1], 1e-300)).toExponential(2)}</b><br>` +
      `pencil concurrency: the ${P3.ints.length} pairwise meetings of 12 rank-3 lines have median spread <b>${VZ.fmt(medSpread, 1)} px</b> ` +
      `(largest ${VZ.fmt(d3.max(dists), 0)} px); after enforcement they coincide to <b>` +
      `${(function () { const dd = P2.ints.map(p => Math.hypot(p[0] - (e2.e1 ? e2.e1[0] : 0), p[1] - (e2.e1 ? e2.e1[1] : 0))); return d3.max(dd).toExponential(2); })()} px</b><br>` +
      `mean symmetric epipolar error: rank 3 <b>${VZ.fmt(e3m, 4)} px</b> → rank 2 <b>${VZ.fmt(e2m, 4)} px</b>, i.e. enforcement is ` +
      `<b>${VZ.fmt(100 * (e2m / e3m - 1), 1)}% worse</b> — and it is still the right thing to do, because only the second matrix is a fundamental matrix`
    );
  }

  [["en-s", v => VZ.fmt(v, 2) + " px"], ["en-n", v => "" + v], ["en-t", v => "" + v]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  draw();
})();

/* ══════════ 8 · #degen-svg — the configurations that cannot be solved ════ */
(function () {
  const svg = d3.select("#degen-svg");
  if (svg.empty()) return;
  const W = 760, H = 424, IW = 640, IH = 480;
  const ro = d3.select("#degen-readout");
  const el = id => document.getElementById(id);
  const KK = VZ.K({ f: 500, cx: 320, cy: 240 });
  const designRow = (a, b) => [b[0] * a[0], b[0] * a[1], b[0], b[1] * a[0], b[1] * a[1], b[1], a[0], a[1], 1];

  /* four configurations, all with 40 points and the same intrinsics */
  function config(which) {
    const r = VZ.rng(13), pts = [];
    let R = VZ.mul(VZ.Ry(VZ.rad(-8)), VZ.Rx(VZ.rad(3))), c1 = [1, 0.05, 0.10];
    if (which === "rot") c1 = [0, 0, 0];
    if (which === "tiny") { R = VZ.Ry(VZ.rad(-0.1)); c1 = [0.01, 0, 0]; }
    for (let i = 0; i < 40; i++) {
      const x = -1.5 + 3 * r(), y = -1 + 2 * r();
      /* the planar case is a VERTICAL plane, z = 6 + 0.35x, so the plan view below
         shows it as a single straight line — it is a plane in space all the same */
      const z = (which === "plane") ? 6 + 0.35 * x : 4 + 5 * r();
      pts.push([x, y, z]);
    }
    const rg = SS.rig({ K0: KK, K1: KK, R1: R, c1: c1 });
    return { pts: pts, rig: rg, c1: c1, which: which };
  }

  function draw() {
    const which = el("dg-c").value, alpha = +el("dg-a").value, sig = +el("dg-s").value;
    const C = config(which);
    const rn = VZ.rng(5);
    const p0 = C.pts.map(p => { const q = C.rig.cam0.project(p); return [q[0] + sig * VZ.randn(rn), q[1] + sig * VZ.randn(rn)]; });
    const p1 = C.pts.map(p => { const q = C.rig.cam1.project(p); return [q[0] + sig * VZ.randn(rn), q[1] + sig * VZ.randn(rn)]; });
    const T0 = SS.normaliseT(p0), T1 = SS.normaliseT(p1);
    const n0 = p0.map(p => VZ.applyH(T0, p)), n1 = p1.map(p => VZ.applyH(T1, p));
    const A = n0.map((p, i) => designRow(p, n1[i]));
    const { s, V } = SS.svd(A);
    const mixF = a => {
      const th = a * Math.PI / 2;
      const f = d3.range(9).map(k => Math.cos(th) * V[k][8] + Math.sin(th) * V[k][7]);
      let M = [[f[0], f[1], f[2]], [f[3], f[4], f[5]], [f[6], f[7], f[8]]];
      M = SS.rank2(M).F;
      M = VZ.mul(VZ.mul(VZ.T(T1), M), T0);
      let mx = 0; M.forEach(row => row.forEach(v => { mx = Math.max(mx, Math.abs(v)); }));
      return mx > 1e-300 ? M.map(row => row.map(v => v / mx)) : M;
    };
    const F = mixF(alpha);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: the plan view ────────────────────────────────────────────────── */
    const ga = VZ.panelBox(g, 40, 40, 200, 340, "plan view: looking down on the rig");
    const xs = d3.scaleLinear().domain([-2.2, 2.2]).range([0, 200]);
    const ys = d3.scaleLinear().domain([-0.6, 10]).range([340, 0]);
    VZ.gridY(ga, ys, 200, 5); VZ.gridX(ga, xs, 340, 5);
    ga.append("g").attr("class", "axis").attr("transform", "translate(0,340)").call(d3.axisBottom(xs).ticks(5));
    ga.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(6));
    ga.append("text").attr("x", 2).attr("y", -6).attr("font-size", 9.5).attr("fill", VC.muted).text("depth z (m) vs lateral x (m)");
    C.pts.forEach(p => ga.append("circle").attr("cx", xs(p[0])).attr("cy", ys(p[2])).attr("r", 2.4)
      .attr("fill", VC.violet).attr("fill-opacity", 0.85));
    [[[0, 0, 0], VC.accent, "c₀"], [C.c1, VC.a2, "c₁"]].forEach(([c, col, lb]) => {
      ga.append("path").attr("d", `M${xs(c[0])},${ys(c[2])} l-7,-11 l14,0 Z`).attr("fill", col);
      ga.append("text").attr("x", xs(c[0]) + 9).attr("y", ys(c[2]) + 4).attr("font-size", 10).attr("fill", col).text(lb);
    });
    ga.append("text").attr("x", 2).attr("y", 356).attr("font-size", 9.5).attr("fill", VC.muted)
      .text("baseline ‖c₁ − c₀‖ = " + VZ.fmt(VZ.norm(C.c1), 3) + " m");

    /* ── B: singular values of A ─────────────────────────────────────────── */
    const gb = VZ.panelBox(g, 288, 40, 170, 340, "singular values of A, ÷ σ₁");
    const yb = d3.scaleLog().domain([1e-18, 2]).range([340, 0]).clamp(true);
    gb.append("g").attr("class", "axis").call(d3.axisLeft(yb).ticks(7, "~e"));
    const xb = d3.scaleBand().domain(d3.range(9)).range([0, 170]).padding(0.25);
    const nullDim = s.filter(v => v / s[0] < 1e-10).length;
    s.forEach((v, i) => {
      const rr = Math.max(v / s[0], 1e-18);
      gb.append("rect").attr("x", xb(i)).attr("y", yb(rr)).attr("width", xb.bandwidth())
        .attr("height", 340 - yb(rr)).attr("fill", i >= 9 - Math.max(nullDim, 1) ? VC.bad : VC.accent)
        .attr("fill-opacity", 0.85);
      if (i >= 6) gb.append("text").attr("x", xb(i) + xb.bandwidth() / 2).attr("y", 354)
        .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", VC.muted).text(i + 1);
    });
    gb.append("text").attr("x", 0).attr("y", 372).attr("font-size", 10)
      .attr("fill", nullDim > 1 ? VC.bad : VC.good)
      .text("null space dimension " + Math.max(nullDim, 1) + (nullDim > 1 ? " — degenerate" : " — well posed"));

    /* ── C: the pencil this member of the family implies ─────────────────── */
    const PS = 0.375, PW = IW * PS, PH = IH * PS;
    const gc = VZ.panelBox(g, 500, 40, PW, PH, "image 1 · the pencil from F(α)", { fill: VC.bg });
    const cp = VZ.clip(svg, "dg-clip", 0, 0, PW, PH);
    const inc = gc.append("g").attr("clip-path", cp);
    p0.forEach(p => {
      const seg = SS.clipLine(VZ.mv(F, SS.hom(p)), IW, IH);
      if (!seg) return;
      inc.append("line").attr("x1", seg[0][0] * PS).attr("y1", seg[0][1] * PS)
        .attr("x2", seg[1][0] * PS).attr("y2", seg[1][1] * PS)
        .attr("stroke", VC.violet).attr("stroke-width", 0.8).attr("stroke-opacity", 0.55);
    });
    p1.forEach(p => inc.append("circle").attr("cx", p[0] * PS).attr("cy", p[1] * PS).attr("r", 1.8).attr("fill", VC.a2));
    const epF = SS.epipoles(F);
    if (epF.e1 && Math.abs(epF.e1[0]) < IW * 3 && Math.abs(epF.e1[1]) < IH * 3) {
      const ex = epF.e1[0] * PS, ey = epF.e1[1] * PS;
      if (ex > -3 && ex < PW + 3 && ey > -3 && ey < PH + 3)
        inc.append("circle").attr("cx", ex).attr("cy", ey).attr("r", 4.5).attr("fill", "none")
          .attr("stroke", VC.bad).attr("stroke-width", 1.8);
    }

    /* ── D: residual against the mixing parameter ────────────────────────── */
    const gd = VZ.panelBox(g, 500, 278, PW, 100, "worst symmetric epipolar residual over all 40 matches, vs α");
    const xd = d3.scaleLinear().domain([0, 1]).range([0, PW]);
    const yd = d3.scaleLog().domain([1e-14, 1e4]).range([100, 0]).clamp(true);
    gd.append("g").attr("class", "axis").attr("transform", "translate(0,100)").call(d3.axisBottom(xd).ticks(5));
    gd.append("g").attr("class", "axis").call(d3.axisLeft(yd).ticks(5, "~e"));
    const curve = [];
    let worstNow = 0;
    for (let a = 0; a <= 1.001; a += 0.02) {
      const Fa = mixF(a);
      let mxr = 0;
      for (let i = 0; i < p0.length; i++) {
        const l1 = SS.normLine(VZ.mv(Fa, SS.hom(p0[i])));
        mxr = Math.max(mxr, Math.abs(VZ.dot(SS.hom(p1[i]), l1)));
      }
      curve.push([xd(a), yd(Math.max(mxr, 1e-14))]);
      if (Math.abs(a - alpha) < 0.011) worstNow = mxr;
    }
    VZ.poly(gd, curve, { stroke: VC.a2, w: 1.8, close: false });
    gd.append("line").attr("x1", xd(alpha)).attr("x2", xd(alpha)).attr("y1", 0).attr("y2", 100)
      .attr("stroke", VC.good).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
    let mxr = 0;
    for (let i = 0; i < p0.length; i++) {
      const l1 = SS.normLine(VZ.mv(F, SS.hom(p0[i])));
      mxr = Math.max(mxr, Math.abs(VZ.dot(SS.hom(p1[i]), l1)));
    }

    const names = {
      plane: "planar scene", rot: "pure rotation", tiny: "1 cm baseline", gen: "general position"
    };
    const degenerate = nullDim > 1;
    ro.html(
      `<b>${names[which]}</b>, ${C.pts.length} points, noise ${VZ.fmt(sig, 2)} px.&nbsp; ` +
      `σ₇/σ₁ = <b>${(s[6] / s[0]).toExponential(2)}</b>, σ₈/σ₁ = <b>${(s[7] / s[0]).toExponential(2)}</b>, ` +
      `σ₉/σ₁ = ${(s[8] / s[0]).toExponential(2)} → null space of dimension <b>${Math.max(nullDim, 1)}</b><br>` +
      `at <span class="keep">α</span> = ${VZ.fmt(alpha, 2)} the worst residual over all 40 matches is <b>${mxr.toExponential(2)} px</b>` +
      (degenerate
        ? ` — and it stays there for <i>every</i> <span class="keep">α</span>. Slide the mixer: the epipolar geometry changes completely and the fit does not change at all. ` +
        `<b>The data cannot tell you which member of the family is right, because they are all exactly right.</b>`
        : ` — and it climbs steeply as soon as <span class="keep">α</span> leaves 0. That climb is what a well-posed problem looks like: only one member of the family fits.`)
    );
  }

  el("dg-c").addEventListener("change", draw);
  [["dg-a", v => VZ.fmt(v, 2)], ["dg-s", v => VZ.fmt(v, 2) + " px"]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  draw();
})();

/* ══════════ 9 · #decomp-svg — four candidates, three killed ═════════════ */
(function () {
  const svg = d3.select("#decomp-svg");
  if (svg.empty()) return;
  const W = 760, H = 424;
  const ro = d3.select("#decomp-readout");
  const el = id => document.getElementById(id);
  const KK = VZ.K({ f: 500, cx: 320, cy: 240 });
  const Rtrue = VZ.mul(VZ.Ry(VZ.rad(-8)), VZ.Rx(VZ.rad(3)));
  const c1 = [1, 0.05, 0.10];
  const Wm = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];

  function decompose(E) {
    let { U, s, V } = SS.svd(E);
    /* fix both to proper rotations */
    if (VZ.det3(U) < 0) U = U.map(r => r.map((v, j) => j === 2 ? -v : v));
    if (VZ.det3(V) < 0) V = V.map(r => r.map((v, j) => j === 2 ? -v : v));
    const Ra = VZ.mul(VZ.mul(U, Wm), VZ.T(V));
    const Rb = VZ.mul(VZ.mul(U, VZ.T(Wm)), VZ.T(V));
    const th = SS.col(U, 2);
    return [{ R: Ra, t: th, lab: "R_a, +t̂" }, { R: Ra, t: VZ.neg(th), lab: "R_a, −t̂" },
    { R: Rb, t: th, lab: "R_b, +t̂" }, { R: Rb, t: VZ.neg(th), lab: "R_b, −t̂" }]
      .map(c => { if (VZ.det3(c.R) < 0) c.R = c.R.map(r => r.map(v => -v)); return c; });
  }

  function draw() {
    const sig = +el("dc-s").value, zmax = +el("dc-z").value, K = +el("dc-k").value;
    const r = VZ.rng(13), pts = [];
    for (let i = 0; i < 40; i++) pts.push([-1.5 + 3 * r(), -1 + 2 * r(), 4 + (zmax - 4) * r()]);
    const rig = SS.rig({ K0: KK, K1: KK, R1: Rtrue, c1: c1 });
    const rn = VZ.rng(7);
    const p0 = pts.map(p => { const q = rig.cam0.project(p); return [q[0] + sig * VZ.randn(rn), q[1] + sig * VZ.randn(rn)]; });
    const p1 = pts.map(p => { const q = rig.cam1.project(p); return [q[0] + sig * VZ.randn(rn), q[1] + sig * VZ.randn(rn)]; });
    /* work in normalised rays; the essential matrix is scaled to (1, 1, 0) */
    const Ki = VZ.inv3(KK);
    const n0 = p0.map(p => { const v = VZ.mv(Ki, SS.hom(p)); return [v[0] / v[2], v[1] / v[2]]; });
    const n1 = p1.map(p => { const v = VZ.mv(Ki, SS.hom(p)); return [v[0] / v[2], v[1] / v[2]]; });
    const raw = SS.eightPoint(n0, n1, true, true).F;
    /* project onto the essential manifold: (a, a, 0) with a the mean of σ₁, σ₂ */
    const sv = SS.svd(raw);
    const a = (sv.s[0] + sv.s[1]) / 2;
    const E = VZ.mul(VZ.mul(sv.U, [[a, 0, 0], [0, a, 0], [0, 0, 0]]), VZ.T(sv.V));
    const cands = decompose(E);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    const cam0 = VZ.camera({ K: VZ.eye(3), R: VZ.eye(3), t: [0, 0, 0] });
    const counts = [], allPts = [];
    cands.forEach((c) => {
      const camB = VZ.camera({ K: VZ.eye(3), R: c.R, t: c.t });
      const P = [], front = [];
      for (let i = 0; i < 40; i++) {
        const X = SS.triDLT(cam0, camB, n0[i], n1[i]);
        if (!X) { P.push(null); front.push(false); continue; }
        const z1 = VZ.add(VZ.mv(c.R, X), c.t)[2];
        P.push(X); front.push(X[2] > 0 && z1 > 0);
      }
      allPts.push({ P: P, front: front, cam: camB, c: c });
      counts.push(front.slice(0, K).filter(Boolean).length);
    });
    const best = counts.indexOf(d3.max(counts));

    const PWp = 168, PHp = 300;
    cands.forEach((c, k) => {
      const x = 26 + k * 184;
      const win = best === k;
      const gg = VZ.panelBox(g, x, 44, PWp, PHp, c.lab, { stroke: win ? VC.good : VC.line });
      /* plan view, autoscaled to the reconstructed cloud */
      const A = allPts[k];
      const zs = A.P.filter(Boolean).map(p => p[2]), xsv = A.P.filter(Boolean).map(p => p[0]);
      const zlo = Math.min(-1, d3.min(zs) || -1), zhi = Math.max(1, d3.max(zs) || 1);
      const xlo = Math.min(-1, d3.min(xsv) || -1), xhi = Math.max(1, d3.max(xsv) || 1);
      const sx = d3.scaleLinear().domain([xlo * 1.1, xhi * 1.1]).range([0, PWp]);
      const sy = d3.scaleLinear().domain([zlo * 1.15, zhi * 1.15]).range([PHp, 0]);
      gg.append("line").attr("x1", 0).attr("x2", PWp).attr("y1", sy(0)).attr("y2", sy(0))
        .attr("stroke", VC.line).attr("stroke-dasharray", "3 3");
      A.P.forEach((p, i) => {
        if (!p) return;
        const cxp = sx(p[0]), cyp = sy(p[2]);
        if (cxp < -4 || cxp > PWp + 4 || cyp < -4 || cyp > PHp + 4) return;
        gg.append("circle").attr("cx", cxp).attr("cy", cyp).attr("r", 2)
          .attr("fill", A.front[i] ? VC.good : VC.bad).attr("fill-opacity", i < K ? 0.9 : 0.25);
      });
      /* the two cameras, as little wedges pointing along their optical axes */
      const drawCam = (C, R, col) => {
        const px = sx(C[0]), py = sy(C[2]);
        if (px < -20 || px > PWp + 20 || py < -20 || py > PHp + 20) return;
        const f = VZ.mv(VZ.T(R), [0, 0, 1]);
        const ang = Math.atan2(-(sy(C[2] + f[2]) - py), sx(C[0] + f[0]) - px);
        gg.append("path").attr("d", "M0,0 L-9,-16 L9,-16 Z")
          .attr("transform", `translate(${px},${py}) rotate(${90 - VZ.deg(ang)})`)
          .attr("fill", col).attr("fill-opacity", 0.9);
      };
      drawCam([0, 0, 0], VZ.eye(3), VC.accent);
      const Cb = VZ.neg(VZ.mv(VZ.T(c.R), c.t));
      drawCam(Cb, c.R, VC.a2);
      gg.append("text").attr("x", 4).attr("y", PHp + 15).attr("font-size", 10.5)
        .attr("fill", win ? VC.good : VC.muted).attr("font-weight", win ? 600 : 400)
        .text(counts[k] + " / " + K + " in front of both");
    });

    /* the reference: how close is the winner to the truth? */
    const bw = cands[best];
    const dR = (function () { const M = VZ.mul(bw.R, VZ.T(Rtrue)); return VZ.deg(VZ.axisAngleFromR(M).angle); })();
    const ttrue = VZ.unit(VZ.neg(VZ.mv(Rtrue, c1)));
    const dt = VZ.deg(Math.acos(VZ.clamp(Math.abs(VZ.dot(VZ.unit(bw.t), ttrue)), -1, 1)));
    const others = counts.map((v, i) => i === best ? -1 : v);
    ro.html(
      `noise <b>${VZ.fmt(sig, 1)} px</b>, scene depth 4–<b>${zmax} m</b>, vote over <b>${K}</b> points.&nbsp; ` +
      `counts in front of both cameras: <b>${counts.join(" · ")}</b> for (R_a,+t̂), (R_a,−t̂), (R_b,+t̂), (R_b,−t̂)<br>` +
      `winner: <b>${cands[best].lab}</b>, runner-up ${d3.max(others)} — a margin of <b>${d3.max(counts) - d3.max(others)}</b> votes<br>` +
      `the winning pose differs from the truth by <b>${VZ.fmt(dR, 3)}°</b> of rotation and <b>${VZ.fmt(dt, 3)}°</b> in translation direction ` +
      `(the length is unrecoverable). ` +
      (d3.max(counts) === K
        ? `Every point agrees — which is what a well-conditioned scene looks like.`
        : `<b>${K - d3.max(counts)} of ${K} points vote against the winner</b>: their triangulated depth uncertainty is comparable to their depth, so their sign is noise.`) +
      `<br>This is <i>one draw</i> of the noise. Over many draws at half a pixel and this rig, the median errors are about 1.2° in rotation ` +
      `and 2.8° in translation direction — the linear estimate is only ever a starting point, which is what §27 is for.`
    );
  }

  [["dc-s", v => VZ.fmt(v, 1) + " px"], ["dc-z", v => v + " m"], ["dc-k", v => "" + v]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  draw();
})();

/* ══════════ 10 · #tri-svg — three triangulation methods ═════════════════ */
(function () {
  const svg = d3.select("#tri-svg");
  if (svg.empty()) return;
  const W = 760, H = 404;
  const ro = d3.select("#tri-readout");
  const el = id => document.getElementById(id);
  const K0 = VZ.K({ f: 500, cx: 320, cy: 240 }), K1 = VZ.K({ f: 520, cx: 316, cy: 244 });
  const R1 = VZ.mul(VZ.mul(VZ.Rz(VZ.rad(2)), VZ.Ry(VZ.rad(-8))), VZ.Rx(VZ.rad(3)));
  const DIR = VZ.unit([1, 0.05, 0.10]);

  function draw() {
    const B = +el("tr-b").value, zmax = +el("tr-z").value, sig = +el("tr-s").value, TR = +el("tr-t").value;
    const RIG = SS.rig({ K0: K0, K1: K1, R1: R1, c1: VZ.scale(DIR, B) });
    let m = 0; RIG.F.forEach(r => r.forEach(v => { m = Math.max(m, Math.abs(v)); }));
    const F = RIG.F.map(r => r.map(v => v / m));
    const rr = VZ.rng(7), pts = [];
    for (let i = 0; i < 40; i++) pts.push([-1.2 + 2.4 * rr(), -0.9 + 1.8 * rr(), 4 + (zmax - 4) * rr()]);

    const reproj = (X, u0, u1) => {
      const a = RIG.cam0.project(X), b = RIG.cam1.project(X);
      return (a[0] - u0[0]) ** 2 + (a[1] - u0[1]) ** 2 + (b[0] - u1[0]) ** 2 + (b[1] - u1[1]) ** 2;
    };
    const rn = VZ.rng(31);
    const e = [0, 0, 0], c = [0, 0, 0];
    let n = 0, demo = null;
    for (let k = 0; k < TR; k++) {
      for (let i = 0; i < 40; i++) {
        const a = RIG.cam0.project(pts[i]), b = RIG.cam1.project(pts[i]);
        const u0 = [a[0] + sig * VZ.randn(rn), a[1] + sig * VZ.randn(rn)];
        const u1 = [b[0] + sig * VZ.randn(rn), b[1] + sig * VZ.randn(rn)];
        const X = [SS.triMid(RIG.cam0, RIG.cam1, u0, u1), SS.triDLT(RIG.cam0, RIG.cam1, u0, u1),
        SS.triOptimal(RIG.cam0, RIG.cam1, u0, u1, F)];
        X.forEach((Xi, j) => {
          if (!Xi) return;
          e[j] += VZ.norm(VZ.sub(Xi, pts[i])) ** 2;
          c[j] += reproj(Xi, u0, u1);
        });
        n++;
        if (k === 0 && i === 5) demo = { u0: u0, u1: u1, X: X };
      }
    }
    const rms = e.map(v => Math.sqrt(v / n)), cost = c.map(v => v / n);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: the 1-D objective over the pencil ───────────────────────────── */
    const ga = VZ.panelBox(g, 52, 40, 286, 320, "the objective s(θ) over the pencil, one correspondence");
    const ep = SS.epipoles(F);
    const sOf = th => {
      const d0 = [Math.cos(th), Math.sin(th)];
      const l0 = SS.normLine([-d0[1], d0[0], d0[1] * ep.e0[0] - d0[0] * ep.e0[1]]);
      const q = [ep.e0[0] + 40 * d0[0], ep.e0[1] + 40 * d0[1]];
      const l1 = SS.normLine(VZ.mv(F, SS.hom(q)));
      const A = SS.ptLine(l0, demo.u0), Bq = SS.ptLine(l1, demo.u1);
      return A * A + Bq * Bq;
    };
    const samples = d3.range(0, 721).map(i => { const th = Math.PI * i / 720; return [th, sOf(th)]; });
    const smin = d3.min(samples, p => p[1]), smax = d3.max(samples, p => p[1]);
    const xs = d3.scaleLinear().domain([0, Math.PI]).range([0, 286]);
    const ys = d3.scaleLog().domain([Math.max(smin * 0.4, 1e-6), Math.max(smax, 1e-5)]).range([320, 0]).clamp(true);
    VZ.gridY(ga, ys, 286, 4);
    ga.append("g").attr("class", "axis").attr("transform", "translate(0,320)")
      .call(d3.axisBottom(xs).ticks(4).tickFormat(d => VZ.fmt(VZ.deg(d), 0) + "°"));
    ga.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5, "~e"));
    VZ.poly(ga, samples.map(p => [xs(p[0]), ys(Math.max(p[1], 1e-9))]), { stroke: VC.violet, w: 1.8, close: false });
    const dcost = demo.X.map(Xi => Xi ? reproj(Xi, demo.u0, demo.u1) : NaN);
    [[dcost[0], VC.bad, "mid-point"], [dcost[1], VC.accent, "linear DLT"], [dcost[2], VC.good, "optimal"]]
      .forEach(([v, col, lb], i) => {
        if (!isFinite(v)) return;
        ga.append("line").attr("x1", 0).attr("x2", 286).attr("y1", ys(v)).attr("y2", ys(v))
          .attr("stroke", col).attr("stroke-width", 1.3).attr("stroke-dasharray", "5 3");
        ga.append("text").attr("x", 4).attr("y", ys(v) - 4).attr("font-size", 9.5).attr("fill", col)
          .text(lb + " " + VZ.fmt(v, 4) + " px²");
      });
    ga.append("text").attr("x", 2).attr("y", -6).attr("font-size", 9.5).attr("fill", VC.muted)
      .text("cost, px² (log) vs orientation of the epipolar plane");

    /* ── B: 3-D error ───────────────────────────────────────────────────── */
    const gb = VZ.panelBox(g, 396, 40, 150, 320, "3-D RMS error (m)");
    const yb = d3.scaleLinear().domain([0, d3.max(rms) * 1.2]).range([320, 0]);
    gb.append("g").attr("class", "axis").call(d3.axisLeft(yb).ticks(5));
    const xb = d3.scaleBand().domain([0, 1, 2]).range([0, 150]).padding(0.28);
    const cols = [VC.bad, VC.accent, VC.good], labs = ["mid", "DLT", "opt"];
    const bestR = rms.indexOf(d3.min(rms));
    rms.forEach((v, i) => {
      gb.append("rect").attr("x", xb(i)).attr("y", yb(v)).attr("width", xb.bandwidth()).attr("height", 320 - yb(v))
        .attr("fill", cols[i]).attr("fill-opacity", i === bestR ? 0.95 : 0.5);
      gb.append("text").attr("x", xb(i) + xb.bandwidth() / 2).attr("y", 334).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", VC.muted).text(labs[i]);
      gb.append("text").attr("x", xb(i) + xb.bandwidth() / 2).attr("y", yb(v) - 5).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", VC.ink).text(VZ.sig(v, 4));
    });

    /* ── C: reprojection cost, zoomed ───────────────────────────────────── */
    const gc = VZ.panelBox(g, 594, 40, 150, 320, "mean reprojection cost (px²)");
    const lo = d3.min(cost) * 0.985, hi = d3.max(cost) * 1.005;
    const yc = d3.scaleLinear().domain([lo, hi]).range([320, 0]);
    gc.append("g").attr("class", "axis").call(d3.axisLeft(yc).ticks(5, "~f"));
    cost.forEach((v, i) => {
      gc.append("rect").attr("x", xb(i)).attr("y", yc(v)).attr("width", xb.bandwidth()).attr("height", Math.max(1, 320 - yc(v)))
        .attr("fill", cols[i]).attr("fill-opacity", i === 2 ? 0.95 : 0.5);
      gc.append("text").attr("x", xb(i) + xb.bandwidth() / 2).attr("y", 334).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", VC.muted).text(labs[i]);
      gc.append("text").attr("x", xb(i) + xb.bandwidth() / 2).attr("y", yc(v) - 5).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", VC.ink).text(VZ.fmt(v, 4));
    });
    gc.append("text").attr("x", 0).attr("y", 350).attr("font-size", 9).attr("fill", VC.muted).text("axis zoomed — note the origin is not 0");

    const inverted = bestR === 0 && cost[0] === d3.max(cost) && rms[1] / rms[0] > 1.25;
    ro.html(
      `baseline <b>${VZ.fmt(B, 2)} m</b>, scene 4–<b>${zmax} m</b>, noise <b>${VZ.fmt(sig, 2)} px</b>, ` +
      `<b>${n}</b> triangulations<br>` +
      `3-D RMS error: mid-point <b>${VZ.sig(rms[0], 5)}</b> · linear <b>${VZ.sig(rms[1], 5)}</b> · optimal <b>${VZ.sig(rms[2], 5)}</b> m<br>` +
      `mean reprojection cost: mid-point <b>${VZ.fmt(cost[0], 5)}</b> · linear <b>${VZ.fmt(cost[1], 5)}</b> · optimal <b>${VZ.fmt(cost[2], 5)}</b> px² ` +
      `— always in that order, because the third one minimises exactly this<br>` +
      (inverted
        ? `<b>The inversion is live at these settings:</b> the mid-point estimate is the best in 3-D (by ${VZ.fmt(rms[1] / rms[0], 2)}×) and the worst by reprojection (by ${VZ.fmt(cost[0] / cost[2], 2)}×). Minimising pixels is not minimising metres.`
        : `Here the three agree in 3-D to within <b>${VZ.fmt(100 * (d3.max(rms) / d3.min(rms) - 1), 2)}%</b> — the choice of method barely matters. Shrink the baseline towards 0.15 m and push the scene out past 20 m to break that.`)
    );
  }

  [["tr-b", v => VZ.fmt(v, 2) + " m"], ["tr-z", v => v + " m"], ["tr-s", v => VZ.fmt(v, 2) + " px"], ["tr-t", v => "" + v]]
    .forEach(([id, f]) => {
      el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
      el(id + "v").textContent = f(+el(id).value);
    });
  draw();
})();

/* ══════════ 11 · #rect-svg — rectification ══════════════════════════════ */
(function () {
  const svg = d3.select("#rect-svg");
  if (svg.empty()) return;
  const W = 760, H = 452, IW = 640, IH = 480;
  const ro = d3.select("#rect-readout");
  const el = id => document.getElementById(id);
  const PS = 0.325, PW = IW * PS, PH = IH * PS;        // 208 × 156
  const K0 = VZ.K({ f: 500, cx: 320, cy: 240 }), K1 = VZ.K({ f: 520, cx: 316, cy: 244 });
  const CLOUD = (function () { const r = VZ.rng(19), o = []; for (let i = 0; i < 18; i++) o.push([-1.2 + 2.4 * r(), -0.9 + 1.8 * r(), 4 + 5 * r()]); return o; })();

  /* the rectifying pair, for a given rig */
  function rectify(rig) {
    const c0 = [0, 0, 0], c1 = rig.c1;
    const b = VZ.sub(c1, c0);
    if (VZ.norm(b) < 1e-9) return null;
    const r1 = VZ.unit(b);
    const k = [0, 0, 1];                                // camera 0's optical axis, in world
    const cr = VZ.cross(k, r1);
    if (VZ.norm(cr) < 1e-7) return null;                // baseline along the optical axis
    const r2 = VZ.unit(cr), r3 = VZ.cross(r1, r2);
    const Rr = [r1, r2, r3];
    const Kn = [[(K0[0][0] + K1[0][0]) / 2, 0, (K0[0][2] + K1[0][2]) / 2],
    [0, (K0[1][1] + K1[1][1]) / 2, (K0[1][2] + K1[1][2]) / 2], [0, 0, 1]];
    const H0 = VZ.mul(VZ.mul(Kn, Rr), VZ.inv3(VZ.mul(K0, VZ.eye(3))));
    const H1 = VZ.mul(VZ.mul(Kn, Rr), VZ.inv3(VZ.mul(K1, rig.R)));
    return { H0: H0, H1: H1, Kn: Kn, Rr: Rr, B: VZ.norm(b) };
  }

  /* The area the warped image needs, ÷ the original area.
     A homography sends one whole LINE to infinity — the line whose coefficients are
     the third row of H, since q₃ = h₃·x̄ is the vanishing denominator. If that line
     crosses the image rectangle, part of the picture maps to infinity and the
     rectified canvas is genuinely unbounded; the bounding box of the four warped
     CORNERS is then a meaningless underestimate, which is the trap this guards. */
  function blowup(H) {
    if (SS.clipLine([H[2][0], H[2][1], H[2][2]], IW, IH)) return Infinity;
    const cs = [[0, 0], [IW, 0], [IW, IH], [0, IH]].map(p => {
      const q = VZ.mv(H, SS.hom(p));
      return Math.abs(q[2]) < 1e-9 ? null : [q[0] / q[2], q[1] / q[2]];
    });
    if (cs.some(c => !c)) return Infinity;
    const xs = cs.map(c => c[0]), ys = cs.map(c => c[1]);
    const a = (d3.max(xs) - d3.min(xs)) * (d3.max(ys) - d3.min(ys));
    return a / (IW * IH);
  }

  function makeRig(verg, cz) {
    const R = VZ.mul(VZ.Ry(VZ.rad(-verg)), VZ.Rx(VZ.rad(3)));
    return SS.rig({ K0: K0, K1: K1, R1: R, c1: [1, 0.05, cz] });
  }

  function draw() {
    const verg = +el("rc-v").value, cz = +el("rc-z").value, mode = el("rc-m").value;
    const rig = makeRig(verg, cz);
    const rec = rectify(rig);
    let mx = 0; rig.F.forEach(r => r.forEach(v => { mx = Math.max(mx, Math.abs(v)); }));
    const F = rig.F.map(r => r.map(v => v / mx));
    const x0 = CLOUD.map(p => rig.cam0.project(p).slice(0, 2));
    const x1 = CLOUD.map(p => rig.cam1.project(p).slice(0, 2));
    const ep = SS.epipoles(F);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    const showOrig = mode === "both";
    const rows = showOrig ? [0, 1] : [1];
    const yTop = showOrig ? 40 : 130;

    /* the rectified points, computed once so the row check can use them */
    let y0 = null, y1 = null, rowErr = NaN;
    if (rec) {
      y0 = x0.map(p => VZ.applyH(rec.H0, p));
      y1 = x1.map(p => VZ.applyH(rec.H1, p));
      rowErr = d3.max(y0.map((p, i) => Math.abs(p[1] - y1[i][1])));
    }

    function panel(px, py, title, pts, lines, warped) {
      const gg = VZ.panelBox(g, px, py, PW, PH, title, { fill: VC.bg });
      const id = "rc-" + px + "-" + py;
      const cp = VZ.clip(svg, id, 0, 0, PW, PH);
      const inner = gg.append("g").attr("clip-path", cp);
      /* when rectified, draw in the ORIGINAL image's pixel box so the panels align;
         the warp is only used to place the points and lines */
      lines.forEach(l => {
        const seg = SS.clipLine(l, IW, IH);
        if (!seg) return;
        inner.append("line").attr("x1", seg[0][0] * PS).attr("y1", seg[0][1] * PS)
          .attr("x2", seg[1][0] * PS).attr("y2", seg[1][1] * PS)
          .attr("stroke", warped ? VC.good : VC.violet).attr("stroke-width", 0.9).attr("stroke-opacity", 0.7);
      });
      pts.forEach(p => inner.append("circle").attr("cx", p[0] * PS).attr("cy", p[1] * PS).attr("r", 2.2)
        .attr("fill", VC.a2));
      return gg;
    }

    if (showOrig) {
      panel(24, 40, "image 0 · original", x0, x1.map(p => VZ.mv(VZ.T(F), SS.hom(p))), false);
      panel(258, 40, "image 1 · original", x1, x0.map(p => VZ.mv(F, SS.hom(p))), false);
    }
    if (rec) {
      /* rectified fundamental matrix: the canonical [[0,0,0],[0,0,1],[0,-1,0]] */
      const Fr = [[0, 0, 0], [0, 0, 1], [0, -1, 0]];
      const g0 = panel(24, yTop + (showOrig ? 212 : 0), "image 0 · rectified", y0, y1.map(p => VZ.mv(VZ.T(Fr), SS.hom(p))), true);
      const g1 = panel(258, yTop + (showOrig ? 212 : 0), "image 1 · rectified — same rows", y1, y0.map(p => VZ.mv(Fr, SS.hom(p))), true);
      /* faint rules joining matched rows across the two rectified panels */
      y0.forEach((p, i) => {
        const yy = p[1] * PS;
        if (yy < 0 || yy > PH) return;
        g0.append("line").attr("x1", 0).attr("x2", PW + 26).attr("y1", yy).attr("y2", yy)
          .attr("stroke", VC.good).attr("stroke-width", 0.5).attr("stroke-opacity", 0.28);
      });
    } else {
      g.append("text").attr("x", 24).attr("y", yTop + (showOrig ? 300 : 90)).attr("font-size", 12).attr("fill", VC.bad)
        .text("rectification undefined: the baseline is along the optical axis");
    }

    /* ── the blow-up curve ──────────────────────────────────────────────── */
    const gc = VZ.panelBox(g, 526, 40, 210, 340, "rectified canvas area ÷ original");
    const xs = d3.scaleLinear().domain([0, 3]).range([0, 210]);
    const ys = d3.scaleLog().domain([0.5, 1e5]).range([340, 0]).clamp(true);
    gc.append("g").attr("class", "axis").attr("transform", "translate(0,340)").call(d3.axisBottom(xs).ticks(4));
    gc.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(6, "~e"));
    gc.append("text").attr("x", 210).attr("y", 334).attr("text-anchor", "end").attr("font-size", 9.5)
      .attr("fill", VC.muted).text("forward travel c₁_z (m)");
    const curve = [];
    let enterZ = null;
    for (let z = 0; z <= 3.0001; z += 0.02) {
      const rg = makeRig(verg, z), rc = rectify(rg);
      if (!rc) continue;
      const a = Math.max(blowup(rc.H0), blowup(rc.H1));
      curve.push([xs(z), ys(VZ.clamp(a, 0.5, 1e5))]);
      if (enterZ === null && !isFinite(a)) enterZ = z;
    }
    VZ.poly(gc, curve, { stroke: VC.a2, w: 1.9, close: false });
    if (enterZ !== null) {
      gc.append("line").attr("x1", xs(enterZ)).attr("x2", xs(enterZ)).attr("y1", 0).attr("y2", 340)
        .attr("stroke", VC.bad).attr("stroke-width", 1.3).attr("stroke-dasharray", "4 3");
      gc.append("rect").attr("x", xs(enterZ)).attr("y", 0).attr("width", 210 - xs(enterZ)).attr("height", 340)
        .attr("fill", VC.bad).attr("fill-opacity", 0.08);
      gc.append("text").attr("x", VZ.clamp(xs(enterZ) + 4, 0, 120)).attr("y", 14).attr("font-size", 9).attr("fill", VC.bad)
        .text("unbounded beyond " + VZ.fmt(enterZ, 2) + " m");
    }
    const nowA = rec ? Math.max(blowup(rec.H0), blowup(rec.H1)) : Infinity;
    if (isFinite(nowA)) {
      gc.append("circle").attr("cx", xs(cz)).attr("cy", ys(VZ.clamp(nowA, 0.5, 1e5))).attr("r", 4.5).attr("fill", VC.good);
    }
    gc.append("line").attr("x1", 0).attr("x2", 210).attr("y1", ys(1)).attr("y2", ys(1))
      .attr("stroke", VC.line).attr("stroke-dasharray", "2 3");

    const insideNow = ep.e0 && ep.e0[0] > 0 && ep.e0[0] < IW && ep.e0[1] > 0 && ep.e0[1] < IH;
    ro.html(rec
      ? `vergence <b>${VZ.fmt(verg, 1)}°</b>, forward travel <b>${VZ.fmt(cz, 2)} m</b>, baseline <b>${VZ.fmt(rec.B, 4)} m</b>.&nbsp; ` +
      `epipole e₀ = ${ep.e0 ? "(" + VZ.fmt(ep.e0[0], 0) + ", " + VZ.fmt(ep.e0[1], 0) + ")" : "at infinity"} px, ` +
      `<b>${insideNow ? "INSIDE the frame — rectification is hopeless here" : "outside the frame"}</b><br>` +
      `row disagreement after rectification: max |v′₀ − v′₁| = <b>${rowErr.toExponential(2)} px</b> over ${CLOUD.length} correspondences ` +
      `— machine precision, not an approximation<br>` +
      `rectified canvas needed: <b>${isFinite(nowA) ? VZ.fmt(nowA, 2) + "×" : "UNBOUNDED"}</b> the original image area` +
      (isFinite(nowA) ? (nowA > 6 ? ` — already impractical.` : `.`)
        : ` — the line that the warp sends to infinity now crosses the picture, so part of the image has nowhere finite to go. This is the failure, and no choice of K_new repairs it.`)
      : `<b>Rectification is undefined at these settings.</b> The baseline is parallel to the optical axis, so k × r₁ = 0 and the ` +
      `new y-axis has no definition. This is the pure forward-motion case; use a plane sweep instead.`);
  }

  [["rc-v", v => VZ.fmt(v, 1) + "°"], ["rc-z", v => VZ.fmt(v, 2) + " m"]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  el("rc-m").addEventListener("change", draw);
  draw();
})();

/* ══════════ 12 · #depth-svg — Z = fB/d and the Z² error law ═════════════ */
(function () {
  const svg = d3.select("#depth-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const ro = d3.select("#depth-readout");
  const el = id => document.getElementById(id);
  const DEPTHS = [1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40];
  const NS = 40000;

  function draw() {
    const f = +el("dp-f").value, B = +el("dp-b").value, sd = +el("dp-s").value, tol = +el("dp-t").value / 100;
    const fB = f * B;
    /* the live Monte Carlo */
    const r = VZ.rng(101);
    const meas = DEPTHS.map(Z => {
      const d = fB / Z;
      let s = 0, s2 = 0, n = 0;
      for (let i = 0; i < NS; i++) {
        const dn = d + sd * VZ.randn(r);
        if (dn <= 1e-6) continue;                 // a non-positive disparity has no depth
        const Zi = fB / dn;
        s += Zi; s2 += Zi * Zi; n++;
      }
      const m = s / n, v = Math.max(s2 / n - m * m, 0);
      return { Z: Z, d: d, sd: Math.sqrt(v), pred: Z * Z * sd / fB, frac: n / NS };
    });
    /* fitted log–log slope over the range where the linearisation holds */
    const good = meas.filter(m => m.d > 8 * sd && m.sd > 0);
    let slope = NaN, slopeAll = NaN;
    const fit = arr => {
      if (arr.length < 3) return NaN;
      const lx = arr.map(m => Math.log(m.Z)), ly = arr.map(m => Math.log(m.sd));
      const mx = d3.mean(lx), my = d3.mean(ly);
      let num = 0, den = 0;
      lx.forEach((x, i) => { num += (x - mx) * (ly[i] - my); den += (x - mx) * (x - mx); });
      return num / den;
    };
    slope = fit(good); slopeAll = fit(meas.filter(m => m.sd > 0));
    /* the range at which the relative error hits the tolerance */
    const Zmax = tol * fB / sd;

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: the hyperbola and its depth quantisation ────────────────────── */
    const ga = VZ.panelBox(g, 46, 40, 200, 330, "Z = fB/d, and the depth levels of whole-pixel d");
    const xa = d3.scaleLinear().domain([0, Math.max(6, Math.min(80, fB / 0.6))]).range([0, 200]);
    const ya = d3.scaleLinear().domain([0, 45]).range([330, 0]);
    VZ.gridY(ga, ya, 200, 5);
    ga.append("g").attr("class", "axis").attr("transform", "translate(0,330)").call(d3.axisBottom(xa).ticks(5));
    ga.append("g").attr("class", "axis").call(d3.axisLeft(ya).ticks(6));
    ga.append("text").attr("x", 200).attr("y", 324).attr("text-anchor", "end").attr("font-size", 9.5)
      .attr("fill", VC.muted).text("disparity d (px)");
    ga.append("text").attr("x", 2).attr("y", -6).attr("font-size", 9.5).attr("fill", VC.muted).text("depth Z (m)");
    const hyp = [];
    for (let dd = xa.domain()[1]; dd >= 0.25; dd -= 0.05) hyp.push([xa(dd), ya(VZ.clamp(fB / dd, 0, 45))]);
    VZ.poly(ga, hyp, { stroke: VC.accent, w: 2, close: false });
    for (let k = 1; k <= 80; k++) {
      const Z = fB / k;
      if (Z > 45 || Z < 0.1) continue;
      if (xa(k) > 200) continue;
      ga.append("line").attr("x1", 0).attr("x2", 14).attr("y1", ya(Z)).attr("y2", ya(Z))
        .attr("stroke", VC.a2).attr("stroke-width", 1).attr("stroke-opacity", 0.8);
    }
    ga.append("line").attr("x1", 0).attr("x2", 200).attr("y1", ya(Zmax)).attr("y2", ya(Zmax))
      .attr("stroke", VC.good).attr("stroke-width", 1.3).attr("stroke-dasharray", "5 3");
    ga.append("text").attr("x", 22).attr("y", VZ.clamp(ya(Zmax) - 5, 10, 320)).attr("font-size", 9.5)
      .attr("fill", VC.good).text(VZ.fmt(100 * tol, 0) + "% range = " + VZ.fmt(Zmax, 1) + " m");

    /* ── B: measured σ_Z against Z, log–log ─────────────────────────────── */
    const gb = VZ.panelBox(g, 300, 40, 200, 330, "measured σ_Z vs Z (log–log)");
    const xb = d3.scaleLog().domain([0.8, 50]).range([0, 200]);
    const yb = d3.scaleLog().domain([1e-4, 1e3]).range([330, 0]).clamp(true);
    gb.append("g").attr("class", "axis").attr("transform", "translate(0,330)").call(d3.axisBottom(xb).ticks(4, "~s"));
    gb.append("g").attr("class", "axis").call(d3.axisLeft(yb).ticks(7, "~e"));
    gb.append("text").attr("x", 200).attr("y", 324).attr("text-anchor", "end").attr("font-size", 9.5)
      .attr("fill", VC.muted).text("Z (m)");
    const pline = [];
    for (let Z = 0.8; Z <= 50; Z *= 1.06) pline.push([xb(Z), yb(VZ.clamp(Z * Z * sd / fB, 1e-4, 1e3))]);
    VZ.poly(gb, pline, { stroke: VC.accent, w: 1.8, close: false, dash: "5 3" });
    meas.forEach(m => {
      if (!(m.sd > 0)) return;
      gb.append("circle").attr("cx", xb(m.Z)).attr("cy", yb(VZ.clamp(m.sd, 1e-4, 1e3))).attr("r", 3)
        .attr("fill", m.d > 8 * sd ? VC.good : VC.bad);
    });
    VZ.legend(gb, [{ color: VC.accent, label: "Z²σ_d/(fB), predicted", dash: "5 3" },
    { color: VC.good, label: "measured, d > 8σ_d" }, { color: VC.bad, label: "measured, d ≤ 8σ_d" }], 8, 16, { gap: 13, font: 9.5 });
    gb.append("text").attr("x", 2).attr("y", 350).attr("font-size", 10).attr("fill", VC.good)
      .text("fitted slope (d > 8σ_d): " + VZ.fmt(slope, 4) + "   predicted: 2");
    gb.append("text").attr("x", 2).attr("y", 362).attr("font-size", 10).attr("fill", VC.muted)
      .text("fitted over ALL depths: " + VZ.fmt(slopeAll, 4));

    /* ── C: measured ÷ predicted ────────────────────────────────────────── */
    const gc = VZ.panelBox(g, 554, 40, 190, 330, "measured ÷ predicted");
    const xc = d3.scaleLog().domain([0.8, 50]).range([0, 190]);
    const yc = d3.scaleLinear().domain([0.9, 1.6]).range([330, 0]).clamp(true);
    VZ.gridY(gc, yc, 190, 5);
    gc.append("g").attr("class", "axis").attr("transform", "translate(0,330)").call(d3.axisBottom(xc).ticks(4, "~s"));
    gc.append("g").attr("class", "axis").call(d3.axisLeft(yc).ticks(6));
    gc.append("line").attr("x1", 0).attr("x2", 190).attr("y1", yc(1)).attr("y2", yc(1))
      .attr("stroke", VC.accent).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
    const rc = meas.filter(m => m.sd > 0).map(m => [xc(m.Z), yc(VZ.clamp(m.sd / m.pred, 0.9, 1.6))]);
    VZ.poly(gc, rc, { stroke: VC.a2, w: 1.9, close: false });
    meas.forEach(m => { if (m.sd > 0) gc.append("circle").attr("cx", xc(m.Z)).attr("cy", yc(VZ.clamp(m.sd / m.pred, 0.9, 1.6))).attr("r", 2.6).attr("fill", VC.a2); });
    gc.append("line").attr("x1", xc(VZ.clamp(Zmax, 0.8, 50))).attr("x2", xc(VZ.clamp(Zmax, 0.8, 50)))
      .attr("y1", 0).attr("y2", 330).attr("stroke", VC.good).attr("stroke-width", 1.3).attr("stroke-dasharray", "4 3");
    gc.append("text").attr("x", VZ.clamp(xc(VZ.clamp(Zmax, 0.8, 50)) + 4, 0, 110)).attr("y", 14)
      .attr("font-size", 9).attr("fill", VC.good).text(VZ.fmt(100 * tol, 0) + "% range");
    gc.append("text").attr("x", 2).attr("y", 350).attr("font-size", 9.5).attr("fill", VC.muted)
      .text("1.00 while the linearisation holds");

    const at5 = meas.find(m => m.Z === 5), at20 = meas.find(m => m.Z === 20);
    ro.html(
      `f = <b>${f} px</b>, B = <b>${VZ.fmt(B, 2)} m</b>, fB = <b>${VZ.fmt(fB, 1)} px·m</b>, ` +
      `<span class="keep">σ</span>_d = <b>${VZ.fmt(sd, 2)} px</b>, ${NS.toLocaleString()} samples per depth<br>` +
      `at Z = 5 m: d = <b>${VZ.fmt(at5.d, 2)} px</b>, measured <span class="keep">σ</span>_Z = <b>${VZ.fmt(at5.sd, 4)} m</b> vs predicted ` +
      `<b>${VZ.fmt(at5.pred, 4)} m</b> (ratio ${VZ.fmt(at5.sd / at5.pred, 4)}, relative error ${VZ.fmt(100 * at5.sd / 5, 2)}%)<br>` +
      `at Z = 20 m: d = <b>${VZ.fmt(at20.d, 2)} px</b>, measured <b>${VZ.fmt(at20.sd, 3)} m</b> vs predicted <b>${VZ.fmt(at20.pred, 3)} m</b> ` +
      `(ratio ${VZ.fmt(at20.sd / at20.pred, 3)}, relative error ${VZ.fmt(100 * at20.sd / 20, 1)}%)<br>` +
      `<b>measured log–log slope ${VZ.fmt(slope, 4)}</b> against the predicted exponent of exactly <b>2</b>. ` +
      `Range at ${VZ.fmt(100 * tol, 0)}% relative depth error: <b>${VZ.fmt(Zmax, 1)} m</b> — and doubling it costs a doubling of f or of B, both of which have a price.`
    );
  }

  [["dp-f", v => VZ.fmt(v, 0) + " px"], ["dp-b", v => VZ.fmt(v, 2) + " m"],
  ["dp-s", v => VZ.fmt(v, 2) + " px"], ["dp-t", v => VZ.fmt(v, 0) + " %"]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  draw();
})();

/* ══════════ the synthetic rectified stereo pair, shared by §19–§24 ═══════
   Built by RENDERING three surfaces into two views rather than by warping one
   image into the other, so that occlusion and half-occlusion are real: a
   background pixel hidden behind the foreground box in the right view genuinely
   has no match, and the figures do not have to pretend. All coordinates are
   already rectified, so a match is a horizontal shift and nothing else.        */
const ST = (function () {
  const W = 96, H = 72, DMAX = 18;

  /* smooth value noise: a random lattice, bilinearly interpolated, two octaves */
  function noise(seed) {
    const r = VZ.rng(seed);
    const g1 = [], g2 = [], s1 = 3, s2 = 7;
    const n1w = Math.ceil((W + 40) / s1) + 2, n1h = Math.ceil(H / s1) + 2;
    const n2w = Math.ceil((W + 40) / s2) + 2, n2h = Math.ceil(H / s2) + 2;
    for (let j = 0; j < n1h; j++) { const row = []; for (let i = 0; i < n1w; i++) row.push(r() * 2 - 1); g1.push(row); }
    for (let j = 0; j < n2h; j++) { const row = []; for (let i = 0; i < n2w; i++) row.push(r() * 2 - 1); g2.push(row); }
    const samp = (G, s, x, y) => {
      const u = (x + 20) / s, v = y / s;
      const i = Math.max(0, Math.min(G[0].length - 2, Math.floor(u))), j = Math.max(0, Math.min(G.length - 2, Math.floor(v)));
      const a = u - i, b = v - j;
      return G[j][i] * (1 - a) * (1 - b) + G[j][i + 1] * a * (1 - b) + G[j + 1][i] * (1 - a) * b + G[j + 1][i + 1] * a * b;
    };
    return (x, y) => 0.68 * samp(g1, s1, x, y) + 0.32 * samp(g2, s2, x, y);
  }

  /* three surfaces. Each knows its disparity as a function of the LEFT-image x,
     and knows how to invert that, so the right view can be rendered exactly. */
  function build() {
    const nb = noise(3), nf = noise(11), nsl = noise(23);
    const surfaces = [
      { /* the foreground box */
        name: "box",
        inBox: (x, y) => x >= 22 && x < 60 && y >= 14 && y < 56,
        d: () => 14,
        inv: xr => xr + 14,
        tex: (x, y) => (x >= 28 && x < 50 && y >= 22 && y < 44)
          ? 150                                        /* the TEXTURELESS patch */
          : VZ.clamp(132 + 42 * nf(x, y), 0, 255)
      },
      { /* a slanted plane on the right */
        name: "slant",
        inBox: (x, y) => x >= 66 && x < 94 && y >= 8 && y < 64,
        d: x => 5 + 0.26 * (x - 66),
        inv: xr => (xr + 5 - 0.26 * 66) / (1 - 0.26),
        tex: (x, y) => (y >= 14 && y < 44)
          ? VZ.clamp(128 + 52 * Math.sin(2 * Math.PI * x / 9), 0, 255)   /* REPEATED */
          : VZ.clamp(126 + 44 * nsl(x, y), 0, 255)
      },
      { /* the background plane, which covers everything */
        name: "bg",
        inBox: () => true,
        d: () => 4,
        inv: xr => xr + 4,
        tex: (x, y) => VZ.clamp(124 + 40 * nb(x, y), 0, 255)
      }
    ];
    const L = VZ.zeros2(H, W), R = VZ.zeros2(H, W), D = VZ.zeros2(H, W), OCC = VZ.zeros2(H, W);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      for (const s of surfaces) {
        if (!s.inBox(x, y)) continue;
        L[y][x] = s.tex(x, y); D[y][x] = s.d(x);
        break;
      }
    }
    for (let y = 0; y < H; y++) for (let xr = 0; xr < W; xr++) {
      for (const s of surfaces) {
        const xl = s.inv(xr);
        if (!s.inBox(xl, y)) continue;
        R[y][xr] = s.tex(xl, y);
        break;
      }
    }
    /* Independent sensor noise in each view, 1.5 grey levels. This matters more
       than it looks: without it the untextured patch produces an EXACTLY tied cost
       curve and the "failure" is a tie-break artefact rather than a real one. With
       it, the winner in a flat region is genuinely decided by noise, which is what
       happens in a real camera and what §21 and §22 are trying to survive. */
    const rL = VZ.rng(101), rR = VZ.rng(202), SIG = 1.5;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      L[y][x] = VZ.clamp(L[y][x] + SIG * VZ.randn(rL), 0, 255);
      R[y][x] = VZ.clamp(R[y][x] + SIG * VZ.randn(rR), 0, 255);
    }
    /* half-occlusion: a LEFT pixel is occluded when the surface in front of it in
       the right view is a different one, i.e. its match is hidden */
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const xr = x - D[y][x];
      let hit = null;
      for (const s of surfaces) { const xl = s.inv(xr); if (s.inBox(xl, y)) { hit = s; break; } }
      let own = null;
      for (const s of surfaces) { if (s.inBox(x, y)) { own = s; break; } }
      OCC[y][x] = (hit !== own || xr < 0) ? 1 : 0;
    }
    return { L: L, R: R, D: D, OCC: OCC, W: W, H: H, DMAX: DMAX };
  }

  const SC = build();

  /* ---- matching costs ---------------------------------------------------
     All defined on a (2r+1)² window centred on (x, y) in the left image and
     (x − d, y) in the right, and all returned SMALLER-IS-BETTER.             */
  function census(A, r) {
    const out = [];
    for (let y = 0; y < H; y++) {
      const row = [];
      for (let x = 0; x < W; x++) {
        let bits = 0, k = 0;
        const c = A[y][x];
        for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) {
          if (i === 0 && j === 0) continue;
          const yy = VZ.clamp(y + j, 0, H - 1), xx = VZ.clamp(x + i, 0, W - 1);
          if (A[yy][xx] < c) bits |= (1 << k);
          k++;
        }
        row.push(bits);
      }
      out.push(row);
    }
    return out;
  }
  const popcount = v => { v = v - ((v >> 1) & 0x55555555); v = (v & 0x33333333) + ((v >> 2) & 0x33333333); return ((((v + (v >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24) & 0x3F; };

  /* the cost volume, C[d][y][x] */
  function volume(L, R, kind, r, dmax) {
    const DM = dmax === undefined ? DMAX : dmax;
    const C = [];
    let cenL = null, cenR = null;
    if (kind === "census") { cenL = census(L, Math.min(r, 2)); cenR = census(R, Math.min(r, 2)); }
    for (let d = 0; d <= DM; d++) {
      const P = VZ.zeros2(H, W);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const xr = x - d;
        /* no partner: charge the worst cost this measure can produce, so the
           volume stays on a sensible scale for plotting */
        if (xr < 0) { P[y][x] = kind === "ncc" ? 2 : (kind === "ssd" ? 20000 : (kind === "sad" ? 200 : 1)); continue; }
        if (kind === "census") { P[y][x] = popcount(cenL[y][x] ^ cenR[y][xr]) / 24; continue; }
        let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, sad = 0, ssd = 0, n = 0;
        for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) {
          const yy = VZ.clamp(y + j, 0, H - 1);
          const xa = VZ.clamp(x + i, 0, W - 1), xb = VZ.clamp(xr + i, 0, W - 1);
          const a = L[yy][xa], b = R[yy][xb];
          sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
          sad += Math.abs(a - b); ssd += (a - b) * (a - b); n++;
        }
        if (kind === "sad") P[y][x] = sad / n;
        else if (kind === "ssd") P[y][x] = ssd / n;
        else {                                     /* NCC, as 1 − correlation */
          const ca = saa - sa * sa / n, cb = sbb - sb * sb / n, cab = sab - sa * sb / n;
          const den = Math.sqrt(Math.max(ca, 1e-9) * Math.max(cb, 1e-9));
          P[y][x] = 1 - VZ.clamp(cab / den, -1, 1);
        }
      }
      C.push(P);
    }
    return C;
  }

  /* winner-take-all over the volume */
  function wta(C) {
    const D = VZ.zeros2(H, W);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let bd = 0, bv = Infinity;
      for (let d = 0; d < C.length; d++) if (C[d][y][x] < bv) { bv = C[d][y][x]; bd = d; }
      D[y][x] = bd;
    }
    return D;
  }
  /* fraction of non-occluded pixels wrong by more than tol disparities */
  function badRate(D, tol, mask) {
    let bad = 0, n = 0;
    for (let y = 2; y < H - 2; y++) for (let x = 2; x < W - 2; x++) {
      if (SC.OCC[y][x]) continue;
      if (mask && !mask(x, y)) continue;
      n++;
      if (Math.abs(D[y][x] - SC.D[y][x]) > tol) bad++;
    }
    return n ? bad / n : 0;
  }
  /* the named regions, so the failure figures can score them separately */
  const REGIONS = {
    textureless: (x, y) => x >= 28 && x < 50 && y >= 22 && y < 44,
    repeated: (x, y) => x >= 66 && x < 94 && y >= 14 && y < 44,
    slanted: (x, y) => x >= 66 && x < 94 && y >= 44 && y < 64,
    background: (x, y) => !(x >= 22 && x < 60 && y >= 14 && y < 56) && !(x >= 66 && x < 94 && y >= 8 && y < 64)
  };

  return {
    W: W, H: H, DMAX: DMAX, scene: SC, volume: volume, wta: wta, badRate: badRate,
    REGIONS: REGIONS, census: census
  };
})();

/* ══════════ 13 · #cost-svg — four matching costs ════════════════════════ */
(function () {
  const svg = d3.select("#cost-svg");
  if (svg.empty()) return;
  const W = 760, H = 438;
  const ro = d3.select("#cost-readout");
  const el = id => document.getElementById(id);
  const S = ST.scene, PX = 2;                           // 96×72 drawn at 2× → 192×144
  const KINDS = [["sad", "SAD", VC.a2], ["ssd", "SSD", VC.bad], ["ncc", "NCC", VC.accent], ["census", "census", VC.good]];
  /* the baseline error rates, computed once */
  let BASE = null;

  function draw() {
    const gA = +el("ct-g").value, bB = +el("ct-b").value, r = +el("ct-r").value;
    const px = +el("ct-x").value, py = +el("ct-y").value;
    const R2 = S.R.map(row => Array.from(row, v => VZ.clamp(gA * v + bB, 0, 255)));
    if (!BASE) BASE = KINDS.map(([k]) => ST.badRate(ST.wta(ST.volume(S.L, S.R, k, 2)), 1));

    const vols = KINDS.map(([k]) => ST.volume(S.L, R2, k, r));
    const rates = vols.map(C => ST.badRate(ST.wta(C), 1));

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── rasters ────────────────────────────────────────────────────────── */
    const mkRaster = (x, title, A, opt) => {
      const gg = VZ.panelBox(g, x, 40, ST.W * PX, ST.H * PX, title);
      VZ.raster(gg, A, 0, 0, ST.W * PX, ST.H * PX, opt || { lo: 0, hi: 255 });
      gg.append("circle").attr("cx", (px + 0.5) * PX).attr("cy", (py + 0.5) * PX).attr("r", 5)
        .attr("fill", "none").attr("stroke", VC.violet).attr("stroke-width", 1.8);
      return gg;
    };
    mkRaster(30, "left image", S.L);
    mkRaster(248, "right image · a = " + VZ.fmt(gA, 2) + ", b = " + VZ.fmt(bB, 0), R2);
    const gd = mkRaster(466, "ground-truth disparity", S.D, { lo: 0, hi: ST.DMAX });
    gd.append("text").attr("x", 0).attr("y", ST.H * PX + 14).attr("font-size", 9.5).attr("fill", VC.muted)
      .text("box d = 14 · background d = 4 · slant d = 5 → 12");

    /* ── the cost curves at the probe ───────────────────────────────────── */
    const gc = VZ.panelBox(g, 60, 246, 300, 150, "cost vs disparity at the probe, each rescaled to [0, 1]");
    const xs = d3.scaleLinear().domain([0, ST.DMAX]).range([0, 300]);
    const ys = d3.scaleLinear().domain([-0.04, 1.04]).range([150, 0]);
    VZ.gridY(gc, ys, 300, 4);
    gc.append("g").attr("class", "axis").attr("transform", "translate(0,150)").call(d3.axisBottom(xs).ticks(7));
    gc.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(3));
    gc.append("text").attr("x", 300).attr("y", 144).attr("text-anchor", "end").attr("font-size", 9.5)
      .attr("fill", VC.muted).text("disparity d");
    const dTrue = S.D[py][px];
    gc.append("line").attr("x1", xs(dTrue)).attr("x2", xs(dTrue)).attr("y1", 0).attr("y2", 150)
      .attr("stroke", VC.violet).attr("stroke-width", 1.5).attr("stroke-dasharray", "4 3");
    gc.append("text").attr("x", xs(dTrue) + 4).attr("y", 11).attr("font-size", 9).attr("fill", VC.violet)
      .text("true d = " + VZ.fmt(dTrue, 1));
    const argmins = [];
    const DVAL = Math.min(ST.DMAX, px);              /* d > x has no partner at all */
    vols.forEach((C, i) => {
      const vals = C.slice(0, DVAL + 1).map(P => P[py][px]);
      const lo = d3.min(vals), hi = d3.max(vals);
      const nv = vals.map(v => (hi - lo) > 1e-12 ? (v - lo) / (hi - lo) : 0.5);
      VZ.poly(gc, nv.map((v, d) => [xs(d), ys(v)]), { stroke: KINDS[i][2], w: 1.7, close: false });
      argmins.push(vals.indexOf(lo));
      gc.append("circle").attr("cx", xs(vals.indexOf(lo))).attr("cy", ys(0)).attr("r", 3).attr("fill", KINDS[i][2]);
    });
    VZ.legend(gc, KINDS.map(([k, lb, c]) => ({ color: c, label: lb })), 244, 14, { gap: 13, font: 9.5 });

    /* ── the error bars ─────────────────────────────────────────────────── */
    const gb = VZ.panelBox(g, 440, 246, 290, 150, "bad pixels (> 1 disparity, non-occluded)");
    const xb = d3.scaleBand().domain(d3.range(4)).range([0, 290]).padding(0.3);
    const yb = d3.scaleLinear().domain([0, 100]).range([150, 0]);
    VZ.gridY(gb, yb, 290, 5);
    gb.append("g").attr("class", "axis").call(d3.axisLeft(yb).ticks(5).tickFormat(d => d + "%"));
    rates.forEach((v, i) => {
      gb.append("rect").attr("x", xb(i)).attr("y", yb(100 * v)).attr("width", xb.bandwidth())
        .attr("height", 150 - yb(100 * v)).attr("fill", KINDS[i][2]).attr("fill-opacity", 0.85);
      gb.append("rect").attr("x", xb(i) - 3).attr("y", yb(100 * BASE[i])).attr("width", xb.bandwidth() + 6)
        .attr("height", 150 - yb(100 * BASE[i])).attr("fill", "none").attr("stroke", VC.muted)
        .attr("stroke-width", 1).attr("stroke-dasharray", "3 2");
      gb.append("text").attr("x", xb(i) + xb.bandwidth() / 2).attr("y", 164).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", VC.muted).text(KINDS[i][1]);
      gb.append("text").attr("x", xb(i) + xb.bandwidth() / 2).attr("y", yb(100 * v) - 4).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", VC.ink).text(VZ.fmt(100 * v, 1) + "%");
    });
    gb.append("text").attr("x", 0).attr("y", 178).attr("font-size", 9).attr("fill", VC.muted)
      .text("dashed outline: the same cost with a = 1, b = 0 and r = 2");

    const region = ST.REGIONS.textureless(px, py) ? "the untextured patch"
      : ST.REGIONS.repeated(px, py) ? "the repeating stripes"
        : S.OCC[py][px] ? "a half-occluded pixel — it has NO correct answer"
          : "a textured region";
    ro.html(
      `probe (${px}, ${py}) — ${region}. true disparity <b>${VZ.fmt(dTrue, 2)}</b>; winners: ` +
      KINDS.map(([k, lb], i) => `${lb} <b>${argmins[i]}</b>`).join(" · ") + `<br>` +
      `right image transformed by a = <b>${VZ.fmt(gA, 2)}</b>, b = <b>${VZ.fmt(bB, 0)}</b>, window ${2 * r + 1} × ${2 * r + 1}<br>` +
      `bad-pixel rate: ` + KINDS.map(([k, lb], i) => `${lb} <b>${VZ.fmt(100 * rates[i], 1)}%</b>`).join(" · ") +
      (Math.abs(gA - 1) < 1e-9 && Math.abs(bB) < 1e-9
        ? ` — this is the clean baseline; SAD is the best of the four here, census the worst.`
        : ` — against ${KINDS.map(([k, lb], i) => `${VZ.fmt(100 * BASE[i], 1)}%`).join(" / ")} undisturbed. ` +
        `NCC and census have not moved; SAD and SSD have been destroyed by a change you can barely see.`)
    );
  }

  [["ct-g", v => VZ.fmt(v, 2)], ["ct-b", v => VZ.fmt(v, 0)], ["ct-r", v => "" + v],
  ["ct-x", v => "" + v], ["ct-y", v => "" + v]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  draw();
})();

/* ══════════ 14 · #dsi-svg — the cost volume and WTA ═════════════════════ */
(function () {
  const svg = d3.select("#dsi-svg");
  if (svg.empty()) return;
  const W = 760, H = 450;
  const ro = d3.select("#dsi-readout");
  const el = id => document.getElementById(id);
  const S = ST.scene;

  /* the right-image cost volume, for cross-checking: cost at right pixel xr for
     disparity d is the same patch pair as left pixel xr + d */
  function rightWTA(C) {
    const D = VZ.zeros2(ST.H, ST.W);
    for (let y = 0; y < ST.H; y++) for (let xr = 0; xr < ST.W; xr++) {
      let bd = 0, bv = Infinity;
      for (let d = 0; d < C.length; d++) {
        const xl = xr + d;
        if (xl >= ST.W) break;
        if (C[d][y][xl] < bv) { bv = C[d][y][xl]; bd = d; }
      }
      D[y][xr] = bd;
    }
    return D;
  }

  function draw() {
    const y0 = +el("ds-y").value, kind = el("ds-k").value, r = +el("ds-r").value;
    const cross = el("ds-x").checked;
    const C = ST.volume(S.L, S.R, kind, r);
    let D = ST.wta(C);
    const DR = rightWTA(C);
    const dropped = VZ.zeros2(ST.H, ST.W);
    let nDrop = 0;
    if (cross) {
      for (let y = 0; y < ST.H; y++) for (let x = 0; x < ST.W; x++) {
        const xr = x - D[y][x];
        if (xr < 0 || Math.abs(DR[y][xr] - D[y][x]) > 1) { dropped[y][x] = 1; nDrop++; }
      }
    }
    const rate = (function () {
      let bad = 0, n = 0;
      for (let y = 2; y < ST.H - 2; y++) for (let x = 2; x < ST.W - 2; x++) {
        if (S.OCC[y][x]) continue;
        if (cross && dropped[y][x]) continue;
        n++; if (Math.abs(D[y][x] - S.D[y][x]) > 1) bad++;
      }
      return { rate: n ? bad / n : 0, n: n };
    })();

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: the (x, d) slice ────────────────────────────────────────────── */
    const SW = 3, SH = 8;
    const ga = VZ.panelBox(g, 40, 40, ST.W * SW, (ST.DMAX + 1) * SH, "cost slice at y = " + y0 + "  ·  x across, d down");
    const slice = [];
    for (let d = 0; d <= ST.DMAX; d++) slice.push(Array.from({ length: ST.W }, (_, x) => C[d][y0][x]));
    VZ.raster(ga, slice, 0, 0, ST.W * SW, (ST.DMAX + 1) * SH, { gamma: 0.55 });
    const gt = [], wt = [];
    for (let x = 0; x < ST.W; x++) {
      gt.push([(x + 0.5) * SW, (S.D[y0][x] + 0.5) * SH]);
      wt.push([(x + 0.5) * SW, (D[y0][x] + 0.5) * SH]);
    }
    VZ.poly(ga, wt, { stroke: VC.a2, w: 1.6, close: false });
    VZ.poly(ga, gt, { stroke: VC.good, w: 1.4, close: false, dash: "4 3" });
    VZ.legend(ga, [{ color: VC.good, label: "ground truth", dash: "4 3" }, { color: VC.a2, label: "winner-take-all" }],
      6, (ST.DMAX + 1) * SH + 16, { gap: 13, font: 9.5, vertical: false, step: 108 });

    /* ── B: WTA map and error map ───────────────────────────────────────── */
    const PX = 2;
    const gb = VZ.panelBox(g, 372, 40, ST.W * PX, ST.H * PX, "WTA disparity");
    VZ.raster(gb, D, 0, 0, ST.W * PX, ST.H * PX, { lo: 0, hi: ST.DMAX });
    gb.append("line").attr("x1", 0).attr("x2", ST.W * PX).attr("y1", (y0 + 0.5) * PX).attr("y2", (y0 + 0.5) * PX)
      .attr("stroke", VC.a2).attr("stroke-width", 1).attr("stroke-opacity", 0.8);
    const gc = VZ.panelBox(g, 372 + ST.W * PX + 18, 40, ST.W * PX, ST.H * PX, "where it is wrong");
    const err = [];
    for (let y = 0; y < ST.H; y++) {
      const row = [];
      for (let x = 0; x < ST.W; x++) {
        row.push(S.OCC[y][x] ? 0.45 : (cross && dropped[y][x] ? 0.7 : (Math.abs(D[y][x] - S.D[y][x]) > 1 ? 1 : 0)));
      }
      err.push(row);
    }
    VZ.cells(gc, 0, 0, PX, ST.W, ST.H, (x, y) => {
      const v = err[y][x];
      return v === 0 ? VC.panel2 : v === 1 ? VC.bad : v === 0.7 ? VC.accent : VC.line;
    });
    VZ.legend(gc, [{ color: VC.bad, label: "wrong by > 1" }, { color: VC.line, label: "half-occluded" }]
      .concat(cross ? [{ color: VC.accent, label: "dropped by cross-check" }] : []),
      0, ST.H * PX + 14, { gap: 12, font: 9.5 });

    /* ── C: three cost curves ───────────────────────────────────────────── */
    const probes = [[16, 62, "textured background"], [38, 32, "untextured patch"], [80, 26, "periodic stripes"]];
    probes.forEach(([px, py, lb], i) => {
      const gg = VZ.panelBox(g, 52 + i * 236, 300, 196, 110, lb + "  (" + px + ", " + py + ")");
      const vals = C.slice(0, Math.min(ST.DMAX, px) + 1).map(P => P[py][px]);
      const lo = d3.min(vals), hi = d3.max(vals);
      const xs = d3.scaleLinear().domain([0, ST.DMAX]).range([0, 196]);
      const ys = d3.scaleLinear().domain([-0.05, 1.05]).range([110, 0]);
      gg.append("g").attr("class", "axis").attr("transform", "translate(0,110)").call(d3.axisBottom(xs).ticks(5));
      const nv = vals.map(v => (hi - lo) > 1e-12 ? (v - lo) / (hi - lo) : 0.5);
      VZ.poly(gg, nv.map((v, d) => [xs(d), ys(v)]), { stroke: VC.accent, w: 1.7, close: false });
      gg.append("line").attr("x1", xs(S.D[py][px])).attr("x2", xs(S.D[py][px])).attr("y1", 0).attr("y2", 110)
        .attr("stroke", VC.good).attr("stroke-width", 1.3).attr("stroke-dasharray", "4 3");
      const am = vals.indexOf(lo);
      gg.append("line").attr("x1", xs(am)).attr("x2", xs(am)).attr("y1", 0).attr("y2", 110)
        .attr("stroke", VC.bad).attr("stroke-width", 1.2);
      const contrast = (hi - lo) > 1e-12 ? (d3.min(vals.filter((v, d) => Math.abs(d - am) > 2)) - lo) / (hi - lo) : 0;
      gg.append("text").attr("x", 2).attr("y", 124).attr("font-size", 9).attr("fill", VC.muted)
        .text("true " + S.D[py][px].toFixed(0) + " · won " + am + " · margin " + VZ.fmt(contrast, 3));
    });

    ro.html(
      `cost <b>${kind.toUpperCase()}</b>, window <b>${2 * r + 1} × ${2 * r + 1}</b>, ${ST.W} × ${ST.H} px, ` +
      `${ST.DMAX + 1} disparity levels → a volume of <b>${(ST.W * ST.H * (ST.DMAX + 1)).toLocaleString()}</b> entries<br>` +
      `bad pixels (> 1 disparity, non-occluded${cross ? ", after cross-check" : ""}): <b>${VZ.fmt(100 * rate.rate, 1)}%</b> of ${rate.n} scored pixels` +
      (cross ? `; the cross-check dropped <b>${nDrop}</b> pixels (${VZ.fmt(100 * nDrop / (ST.W * ST.H), 1)}% of the image), ` +
        `most of them in the occlusion band left of the box — which is where they should be` : `. Turn on the cross-check to see the occlusions found automatically.`) + `<br>` +
      `per region: untextured <b>${VZ.fmt(100 * ST.badRate(D, 1, ST.REGIONS.textureless), 1)}%</b> · ` +
      `stripes <b>${VZ.fmt(100 * ST.badRate(D, 1, ST.REGIONS.repeated), 1)}%</b> · ` +
      `slanted <b>${VZ.fmt(100 * ST.badRate(D, 1, ST.REGIONS.slanted), 1)}%</b> · ` +
      `textured background <b>${VZ.fmt(100 * ST.badRate(D, 1, ST.REGIONS.background), 1)}%</b>`
    );
  }

  [["ds-y", v => "" + v], ["ds-r", v => "" + v]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  el("ds-k").addEventListener("change", draw);
  el("ds-x").addEventListener("change", draw);
  draw();
})();

/* ══════════ 15 · #agg-svg — aggregation window size ═════════════════════ */
(function () {
  const svg = d3.select("#agg-svg");
  if (svg.empty()) return;
  const W = 760, H = 462;
  const ro = d3.select("#agg-readout");
  const el = id => document.getElementById(id);
  const S = ST.scene;

  /* the masks the two curves are scored on */
  const DISC = VZ.zeros2(ST.H, ST.W);
  for (let y = 1; y < ST.H - 1; y++) for (let x = 1; x < ST.W - 1; x++) {
    const g = Math.max(Math.abs(S.D[y][x] - S.D[y][x - 1]), Math.abs(S.D[y][x] - S.D[y][x + 1]),
      Math.abs(S.D[y][x] - S.D[y - 1][x]), Math.abs(S.D[y][x] - S.D[y + 1][x]));
    if (g > 2) DISC[y][x] = 1;
  }
  const NEAR = (function () {
    const O = VZ.zeros2(ST.H, ST.W);
    for (let y = 0; y < ST.H; y++) for (let x = 0; x < ST.W; x++) {
      let n = 0;
      for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++)
        if (DISC[VZ.clamp(y + j, 0, ST.H - 1)][VZ.clamp(x + i, 0, ST.W - 1)]) n = 1;
      O[y][x] = n;
    }
    return O;
  })();
  const flat = (x, y) => ST.REGIONS.textureless(x, y) || ST.REGIONS.repeated(x, y);

  /* separable min-filter — the shiftable-window identity of §21 */
  function minFilter(P, r) {
    const A = VZ.zeros2(ST.H, ST.W), B = VZ.zeros2(ST.H, ST.W);
    for (let y = 0; y < ST.H; y++) for (let x = 0; x < ST.W; x++) {
      let m = Infinity;
      for (let i = -r; i <= r; i++) { const v = P[y][VZ.clamp(x + i, 0, ST.W - 1)]; if (v < m) m = v; }
      A[y][x] = m;
    }
    for (let y = 0; y < ST.H; y++) for (let x = 0; x < ST.W; x++) {
      let m = Infinity;
      for (let j = -r; j <= r; j++) { const v = A[VZ.clamp(y + j, 0, ST.H - 1)][x]; if (v < m) m = v; }
      B[y][x] = m;
    }
    return B;
  }

  const CACHE = {};
  function maps(kind, r) {
    const key = kind + ":" + r;
    if (CACHE[key]) return CACHE[key];
    const C = ST.volume(S.L, S.R, kind, r);
    const box = ST.wta(C);
    const shift = ST.wta(C.map(P => minFilter(P, r)));
    const out = {
      box: box, shift: shift,
      bFar: ST.badRate(box, 1, (x, y) => NEAR[y][x] === 0 && !flat(x, y)),
      bNear: ST.badRate(box, 1, (x, y) => NEAR[y][x] === 1 && !flat(x, y)),
      bAll: ST.badRate(box, 1),
      sNear: ST.badRate(shift, 1, (x, y) => NEAR[y][x] === 1 && !flat(x, y)),
      sAll: ST.badRate(shift, 1)
    };
    CACHE[key] = out;
    return out;
  }

  function draw() {
    const r = +el("ag-r").value, kind = el("ag-k").value, y0 = +el("ag-y").value;
    const cur = maps(kind, r);
    const RS = d3.range(1, 7);
    const all = RS.map(rr => maps(kind, rr));

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: the curves ──────────────────────────────────────────────────── */
    const ga = VZ.panelBox(g, 46, 40, 250, 250, "bad-pixel rate vs window radius");
    const xs = d3.scaleLinear().domain([1, 6]).range([0, 250]);
    const ys = d3.scaleLinear().domain([0, Math.max(0.28, d3.max(all, a => Math.max(a.bNear, a.sNear, a.bAll)) * 1.15)]).range([250, 0]);
    VZ.gridY(ga, ys, 250, 5);
    ga.append("g").attr("class", "axis").attr("transform", "translate(0,250)").call(d3.axisBottom(xs).ticks(6));
    ga.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(6).tickFormat(d3.format(".0%")));
    ga.append("text").attr("x", 250).attr("y", 244).attr("text-anchor", "end").attr("font-size", 9.5)
      .attr("fill", VC.muted).text("radius r");
    const series = [
      { get: a => a.bFar, c: VC.good, lb: "box · textured, far from edges" },
      { get: a => a.bNear, c: VC.bad, lb: "box · within 2 px of an edge" },
      { get: a => a.bAll, c: VC.accent, lb: "box · everything" },
      { get: a => a.sAll, c: VC.violet, lb: "shiftable · everything", dash: "5 3" }
    ];
    series.forEach(s => {
      VZ.poly(ga, RS.map((rr, i) => [xs(rr), ys(s.get(all[i]))]), { stroke: s.c, w: 1.8, close: false, dash: s.dash || null });
      RS.forEach((rr, i) => ga.append("circle").attr("cx", xs(rr)).attr("cy", ys(s.get(all[i]))).attr("r", 2.4).attr("fill", s.c));
    });
    ga.append("line").attr("x1", xs(r)).attr("x2", xs(r)).attr("y1", 0).attr("y2", 250)
      .attr("stroke", VC.a2).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
    VZ.legend(ga, series.map(s => ({ color: s.c, label: s.lb, dash: s.dash })), 6, 268, { gap: 12, font: 9.5 });

    /* ── B, C: the two disparity maps ───────────────────────────────────── */
    const PX = 2;
    const gb = VZ.panelBox(g, 336, 40, ST.W * PX, ST.H * PX, "box aggregation, r = " + r);
    VZ.raster(gb, cur.box, 0, 0, ST.W * PX, ST.H * PX, { lo: 0, hi: ST.DMAX });
    gb.append("line").attr("x1", 0).attr("x2", ST.W * PX).attr("y1", (y0 + 0.5) * PX).attr("y2", (y0 + 0.5) * PX)
      .attr("stroke", VC.a2).attr("stroke-width", 1).attr("stroke-opacity", 0.85);
    const gc = VZ.panelBox(g, 548, 40, ST.W * PX, ST.H * PX, "shiftable windows, r = " + r);
    VZ.raster(gc, cur.shift, 0, 0, ST.W * PX, ST.H * PX, { lo: 0, hi: ST.DMAX });
    gc.append("line").attr("x1", 0).attr("x2", ST.W * PX).attr("y1", (y0 + 0.5) * PX).attr("y2", (y0 + 0.5) * PX)
      .attr("stroke", VC.a2).attr("stroke-width", 1).attr("stroke-opacity", 0.85);
    /* the ground truth, small, for reference */
    const gt = VZ.panelBox(g, 336, 214, ST.W * PX, ST.H * PX, "ground truth");
    VZ.raster(gt, S.D, 0, 0, ST.W * PX, ST.H * PX, { lo: 0, hi: ST.DMAX });

    /* ── D: the slant profile ───────────────────────────────────────────── */
    const gd = VZ.panelBox(g, 548, 214, 192, 144, "disparity along row y = " + y0 + ", across the slant");
    const xd = d3.scaleLinear().domain([64, 96]).range([0, 192]);
    const yd = d3.scaleLinear().domain([2, 16]).range([144, 0]);
    VZ.gridY(gd, yd, 192, 4);
    gd.append("g").attr("class", "axis").attr("transform", "translate(0,144)").call(d3.axisBottom(xd).ticks(4));
    gd.append("g").attr("class", "axis").call(d3.axisLeft(yd).ticks(5));
    const mk = (D, c, dash) => {
      const pts = [];
      for (let x = 64; x < 96; x++) pts.push([xd(x), yd(D[y0][x])]);
      VZ.poly(gd, pts, { stroke: c, w: 1.6, close: false, dash: dash || null });
    };
    mk(S.D, VC.good, "4 3"); mk(cur.box, VC.accent); mk(cur.shift, VC.violet);
    VZ.legend(gd, [{ color: VC.good, label: "truth", dash: "4 3" }, { color: VC.accent, label: "box" },
    { color: VC.violet, label: "shiftable" }], 4, 12, { gap: 12, font: 9 });

    /* the staircase width: how many x-steps between disparity changes, box */
    let steps = 0, runs = [];
    let last = cur.box[y0][66], run = 0;
    for (let x = 66; x < 94; x++) {
      if (cur.box[y0][x] === last) run++;
      else { runs.push(run); run = 1; last = cur.box[y0][x]; steps++; }
    }
    runs.push(run);
    const meanRun = d3.mean(runs);

    ro.html(
      `cost <b>${kind.toUpperCase()}</b>, radius <b>${r}</b> (${2 * r + 1} × ${2 * r + 1}).&nbsp; ` +
      `box: textured-far <b>${VZ.fmt(100 * cur.bFar, 1)}%</b> · near an edge <b>${VZ.fmt(100 * cur.bNear, 1)}%</b> · ` +
      `everything <b>${VZ.fmt(100 * cur.bAll, 1)}%</b><br>` +
      `shiftable windows: near an edge <b>${VZ.fmt(100 * cur.sNear, 1)}%</b> ` +
      (cur.sNear < cur.bNear
        ? `(a factor of <b>${VZ.fmt(cur.bNear / Math.max(cur.sNear, 1e-6), 1)}×</b> better) `
        : `(<b>${VZ.fmt(cur.sNear / Math.max(cur.bNear, 1e-6), 1)}×</b> WORSE — at this radius the min-filter finds a spuriously low cost somewhere in every neighbourhood) `) +
      `· everything <b>${VZ.fmt(100 * cur.sAll, 1)}%</b><br>` +
      `on the slanted plane the box result is a staircase of <b>${steps + 1}</b> treads averaging <b>${VZ.fmt(meanRun, 1)} px</b> wide, ` +
      `because a box window can only vote for one disparity at a time — the fronto-parallel assumption made visible`
    );
  }

  [["ag-r", v => "" + v], ["ag-y", v => "" + v]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  el("ag-k").addEventListener("change", draw);
  draw();
})();

/* ══════════ 16 · #sgm-svg — scanline optimisation and SGM ═══════════════ */
(function () {
  const svg = d3.select("#sgm-svg");
  if (svg.empty()) return;
  const W = 760, H = 452;
  const ro = d3.select("#sgm-readout");
  const el = id => document.getElementById(id);
  const S = ST.scene;
  const DIRS8 = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]];

  /* the semi-global recursion, exactly as §22 writes it */
  function sgm(C, nd, P1, P2) {
    const ND = C.length, Hh = ST.H, Ww = ST.W;
    const Acc = []; for (let d = 0; d < ND; d++) Acc.push(VZ.zeros2(Hh, Ww));
    for (const [dx, dy] of DIRS8.slice(0, nd)) {
      const L = []; for (let d = 0; d < ND; d++) L.push(VZ.zeros2(Hh, Ww));
      const xs = dx > 0 ? d3.range(Ww) : d3.range(Ww).reverse();
      const ys = dy > 0 ? d3.range(Hh) : d3.range(Hh).reverse();
      for (const y of ys) for (const x of xs) {
        const px = x - dx, py = y - dy;
        if (px < 0 || px >= Ww || py < 0 || py >= Hh) { for (let d = 0; d < ND; d++) L[d][y][x] = C[d][y][x]; continue; }
        let mprev = Infinity;
        for (let d = 0; d < ND; d++) if (L[d][py][px] < mprev) mprev = L[d][py][px];
        for (let d = 0; d < ND; d++) {
          let best = mprev + P2;
          const a = L[d][py][px]; if (a < best) best = a;
          if (d > 0) { const b = L[d - 1][py][px] + P1; if (b < best) best = b; }
          if (d < ND - 1) { const b = L[d + 1][py][px] + P1; if (b < best) best = b; }
          L[d][y][x] = C[d][y][x] + best - mprev;
        }
      }
      for (let d = 0; d < ND; d++) for (let y = 0; y < Hh; y++) for (let x = 0; x < Ww; x++) Acc[d][y][x] += L[d][y][x];
    }
    return Acc;
  }

  const VOL = {}, SWEEP = {};
  const vol = r => VOL[r] || (VOL[r] = ST.volume(S.L, S.R, "sad", r));
  const P2S = [8, 16, 24, 32, 48, 64, 88, 120, 160, 200];
  function sweep(r, P1) {
    const key = r + ":" + P1;
    if (SWEEP[key]) return SWEEP[key];
    const C = vol(r);
    const out = [1, 2, 4, 8].map(nd => P2S.map(p2 => ST.badRate(ST.wta(sgm(C, nd, P1, p2)), 1)));
    SWEEP[key] = out;
    return out;
  }

  function draw() {
    const nd = +el("sg-n").value, P1 = +el("sg-p1").value, P2 = +el("sg-p2").value, r = +el("sg-r").value;
    const C = vol(r);
    const Dw = ST.wta(C);
    const D1 = ST.wta(sgm(C, 1, P1, P2));
    const Dn = ST.wta(sgm(C, nd, P1, P2));
    const sw = sweep(r, P1);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    const PX = 2;
    const rast = (x, title, D) => {
      const gg = VZ.panelBox(g, x, 40, ST.W * PX, ST.H * PX, title);
      VZ.raster(gg, D, 0, 0, ST.W * PX, ST.H * PX, { lo: 0, hi: ST.DMAX });
      gg.append("text").attr("x", 0).attr("y", ST.H * PX + 14).attr("font-size", 9.5).attr("fill", VC.muted)
        .text("bad " + VZ.fmt(100 * ST.badRate(D, 1), 1) + "%");
      return gg;
    };
    rast(24, "winner-take-all", Dw);
    rast(230, "1 direction — streaks", D1);
    rast(436, nd + " direction" + (nd > 1 ? "s" : ""), Dn);

    /* the direction rose */
    const gr = VZ.panelBox(g, 654, 40, 84, 84, "directions");
    const cxr = 42, cyr = 42;
    DIRS8.forEach(([dx, dy], i) => {
      const on = i < nd;
      VZ.arrow(gr, cxr, cyr, cxr + dx * 30, cyr + dy * 30,
        { color: on ? VC.a2 : VC.line, w: on ? 1.8 : 1, head: on ? 6 : 4, op: on ? 1 : 0.5 });
    });
    gr.append("text").attr("x", 0).attr("y", 98).attr("font-size", 9.5).attr("fill", VC.muted)
      .text("summed, then argmin");

    /* ── bars: bad rate vs number of directions ─────────────────────────── */
    const gb = VZ.panelBox(g, 60, 254, 200, 150, "bad rate vs number of directions");
    const rates = [1, 2, 4, 8].map((n, i) => sw[i][P2S.indexOf(P2) >= 0 ? P2S.indexOf(P2) : 0]);
    const live = [1, 2, 4, 8].map(n => ST.badRate(ST.wta(sgm(C, n, P1, P2)), 1));
    const wtaR = ST.badRate(Dw, 1);
    const yb = d3.scaleLinear().domain([0, Math.max(wtaR, d3.max(live)) * 1.15]).range([150, 0]);
    const xb = d3.scaleBand().domain([0, 1, 2, 3]).range([0, 200]).padding(0.3);
    VZ.gridY(gb, yb, 200, 4);
    gb.append("g").attr("class", "axis").call(d3.axisLeft(yb).ticks(5).tickFormat(d3.format(".0%")));
    live.forEach((v, i) => {
      gb.append("rect").attr("x", xb(i)).attr("y", yb(v)).attr("width", xb.bandwidth()).attr("height", 150 - yb(v))
        .attr("fill", [1, 2, 4, 8][i] === nd ? VC.a2 : VC.accent).attr("fill-opacity", 0.85);
      gb.append("text").attr("x", xb(i) + xb.bandwidth() / 2).attr("y", 164).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", VC.muted).text([1, 2, 4, 8][i]);
      gb.append("text").attr("x", xb(i) + xb.bandwidth() / 2).attr("y", yb(v) - 4).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("fill", VC.ink).text(VZ.fmt(100 * v, 1));
    });
    gb.append("line").attr("x1", 0).attr("x2", 200).attr("y1", yb(wtaR)).attr("y2", yb(wtaR))
      .attr("stroke", VC.bad).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
    gb.append("text").attr("x", 2).attr("y", yb(wtaR) - 4).attr("font-size", 9).attr("fill", VC.bad)
      .text("winner-take-all " + VZ.fmt(100 * wtaR, 1) + "%");
    gb.append("text").attr("x", 2).attr("y", 178).attr("font-size", 9).attr("fill", VC.muted).text("number of directions");

    /* ── the P2 sweep ───────────────────────────────────────────────────── */
    const gc = VZ.panelBox(g, 330, 254, 380, 150, "bad rate vs P₂, for each direction count");
    const xc = d3.scaleLog().domain([8, 200]).range([0, 380]);
    const yc = d3.scaleLinear().domain([0, Math.max(0.2, d3.max(sw, a => d3.max(a)) * 1.1)]).range([150, 0]);
    VZ.gridY(gc, yc, 380, 4);
    gc.append("g").attr("class", "axis").attr("transform", "translate(0,150)").call(d3.axisBottom(xc).ticks(5, "~s"));
    gc.append("g").attr("class", "axis").call(d3.axisLeft(yc).ticks(5).tickFormat(d3.format(".0%")));
    const cols = [VC.bad, VC.a2, VC.accent, VC.good];
    sw.forEach((arr, i) => {
      VZ.poly(gc, arr.map((v, k) => [xc(P2S[k]), yc(v)]), { stroke: cols[i], w: 1.7, close: false });
      const best = arr.indexOf(d3.min(arr));
      gc.append("circle").attr("cx", xc(P2S[best])).attr("cy", yc(arr[best])).attr("r", 3).attr("fill", cols[i]);
    });
    gc.append("line").attr("x1", xc(VZ.clamp(P2, 8, 200))).attr("x2", xc(VZ.clamp(P2, 8, 200))).attr("y1", 0).attr("y2", 150)
      .attr("stroke", VC.violet).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
    VZ.legend(gc, [1, 2, 4, 8].map((n, i) => ({ color: cols[i], label: n + " dir" })), 320, 14, { gap: 13, font: 9.5 });
    gc.append("text").attr("x", 2).attr("y", 168).attr("font-size", 9).attr("fill", VC.muted)
      .text("P₂ (log) · dots mark each curve's minimum · P₁ = " + P1 + ", base window " + (2 * r + 1) + "×" + (2 * r + 1));

    const flatR = ST.badRate(Dn, 1, ST.REGIONS.textureless), strR = ST.badRate(Dn, 1, ST.REGIONS.repeated);
    /* streakiness: horizontal streaks make neighbouring ROWS disagree, so the mean
       vertical disparity gradient rises relative to the horizontal one */
    const aniso = D => {
      let gx = 0, gy = 0, n = 0;
      for (let y = 1; y < ST.H - 1; y++) for (let x = 1; x < ST.W - 1; x++) {
        gx += Math.abs(D[y][x + 1] - D[y][x - 1]); gy += Math.abs(D[y + 1][x] - D[y - 1][x]); n++;
      }
      return gx > 0 ? gy / gx : NaN;
    };
    const a1 = aniso(D1), an = aniso(Dn), at = aniso(S.D);
    ro.html(
      `base cost SAD on ${2 * r + 1} × ${2 * r + 1}, P₁ = <b>${P1}</b>, P₂ = <b>${P2}</b>, <b>${nd}</b> direction${nd > 1 ? "s" : ""}<br>` +
      `winner-take-all <b>${VZ.fmt(100 * wtaR, 1)}%</b> → 1 direction <b>${VZ.fmt(100 * live[0], 1)}%</b> → ` +
      `${nd} directions <b>${VZ.fmt(100 * ST.badRate(Dn, 1), 1)}%</b>, a factor of <b>${VZ.fmt(wtaR / Math.max(ST.badRate(Dn, 1), 1e-6), 1)}×</b><br>` +
      `and the two pathological regions: untextured patch <b>${VZ.fmt(100 * flatR, 1)}%</b>, periodic stripes <b>${VZ.fmt(100 * strR, 1)}%</b> ` +
      `— against ${VZ.fmt(100 * ST.badRate(Dw, 1, ST.REGIONS.textureless), 0)}% and ${VZ.fmt(100 * ST.badRate(Dw, 1, ST.REGIONS.repeated), 0)}% for winner-take-all. ` +
      `A region with no information of its own has borrowed some from its neighbours; that is the one thing a local method can never do.<br>` +
      `streaking, measured as the ratio of the mean vertical disparity gradient to the mean horizontal one: ground truth <b>${VZ.fmt(at, 3)}</b>, ` +
      `one direction <b>${VZ.fmt(a1, 3)}</b>, ${nd} direction${nd > 1 ? "s" : ""} <b>${VZ.fmt(an, 3)}</b> — a single scan order leaves rows disagreeing with their neighbours, and summing over directions removes it.`
    );
  }

  el("sg-n").addEventListener("change", draw);
  [["sg-p1", v => "" + v], ["sg-p2", v => "" + v], ["sg-r", v => "" + v]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  draw();
})();

/* ══════════ 17 · #sub-svg — sub-pixel fit, variance and bias ════════════ */
(function () {
  const svg = d3.select("#sub-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;
  const ro = d3.select("#sub-readout");
  const el = id => document.getElementById(id);
  const N = 600, C0 = 300;

  const BASE = {};
  const base = lam => BASE[lam] || (BASE[lam] = SS.scanline(N, 21, { kmin: 4, kmax: Math.max(6, Math.round(N / lam)), amp: 40, mean: 128 }));

  /* one trial: noisy left window, noisy right scanline, integer search then parabola */
  /* fixIdx = true reproduces the DERIVATION exactly: one refinement step from the
     integer nearest the truth. fixIdx = false is what a real matcher does: search
     for the integer winner first, then refine. They are different estimators and
     the figure measures both, because only the first one the formula describes. */
  function trial(b, dTrue, sig, win, rng, dLo, dHi, fixIdx) {
    const r = win >> 1;
    const idx = d3.range(-r, r + 1).map(i => C0 + i);
    const a = idx.map(t => SS.lerpAt(b, t) + sig * VZ.randn(rng));
    /* the right scanline, sampled on the integer grid, with its own noise */
    const need = {};
    for (let d = dLo; d <= dHi; d++) idx.forEach(t => { need[t - d] = 1; });
    const Rn = {};
    Object.keys(need).forEach(k => { const j = +k; Rn[j] = SS.lerpAt(b, j + dTrue) + sig * VZ.randn(rng); });
    const cost = d => { let s = 0; for (let i = 0; i < idx.length; i++) { const e = a[i] - Rn[idx[i] - d]; s += e * e; } return s; };
    let bd = dLo, bv = Infinity;
    if (fixIdx) bd = Math.round(dTrue);
    else for (let d = dLo; d <= dHi; d++) { const c = cost(d); if (c < bv) { bv = c; bd = d; } }
    if (bd === dLo || bd === dHi) return { d: bd, delta: 0, clamped: true, cs: null, bd: bd };
    const cs = [cost(bd - 1), cost(bd), cost(bd + 1)];
    const den = cs[0] - 2 * cs[1] + cs[2];
    let del = den > 1e-9 ? (cs[0] - cs[2]) / (2 * den) : 0;
    const clamped = Math.abs(del) > 0.5;
    del = VZ.clamp(del, -0.5, 0.5);
    return { d: bd + del, delta: del, clamped: clamped, cs: cs, bd: bd };
  }

  /* the gradient energy of the clean window, G = ∑ I_x² */
  function gradEnergy(b, win) {
    const r = win >> 1, v = d3.range(-r, r + 1).map(i => SS.lerpAt(b, C0 + i));
    let G = 0;
    for (let i = 0; i < v.length; i++) {
      const g = i === 0 ? v[1] - v[0] : (i === v.length - 1 ? v[i] - v[i - 1] : (v[i + 1] - v[i - 1]) / 2);
      G += g * g;
    }
    return G;
  }

  function draw() {
    const sig = +el("sb-s").value, lam = +el("sb-w").value, win = +el("sb-n").value, dT = +el("sb-d").value;
    const b = base(lam);
    const G = gradEnergy(b, win);
    const dLo = 3, dHi = 12;

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: one realisation, the three points, the parabola ─────────────── */
    const ga = VZ.panelBox(g, 52, 40, 210, 300, "SSD vs integer d, and the fitted parabola");
    const rr = VZ.rng(1234);
    const one = trial(b, dT, sig, win, rr, dLo, dHi);
    /* recompute the whole curve for drawing, with the same noise draw */
    const rr2 = VZ.rng(1234);
    const r0 = win >> 1, idx = d3.range(-r0, r0 + 1).map(i => C0 + i);
    const aa = idx.map(t => SS.lerpAt(b, t) + sig * VZ.randn(rr2));
    const need = {};
    for (let d = dLo; d <= dHi; d++) idx.forEach(t => { need[t - d] = 1; });
    const Rn = {};
    Object.keys(need).forEach(k => { const j = +k; Rn[j] = SS.lerpAt(b, j + dT) + sig * VZ.randn(rr2); });
    const curve = [];
    for (let d = dLo; d <= dHi; d++) {
      let s = 0; for (let i = 0; i < idx.length; i++) { const e = aa[i] - Rn[idx[i] - d]; s += e * e; }
      curve.push([d, s]);
    }
    const xs = d3.scaleLinear().domain([dLo, dHi]).range([0, 210]);
    const ys = d3.scaleLinear().domain([0, d3.max(curve, p => p[1]) * 1.08]).range([300, 0]);
    VZ.gridY(ga, ys, 210, 4);
    ga.append("g").attr("class", "axis").attr("transform", "translate(0,300)").call(d3.axisBottom(xs).ticks(6));
    ga.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5, "~s"));
    VZ.poly(ga, curve.map(p => [xs(p[0]), ys(p[1])]), { stroke: VC.muted, w: 1.2, close: false });
    curve.forEach(p => ga.append("circle").attr("cx", xs(p[0])).attr("cy", ys(p[1])).attr("r", 2).attr("fill", VC.muted));
    if (one.cs) {
      const [cm, c0v, cp] = one.cs, bd = one.bd;
      const A = (cm - 2 * c0v + cp) / 2, Bv = (cp - cm) / 2;
      const par = [];
      for (let t = -1.3; t <= 1.3; t += 0.05) par.push([xs(bd + t), ys(A * t * t + Bv * t + c0v)]);
      VZ.poly(ga, par, { stroke: VC.accent, w: 1.8, close: false });
      [-1, 0, 1].forEach((k, i) => ga.append("circle").attr("cx", xs(bd + k)).attr("cy", ys(one.cs[i])).attr("r", 4.5)
        .attr("fill", "none").attr("stroke", VC.accent).attr("stroke-width", 1.6));
      ga.append("line").attr("x1", xs(one.d)).attr("x2", xs(one.d)).attr("y1", 0).attr("y2", 300)
        .attr("stroke", VC.a2).attr("stroke-width", 1.4);
      ga.append("line").attr("x1", xs(dT)).attr("x2", xs(dT)).attr("y1", 0).attr("y2", 300)
        .attr("stroke", VC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3");
      VZ.legend(ga, [{ color: VC.a2, label: "estimate " + VZ.fmt(one.d, 3) },
      { color: VC.good, label: "truth " + VZ.fmt(dT, 2), dash: "4 3" }], 6, 316, { gap: 12, font: 9.5 });
    }

    /* ── B: measured σ_d against σ_I ────────────────────────────────────── */
    const gb = VZ.panelBox(g, 320, 40, 190, 300, "measured σ_d vs σ_I (log–log)");
    const SIGS = [0.1, 0.25, 0.5, 1, 2, 3, 4];
    const TR = 2500;
    const series = [true, false].map(fix => SIGS.map(s => {
      const rg = VZ.rng(77);
      let sum = 0, sum2 = 0, nc = 0;
      for (let t = 0; t < TR; t++) {
        const o = trial(b, dT, s, win, rg, dLo, dHi, fix);
        sum += o.d; sum2 += o.d * o.d; if (o.clamped) nc++;
      }
      const m = sum / TR, v = Math.max(sum2 / TR - m * m, 1e-12);
      return { s: s, sd: Math.sqrt(v), mean: m, pred: s * Math.sqrt(2 / G), clamp: nc / TR };
    }));
    const pts = series[0], ptsSearch = series[1];
    const xb = d3.scaleLog().domain([0.08, 5]).range([0, 190]);
    const yb = d3.scaleLog().domain([5e-3, 3]).range([300, 0]).clamp(true);
    gb.append("g").attr("class", "axis").attr("transform", "translate(0,300)").call(d3.axisBottom(xb).ticks(4, "~g"));
    gb.append("g").attr("class", "axis").call(d3.axisLeft(yb).ticks(5, "~e"));
    const pl = [];
    for (let s = 0.08; s <= 5; s *= 1.1) pl.push([xb(s), yb(VZ.clamp(s * Math.sqrt(2 / G), 5e-3, 3))]);
    VZ.poly(gb, pl, { stroke: VC.accent, w: 1.7, close: false, dash: "5 3" });
    pts.forEach(p => gb.append("circle").attr("cx", xb(p.s)).attr("cy", yb(VZ.clamp(p.sd, 5e-3, 3))).attr("r", 3)
      .attr("fill", p.clamp < 0.02 ? VC.good : VC.bad));
    ptsSearch.forEach(p => gb.append("rect").attr("x", xb(p.s) - 2.6).attr("y", yb(VZ.clamp(p.sd, 5e-3, 3)) - 2.6)
      .attr("width", 5.2).attr("height", 5.2).attr("fill", "none").attr("stroke", VC.violet).attr("stroke-width", 1.4));
    gb.append("line").attr("x1", xb(sig)).attr("x2", xb(sig)).attr("y1", 0).attr("y2", 300)
      .attr("stroke", VC.a2).attr("stroke-width", 1.1).attr("stroke-dasharray", "3 3");
    VZ.legend(gb, [{ color: VC.accent, label: "σ_I √(2/G)", dash: "5 3" }, { color: VC.good, label: "one step from truth" },
    { color: VC.bad, label: "same, clamp active" }, { color: VC.violet, label: "search, then refine" }], 6, 316, { gap: 12, font: 9.5 });
    gb.append("text").attr("x", 2).attr("y", -6).attr("font-size", 9.5).attr("fill", VC.muted)
      .text("G = ∑ I_x² = " + VZ.fmt(G, 1));

    /* ── C: the pixel-locking curve ─────────────────────────────────────── */
    const gc = VZ.panelBox(g, 566, 40, 174, 300, "bias vs true disparity");
    const DS = d3.range(0, 11).map(i => 7 + i / 10);
    const TR2 = 1500;
    const bias = DS.map(d => {
      const rg = VZ.rng(55);
      let sum = 0;
      for (let t = 0; t < TR2; t++) sum += trial(b, d, sig, win, rg, dLo, dHi).d;
      return { d: d, b: sum / TR2 - d };
    });
    const mxb = Math.max(0.02, d3.max(bias, p => Math.abs(p.b)) * 1.25);
    const xc = d3.scaleLinear().domain([7, 8]).range([0, 174]);
    const yc = d3.scaleLinear().domain([-mxb, mxb]).range([300, 0]);
    VZ.gridY(gc, yc, 174, 5);
    gc.append("g").attr("class", "axis").attr("transform", "translate(0,300)").call(d3.axisBottom(xc).ticks(5));
    gc.append("g").attr("class", "axis").call(d3.axisLeft(yc).ticks(5, "+.3f"));
    gc.append("line").attr("x1", 0).attr("x2", 174).attr("y1", yc(0)).attr("y2", yc(0))
      .attr("stroke", VC.line).attr("stroke-width", 1.2);
    VZ.poly(gc, bias.map(p => [xc(p.d), yc(VZ.clamp(p.b, -mxb, mxb))]), { stroke: VC.violet, w: 1.9, close: false });
    bias.forEach(p => gc.append("circle").attr("cx", xc(p.d)).attr("cy", yc(VZ.clamp(p.b, -mxb, mxb))).attr("r", 2.4).attr("fill", VC.violet));
    gc.append("text").attr("x", 2).attr("y", 316).attr("font-size", 9.5).attr("fill", VC.muted)
      .text("mean estimate − truth, px");

    const iAt = SIGS.reduce((bi, v, i) => Math.abs(v - sig) < Math.abs(SIGS[bi] - sig) ? i : bi, 0);
    const at = pts[iAt], atS = ptsSearch[iAt];
    const amp = d3.max(bias, p => p.b) - d3.min(bias, p => p.b);
    ro.html(
      `shortest wavelength <b>${lam} px</b>, window <b>${win} px</b>, G = ∑ I_x² = <b>${VZ.fmt(G, 1)}</b>, ` +
      `<span class="keep">σ</span>_I = <b>${VZ.fmt(sig, 2)}</b>, ${TR} trials per noise level<br>` +
      `at <span class="keep">σ</span>_I = ${VZ.fmt(at.s, 2)}: measured <span class="keep">σ</span>_d = <b>${VZ.fmt(at.sd, 5)} px</b> vs predicted ` +
      `<span class="keep">σ</span>_I√(2/G) = <b>${VZ.fmt(at.pred, 5)} px</b>, ratio <b>${VZ.fmt(at.sd / at.pred, 3)}</b>; ` +
      `the clamp fired on <b>${VZ.fmt(100 * at.clamp, 1)}%</b> of trials. The practical estimator — search for the integer ` +
      `winner first, then refine — gives <b>${VZ.fmt(atS.sd, 5)} px</b>, ratio <b>${VZ.fmt(atS.sd / atS.pred, 3)}</b><br>` +
      `pixel locking: the bias sweeps <b>${VZ.fmt(amp, 4)} px</b> peak to peak across one disparity pixel` +
      (amp > 0.03 ? ` — a clear S-curve pulling every estimate towards the nearest integer, because this texture has energy near the sampling limit.`
        : ` — essentially flat, because at ${lam} px the shortest wavelength is well above the sampling limit and the cost really is close to parabolic.`)
    );
  }

  [["sb-s", v => VZ.fmt(v, 1)], ["sb-w", v => v + " px"], ["sb-n", v => v + " px"], ["sb-d", v => VZ.fmt(v, 2)]]
    .forEach(([id, f]) => {
      el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
      el(id + "v").textContent = f(+el(id).value);
    });
  draw();
})();

/* ══════════ 18 · #fail-svg — every failure mode, scored ═════════════════ */
(function () {
  const svg = d3.select("#fail-svg");
  if (svg.empty()) return;
  const W = 760, H = 440;
  const ro = d3.select("#fail-readout");
  const el = id => document.getElementById(id);
  const S = ST.scene;
  const DIRS8 = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]];
  const HL = { x: 55, y: 50, r: 6 }, TRUE_D = 14;

  function sgm(C, nd, P1, P2) {
    const ND = C.length, Hh = ST.H, Ww = ST.W;
    const Acc = []; for (let d = 0; d < ND; d++) Acc.push(VZ.zeros2(Hh, Ww));
    for (const [dx, dy] of DIRS8.slice(0, nd)) {
      const L = []; for (let d = 0; d < ND; d++) L.push(VZ.zeros2(Hh, Ww));
      const xs = dx > 0 ? d3.range(Ww) : d3.range(Ww).reverse();
      const ys = dy > 0 ? d3.range(Hh) : d3.range(Hh).reverse();
      for (const y of ys) for (const x of xs) {
        const px = x - dx, py = y - dy;
        if (px < 0 || px >= Ww || py < 0 || py >= Hh) { for (let d = 0; d < ND; d++) L[d][y][x] = C[d][y][x]; continue; }
        let mp = Infinity;
        for (let d = 0; d < ND; d++) if (L[d][py][px] < mp) mp = L[d][py][px];
        for (let d = 0; d < ND; d++) {
          let best = mp + P2;
          const a = L[d][py][px]; if (a < best) best = a;
          if (d > 0) { const b = L[d - 1][py][px] + P1; if (b < best) best = b; }
          if (d < ND - 1) { const b = L[d + 1][py][px] + P1; if (b < best) best = b; }
          L[d][y][x] = C[d][y][x] + best - mp;
        }
      }
      for (let d = 0; d < ND; d++) for (let y = 0; y < Hh; y++) for (let x = 0; x < Ww; x++) Acc[d][y][x] += L[d][y][x];
    }
    return Acc;
  }
  const spec = (A, cx, cy, amp, rad) => A.map((row, y) => Array.from(row, (v, x) => {
    const d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
    return VZ.clamp(v + amp * Math.exp(-d2 / (2 * rad * rad)), 0, 255);
  }));
  const inHL = (x, y) => ((x - HL.x) ** 2 + (y - HL.y) ** 2) < 14 * 14;

  /* the occlusion band, as a region predicate */
  const occBand = (x, y) => S.OCC[y][x] === 1;

  const BASE = {};
  function baseline(kind) {
    if (!BASE[kind]) BASE[kind] = ST.badRate(ST.wta(ST.volume(S.L, S.R, kind, 2)), 1, inHL);
    return BASE[kind];
  }

  function draw() {
    const amp = +el("fl-a").value, shift = +el("fl-s").value, cross = el("fl-c").checked;
    /* panel B: regions, WTA vs SGM, on the clean pair */
    const C2 = ST.volume(S.L, S.R, "sad", 2), C1 = ST.volume(S.L, S.R, "sad", 1);
    let Dw = ST.wta(C2);
    const Dg = ST.wta(sgm(C1, 8, 8, 64));
    let dropped = null, nDrop = 0, occCaught = 0, occTot = 0;
    if (cross) {
      const DR = (function () {
        const D = VZ.zeros2(ST.H, ST.W);
        for (let y = 0; y < ST.H; y++) for (let xr = 0; xr < ST.W; xr++) {
          let bd = 0, bv = Infinity;
          for (let d = 0; d < C2.length; d++) { const xl = xr + d; if (xl >= ST.W) break; if (C2[d][y][xl] < bv) { bv = C2[d][y][xl]; bd = d; } }
          D[y][xr] = bd;
        }
        return D;
      })();
      dropped = VZ.zeros2(ST.H, ST.W);
      for (let y = 0; y < ST.H; y++) for (let x = 0; x < ST.W; x++) {
        const xr = x - Dw[y][x];
        const bad = xr < 0 || Math.abs(DR[y][xr] - Dw[y][x]) > 1;
        if (bad) { dropped[y][x] = 1; nDrop++; }
        if (S.OCC[y][x]) { occTot++; if (bad) occCaught++; }
      }
    }
    /* Occluded pixels are excluded from every region EXCEPT the occlusion band
       itself, where the whole point is that there is no right answer to find. */
    const scored = (D, pred, keepOcc) => {
      let bad = 0, n = 0;
      for (let y = 2; y < ST.H - 2; y++) for (let x = 2; x < ST.W - 2; x++) {
        if (!pred(x, y)) continue;
        if (!keepOcc && S.OCC[y][x]) continue;
        if (cross && dropped && dropped[y][x]) continue;
        n++; if (Math.abs(D[y][x] - S.D[y][x]) > 1) bad++;
      }
      return n ? bad / n : 0;
    };

    /* panel C: the specular attack */
    const L2 = spec(S.L, HL.x, HL.y, amp, HL.r);
    const R2 = spec(S.R, HL.x - TRUE_D - shift, HL.y, amp, HL.r);
    const KINDS = [["sad", "SAD", VC.a2], ["ssd", "SSD", VC.bad], ["ncc", "NCC", VC.accent], ["census", "census", VC.good]];
    const hlRates = KINDS.map(([k]) => ST.badRate(ST.wta(ST.volume(L2, R2, k, 2)), 1, inHL));
    const hlBase = KINDS.map(([k]) => baseline(k));

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: the annotated left image ────────────────────────────────────── */
    const PX = 2.4;
    const ga = VZ.panelBox(g, 24, 40, ST.W * PX, ST.H * PX, "the scene, with its problem regions");
    VZ.raster(ga, L2, 0, 0, ST.W * PX, ST.H * PX, { lo: 0, hi: 255 });
    const boxes = [
      [28, 22, 22, 22, "untextured", VC.bad],
      [66, 14, 28, 30, "repeated", VC.a2],
      [66, 44, 28, 20, "slanted", VC.accent],
      [12, 14, 10, 42, "occlusion band", VC.violet]
    ];
    boxes.forEach(([x, y, w, h, lb, c]) => {
      ga.append("rect").attr("x", x * PX).attr("y", y * PX).attr("width", w * PX).attr("height", h * PX)
        .attr("fill", "none").attr("stroke", c).attr("stroke-width", 1.4).attr("stroke-dasharray", "4 2");
      ga.append("text").attr("x", VZ.clamp(x * PX, 0, ST.W * PX - 60)).attr("y", y * PX - 3)
        .attr("font-size", 8.5).attr("fill", c).text(lb);
    });
    if (amp > 0) {
      ga.append("circle").attr("cx", HL.x * PX).attr("cy", HL.y * PX).attr("r", 14 * PX)
        .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.5);
      ga.append("text").attr("x", VZ.clamp((HL.x - 14) * PX, 0, ST.W * PX - 60)).attr("y", (HL.y + 17) * PX)
        .attr("font-size", 8.5).attr("fill", VC.good).text("specular");
    }

    /* ── B: per-region bars, WTA vs SGM ─────────────────────────────────── */
    const REGS = [["untextured", ST.REGIONS.textureless, false], ["repeated", ST.REGIONS.repeated, false],
    ["slanted", ST.REGIONS.slanted, false], ["occl. band", occBand, true], ["background", ST.REGIONS.background, false],
    ["all", () => true, false]];
    const gb = VZ.panelBox(g, 288, 40, 250, 300, "bad rate by region: WTA vs SGM");
    const yb = d3.scaleLinear().domain([0, 1]).range([300, 0]);
    const xb = d3.scaleBand().domain(d3.range(REGS.length)).range([0, 250]).padding(0.28);
    VZ.gridY(gb, yb, 250, 5);
    gb.append("g").attr("class", "axis").call(d3.axisLeft(yb).ticks(5).tickFormat(d3.format(".0%")));
    const wR = REGS.map(([, p, k]) => scored(Dw, p, k)), gR = REGS.map(([, p, k]) => scored(Dg, p, k));
    REGS.forEach((rg, i) => {
      const bw = xb.bandwidth() / 2;
      gb.append("rect").attr("x", xb(i)).attr("y", yb(wR[i])).attr("width", bw).attr("height", 300 - yb(wR[i]))
        .attr("fill", VC.bad).attr("fill-opacity", 0.8);
      gb.append("rect").attr("x", xb(i) + bw).attr("y", yb(gR[i])).attr("width", bw).attr("height", 300 - yb(gR[i]))
        .attr("fill", VC.good).attr("fill-opacity", 0.85);
      gb.append("text").attr("x", xb(i) + xb.bandwidth() / 2).attr("y", 314).attr("text-anchor", "middle")
        .attr("font-size", 8).attr("fill", VC.muted)
        .attr("transform", `rotate(-24, ${xb(i) + xb.bandwidth() / 2}, 314)`).text(rg[0]);
    });
    VZ.legend(gb, [{ color: VC.bad, label: "WTA, SAD 5×5" }, { color: VC.good, label: "SGM, 8 dir" }], 4, 12, { gap: 13, font: 9.5 });

    /* ── C: the specular attack ─────────────────────────────────────────── */
    const gc = VZ.panelBox(g, 570, 40, 168, 300, "bad rate inside the highlight");
    const yc = d3.scaleLinear().domain([0, Math.max(0.3, d3.max(hlRates) * 1.2)]).range([300, 0]);
    const xc = d3.scaleBand().domain(d3.range(4)).range([0, 168]).padding(0.3);
    VZ.gridY(gc, yc, 168, 5);
    gc.append("g").attr("class", "axis").call(d3.axisLeft(yc).ticks(5).tickFormat(d3.format(".0%")));
    hlRates.forEach((v, i) => {
      gc.append("rect").attr("x", xc(i)).attr("y", yc(v)).attr("width", xc.bandwidth()).attr("height", 300 - yc(v))
        .attr("fill", KINDS[i][2]).attr("fill-opacity", 0.85);
      gc.append("rect").attr("x", xc(i) - 3).attr("y", yc(hlBase[i])).attr("width", xc.bandwidth() + 6)
        .attr("height", 300 - yc(hlBase[i])).attr("fill", "none").attr("stroke", VC.muted)
        .attr("stroke-width", 1).attr("stroke-dasharray", "3 2");
      gc.append("text").attr("x", xc(i) + xc.bandwidth() / 2).attr("y", 314).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("fill", VC.muted).text(KINDS[i][1]);
    });
    gc.append("text").attr("x", 0).attr("y", 330).attr("font-size", 9).attr("fill", VC.muted)
      .text("dashed: no highlight");

    ro.html(
      `by region, bad &gt; 1 disparity${cross ? ", after cross-check" : ""}:&nbsp; ` +
      REGS.map((r, i) => `${r[0]} <b>${VZ.fmt(100 * wR[i], 1)}%</b> → SGM <b>${VZ.fmt(100 * gR[i], 1)}%</b>`).join(" · ") + `<br>` +
      (cross ? `cross-check dropped <b>${nDrop}</b> pixels (${VZ.fmt(100 * nDrop / (ST.W * ST.H), 1)}% of the image) and flagged ` +
        `<b>${VZ.fmt(100 * occCaught / Math.max(occTot, 1), 1)}%</b> of the ${occTot} genuinely half-occluded pixels<br>`
        : `${(function () { let n = 0; for (let y = 0; y < ST.H; y++)for (let x = 0; x < ST.W; x++)n += S.OCC[y][x]; return n; })()} left pixels ` +
        `(<b>11.1%</b>) are half-occluded and have no correct disparity at all — tick the cross-check box to find them<br>`) +
      `specular highlight strength <b>${amp}</b>, displaced <b>${shift} px</b> beyond the true disparity: inside it, ` +
      KINDS.map(([k, lb], i) => `${lb} <b>${VZ.fmt(100 * hlRates[i], 0)}%</b> (was ${VZ.fmt(100 * hlBase[i], 0)}%)`).join(" · ") +
      (amp > 0 ? ` — the two intensity-difference costs are wrecked; NCC is not, but nor is it repaired.` : ` — the baseline, with no highlight.`)
    );
  }

  [["fl-a", v => "" + v], ["fl-s", v => v + " px"]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  el("fl-c").addEventListener("change", draw);
  draw();
})();

/* ══════════ 19 · #fact-svg — affine factorization ═══════════════════════ */
(function () {
  const svg = d3.select("#fact-svg");
  if (svg.empty()) return;
  const W = 760, H = 416;
  const ro = d3.select("#fact-readout");
  const el = id => document.getElementById(id);
  const NPT = 30;

  function scene(M, eta, sig) {
    const r = VZ.rng(9), P = [];
    for (let i = 0; i < NPT; i++) P.push([VZ.randn(r), VZ.randn(r), VZ.randn(r)]);
    const mu = [0, 1, 2].map(k => P.reduce((s, p) => s + p[k], 0) / NPT);
    P.forEach(p => { for (let k = 0; k < 3; k++) p[k] -= mu[k]; });
    const Wm = [], tracks = [];
    for (let i = 0; i < NPT; i++) tracks.push([]);
    for (let j = 0; j < M; j++) {
      const R = VZ.mul(VZ.Ry(VZ.rad(-40 + 80 * j / Math.max(M - 1, 1))), VZ.Rx(VZ.rad(12 * Math.sin(j))));
      const rx = [], ry = [];
      for (let i = 0; i < NPT; i++) {
        const q = VZ.mv(R, P[i]), den = 1 + eta * q[2];
        const u = 90 * q[0] / den + 320 + sig * VZ.randn(r);
        const v = 90 * q[1] / den + 240 + sig * VZ.randn(r);
        rx.push(u); ry.push(v); tracks[i].push([u, v]);
      }
      Wm.push(rx); Wm.push(ry);
    }
    return { P: P, W: Wm, tracks: tracks };
  }

  function factor(Wm) {
    const N = Wm[0].length;
    const Wc = Wm.map(row => { const m = d3.mean(row); return row.map(v => v - m); });
    const { U, s, V } = SS.svd(Wc);
    const U3 = U.map(r => [r[0], r[1], r[2]]);
    const S3 = [0, 1, 2].map(k => V.map(r => r[k]).map(v => v * s[k]));
    return { U3: U3, S3: S3, s: s };
  }
  function metricQ(U3) {
    const bil = (u, v) => [u[0] * v[0], u[0] * v[1] + u[1] * v[0], u[0] * v[2] + u[2] * v[0],
    u[1] * v[1], u[1] * v[2] + u[2] * v[1], u[2] * v[2]];
    const rows = [];
    for (let j = 0; j < U3.length / 2; j++) {
      const a = U3[2 * j], b = U3[2 * j + 1];
      const A = bil(a, a), B = bil(b, b);
      rows.push(A.map((v, k) => v - B[k]));
      rows.push(bil(a, b));
    }
    const l = SS.col(SS.svd(rows).V, 5);
    let L = [[l[0], l[1], l[2]], [l[1], l[3], l[4]], [l[2], l[4], l[5]]];
    if (L[0][0] + L[1][1] + L[2][2] < 0) L = L.map(r => r.map(v => -v));
    const e = VZ.jacobiEig(L);
    const Q = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let k = 0; k < 3; k++) {
      const sq = Math.sqrt(Math.max(e.values[k], 0));
      for (let i = 0; i < 3; i++) Q[i][k] = e.vectors[k][i] * sq;
    }
    return Q;
  }
  /* best rotation + scale mapping A → B, both 3 × N and centred */
  function kabsch(A, B) {
    const N = A[0].length, Hm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < N; i++) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) Hm[a][b] += A[a][i] * B[b][i];
    const { U, V } = SS.svd(Hm);
    const R = VZ.mul(V, VZ.T(U));
    let num = 0, den = 0;
    for (let i = 0; i < N; i++) {
      const ra = VZ.mv(R, [A[0][i], A[1][i], A[2][i]]);
      for (let k = 0; k < 3; k++) { num += ra[k] * B[k][i]; den += ra[k] * ra[k]; }
    }
    return { R: R, s: den > 0 ? num / den : 1 };
  }

  function draw() {
    const eta = +el("fc-e").value, sig = +el("fc-s").value, M = +el("fc-m").value;
    const flip = el("fc-r").checked ? -1 : 1;
    const sc = scene(M, eta, sig);
    const f = factor(sc.W);
    const Q = metricQ(f.U3), Qi = VZ.inv3(Q);
    const Ptrue = [0, 1, 2].map(k => sc.P.map(p => p[k]));
    let Shat = Qi ? VZ.mul(Qi, f.S3) : null;
    let rms = NaN, det = NaN, best = null;
    if (Shat) {
      /* the reflection is a genuine ambiguity: try both and report the better */
      let bE = Infinity;
      [1, -1].forEach(fl => {
        const Sf = Shat.map((row, k) => k === 2 ? row.map(v => fl * v) : row.slice());
        const k2 = kabsch(Sf, Ptrue);
        let e2 = 0;
        const al = [];
        for (let i = 0; i < NPT; i++) {
          const ra = VZ.scale(VZ.mv(k2.R, [Sf[0][i], Sf[1][i], Sf[2][i]]), k2.s);
          al.push(ra);
          for (let k = 0; k < 3; k++) e2 += (ra[k] - Ptrue[k][i]) ** 2;
        }
        const e = Math.sqrt(e2 / NPT);
        if (e < bE) { bE = e; best = { al: al, det: VZ.det3(k2.R), fl: fl }; }
      });
      rms = bE; det = best.det;
      if (flip === -1) {
        /* draw the OTHER solution: same shape, depth reversed */
        const Sf = Shat.map((row, k) => k === 2 ? row.map(v => -best.fl * v) : row.slice());
        const k2 = kabsch(Sf, Ptrue);
        best = {
          al: d3.range(NPT).map(i => VZ.scale(VZ.mv(k2.R, [Sf[0][i], Sf[1][i], Sf[2][i]]), k2.s)),
          det: VZ.det3(k2.R), fl: -best.fl
        };
      }
    }

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: the tracks ──────────────────────────────────────────────────── */
    const ga = VZ.panelBox(g, 24, 40, 230, 320, "the feature tracks — the raw data");
    const allx = sc.tracks.flat().map(p => p[0]), ally = sc.tracks.flat().map(p => p[1]);
    const xa = d3.scaleLinear().domain([d3.min(allx) - 6, d3.max(allx) + 6]).range([0, 230]);
    const ya = d3.scaleLinear().domain([d3.min(ally) - 6, d3.max(ally) + 6]).range([320, 0]);
    sc.tracks.forEach(t => {
      VZ.poly(ga, t.map(p => [xa(p[0]), ya(p[1])]), { stroke: VC.accent, w: 0.9, op: 0.5, close: false });
      ga.append("circle").attr("cx", xa(t[0][0])).attr("cy", ya(t[0][1])).attr("r", 1.8).attr("fill", VC.a2);
    });
    ga.append("text").attr("x", 0).attr("y", 336).attr("font-size", 9.5).attr("fill", VC.muted)
      .text(NPT + " points × " + M + " frames — Ŵ is " + (2 * M) + " × " + NPT);

    /* ── B: the singular values ─────────────────────────────────────────── */
    const gb = VZ.panelBox(g, 300, 40, 180, 320, "singular values of the centred Ŵ");
    const nsv = Math.min(f.s.length, 8);
    const yb = d3.scaleLog().domain([Math.max(1e-13, f.s[0] * 1e-16), f.s[0] * 2]).range([320, 0]).clamp(true);
    gb.append("g").attr("class", "axis").call(d3.axisLeft(yb).ticks(7, "~e"));
    const xb = d3.scaleBand().domain(d3.range(nsv)).range([0, 180]).padding(0.24);
    for (let k = 0; k < nsv; k++) {
      const v = Math.max(f.s[k], f.s[0] * 1e-16);
      gb.append("rect").attr("x", xb(k)).attr("y", yb(v)).attr("width", xb.bandwidth()).attr("height", 320 - yb(v))
        .attr("fill", k < 3 ? VC.good : VC.bad).attr("fill-opacity", 0.85);
      gb.append("text").attr("x", xb(k) + xb.bandwidth() / 2).attr("y", 334).attr("text-anchor", "middle")
        .attr("font-size", 8.5).attr("fill", VC.muted).text(k + 1);
    }
    gb.append("text").attr("x", 0).attr("y", 350).attr("font-size", 10)
      .attr("fill", f.s[3] / f.s[2] < 1e-6 ? VC.good : VC.bad)
      .text("σ₄/σ₃ = " + (f.s[3] / f.s[2]).toExponential(2));

    /* ── C: the recovered shape ─────────────────────────────────────────── */
    const gc = VZ.panelBox(g, 528, 40, 210, 320, "recovered shape vs truth (plan view)");
    const xs = d3.scaleLinear().domain([-3.2, 3.2]).range([0, 210]);
    const ys = d3.scaleLinear().domain([-3.2, 3.2]).range([320, 0]);
    VZ.gridY(gc, ys, 210, 5); VZ.gridX(gc, xs, 320, 5);
    gc.append("g").attr("class", "axis").attr("transform", "translate(0,320)").call(d3.axisBottom(xs).ticks(5));
    gc.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5));
    if (best) {
      for (let i = 0; i < NPT; i++) {
        const a = best.al[i];
        gc.append("line").attr("x1", xs(Ptrue[0][i])).attr("y1", ys(Ptrue[2][i]))
          .attr("x2", xs(VZ.clamp(a[0], -3.2, 3.2))).attr("y2", ys(VZ.clamp(a[2], -3.2, 3.2)))
          .attr("stroke", VC.bad).attr("stroke-width", 1);
        gc.append("circle").attr("cx", xs(Ptrue[0][i])).attr("cy", ys(Ptrue[2][i])).attr("r", 2.6)
          .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.3);
        gc.append("circle").attr("cx", xs(VZ.clamp(a[0], -3.2, 3.2))).attr("cy", ys(VZ.clamp(a[2], -3.2, 3.2)))
          .attr("r", 2).attr("fill", VC.a2);
      }
    }
    VZ.legend(gc, [{ color: VC.good, label: "ground truth" }, { color: VC.a2, label: "recovered" }], 4, 12, { gap: 13, font: 9.5 });

    let rmsShown = rms;
    if (best && flip === -1) {
      let e2 = 0;
      for (let i = 0; i < NPT; i++) for (let k = 0; k < 3; k++) e2 += (best.al[i][k] - Ptrue[k][i]) ** 2;
      rmsShown = Math.sqrt(e2 / NPT);
    }
    ro.html(
      `<span class="keep">η</span> = 1/Z₀ = <b>${VZ.fmt(eta, 3)}</b>${eta === 0 ? " (exact scaled orthography)" : " → Z₀ = " + VZ.fmt(1 / eta, 1) + " for an object about 2 units across"}, ` +
      `noise <b>${VZ.fmt(sig, 1)} px</b>, ${M} frames<br>` +
      `singular values ${f.s.slice(0, 5).map(v => VZ.sig(v, 4)).join(", ")}, … &nbsp; ` +
      `<b>σ₄/σ₃ = ${(f.s[3] / f.s[2]).toExponential(2)}</b>` +
      (f.s[3] / f.s[2] < 1e-6 ? " — the measurement matrix is exactly rank 3, as the theorem says"
        : (f.s[3] / f.s[2] < 0.02 ? " — still safely rank 3 for practical purposes"
          : " — <b>no longer rank 3</b>: the affine camera model is being violated and the reconstruction below is degrading")) + `<br>` +
      `shape RMS after similarity alignment: <b>${VZ.fmt(rmsShown, 5)}</b> on points of unit scale` +
      (flip === -1 ? `. This is the <b>depth-reversed</b> solution: it reproduces every image measurement exactly as well as the other one, and the data cannot choose between them.`
        : `, and the alignment that achieves it has determinant <b>${VZ.fmt(det, 2)}</b>` +
        (det < 0 ? ` — a REFLECTION. The recovered shape is the mirror image; tick the box to see the other solution.` : `.`))
    );
  }

  [["fc-e", v => VZ.fmt(v, 3)], ["fc-s", v => VZ.fmt(v, 1) + " px"], ["fc-m", v => "" + v]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  el("fc-r").addEventListener("change", draw);
  draw();
})();

/* ══════════ the bundle-adjustment problem, shared by §27 and §28 ════════ */
const BA = (function () {
  const f = 500, cx = 320, cy = 240, M = 6, N = 25;
  const rodr = w => { const th = VZ.norm(w); return th < 1e-12 ? VZ.eye(3) : VZ.rodrigues(w, th); };

  /* the visibility pattern: camera j sees a sliding window of points, so the
     Hessian has real structure rather than being dense (§28 needs that) */
  const sees = (j, i) => Math.abs(i - (j * (N - 8) / (M - 1) + 4)) <= 7.5;

  function problem(pert, noise, full) {
    const r = VZ.rng(5), P = [];
    for (let i = 0; i < N; i++) P.push([-1 + 2 * r(), -1 + 2 * r(), -1 + 2 * r()]);
    const cams = [];
    for (let j = 0; j < M; j++) {
      const a = VZ.rad(-30 + 60 * j / (M - 1));
      const C = [4 * Math.sin(a), 0.4 * Math.cos(3 * a), -4 * Math.cos(a)];
      const w = [0.02 * j, a, 0.01 * j];
      cams.push({ w: w, t: VZ.neg(VZ.mv(rodr(w), C)), C: C });
    }
    const obs = [];
    for (let j = 0; j < M; j++) for (let i = 0; i < N; i++) {
      if (!full && !sees(j, i)) continue;
      const X = VZ.add(VZ.mv(rodr(cams[j].w), P[i]), cams[j].t);
      if (X[2] <= 0.2) continue;
      obs.push({ j: j, i: i, u: f * X[0] / X[2] + cx + noise * VZ.randn(r), v: f * X[1] / X[2] + cy + noise * VZ.randn(r) });
    }
    const p0 = [], ptrue = [];
    cams.forEach(c => {
      c.w.forEach(v => { ptrue.push(v); p0.push(v + pert * 0.02 * VZ.randn(r)); });
      c.t.forEach(v => { ptrue.push(v); p0.push(v + pert * 0.05 * VZ.randn(r)); });
    });
    P.forEach(p => p.forEach(v => { ptrue.push(v); p0.push(v + pert * 0.08 * VZ.randn(r)); }));
    return { obs: obs, p0: p0, ptrue: ptrue, M: M, N: N, P: P, cams: cams, f: f, cx: cx, cy: cy };
  }

  function resid(pr, p) {
    const out = new Float64Array(2 * pr.obs.length);
    const Rs = [];
    for (let j = 0; j < pr.M; j++) Rs.push(rodr([p[6 * j], p[6 * j + 1], p[6 * j + 2]]));
    for (let k = 0; k < pr.obs.length; k++) {
      const o = pr.obs[k], b = 6 * pr.M + 3 * o.i;
      const X = VZ.add(VZ.mv(Rs[o.j], [p[b], p[b + 1], p[b + 2]]), [p[6 * o.j + 3], p[6 * o.j + 4], p[6 * o.j + 5]]);
      const z = Math.max(X[2], 1e-6);
      out[2 * k] = f * X[0] / z + cx - o.u;
      out[2 * k + 1] = f * X[1] / z + cy - o.v;
    }
    return out;
  }
  const cost = (pr, p) => { const r = resid(pr, p); let s = 0; for (let i = 0; i < r.length; i++) s += r[i] * r[i]; return s; };

  /* Jacobian by central differences — only the blocks that can be non-zero are
     touched, which is both faster and the sparsity pattern §28 draws */
  function jac(pr, p) {
    const m = 2 * pr.obs.length, n = p.length, J = [];
    for (let i = 0; i < m; i++) J.push(new Float64Array(n));
    const h = 1e-6;
    for (let k = 0; k < n; k++) {
      const pp = p.slice(), pm = p.slice(); pp[k] += h; pm[k] -= h;
      const a = resid(pr, pp), b = resid(pr, pm);
      for (let i = 0; i < m; i++) J[i][k] = (a[i] - b[i]) / (2 * h);
    }
    return J;
  }
  function normalEq(J, r) {
    const m = J.length, n = J[0].length;
    const A = []; for (let a = 0; a < n; a++) A.push(new Float64Array(n));
    const b = new Float64Array(n);
    for (let i = 0; i < m; i++) {
      const Ji = J[i];
      for (let a = 0; a < n; a++) {
        const ja = Ji[a]; if (ja === 0) continue;
        b[a] -= ja * r[i];
        for (let c = a; c < n; c++) A[a][c] += ja * Ji[c];
      }
    }
    for (let a = 0; a < n; a++) for (let c = 0; c < a; c++) A[a][c] = A[c][a];
    return { A: A, b: b };
  }

  function run(pr, lam0, iters) {
    let p = pr.p0.slice(), lam = lam0;
    const hist = [{ p: p.slice(), c: cost(pr, p), lam: lam, note: "" }];
    for (let it = 0; it < iters; it++) {
      const J = jac(pr, p), r = resid(pr, p), { A, b } = normalEq(J, r);
      const n = p.length;
      if (lam0 === 0) {                                   /* pure Gauss–Newton */
        const Ai = VZ.invN(A.map(row => Array.from(row)));
        if (!Ai) { hist.push({ p: p.slice(), c: hist[hist.length - 1].c, lam: 0, note: "singular" }); break; }
        const dp = VZ.mv(Ai, Array.from(b));
        p = p.map((v, k) => v + dp[k]);
        hist.push({ p: p.slice(), c: cost(pr, p), lam: 0, note: "" });
        continue;
      }
      let ok = false;
      for (let tries = 0; tries < 8 && !ok; tries++) {
        const Ad = A.map((row, a) => Array.from(row, (v, c) => c === a ? v * (1 + lam) : v));
        const Ai = VZ.invN(Ad);
        if (!Ai) { lam *= 10; continue; }
        const dp = VZ.mv(Ai, Array.from(b));
        const pn = p.map((v, k) => v + dp[k]);
        const cn = cost(pr, pn);
        if (cn < hist[hist.length - 1].c) { p = pn; lam = Math.max(lam / 3, 1e-12); ok = true; hist.push({ p: p.slice(), c: cn, lam: lam, note: "" }); }
        else lam *= 10;
      }
      if (!ok) { hist.push({ p: p.slice(), c: hist[hist.length - 1].c, lam: lam, note: "no step" }); break; }
      if (hist.length > 2 && hist[hist.length - 2].c - hist[hist.length - 1].c < 1e-8 * hist[hist.length - 1].c) break;
    }
    return hist;
  }

  return { problem: problem, resid: resid, cost: cost, jac: jac, normalEq: normalEq, run: run, rodr: rodr, sees: sees, f: f, cx: cx, cy: cy };
})();

/* ══════════ 20 · #ba-svg — bundle adjustment converging ═════════════════ */
(function () {
  const svg = d3.select("#ba-svg");
  if (svg.empty()) return;
  const W = 760, H = 404;
  const ro = d3.select("#ba-readout");
  const el = id => document.getElementById(id);
  const CACHE = {};

  function solve(pert, noise, gn) {
    const key = pert + ":" + noise + ":" + gn;
    if (CACHE[key]) return CACHE[key];
    const pr = BA.problem(pert, noise, true);
    const hist = BA.run(pr, gn ? 0 : 1e-3, 10);
    const out = { pr: pr, hist: hist, ctrue: BA.cost(pr, pr.ptrue) };
    CACHE[key] = out;
    return out;
  }

  function draw() {
    const pert = +el("ba-p").value, noise = +el("ba-n").value, gn = el("ba-g").checked;
    const S = solve(pert, noise, gn);
    const it = VZ.clamp(+el("ba-i").value, 0, S.hist.length - 1);
    const st = S.hist[it];
    const m = 2 * S.pr.obs.length;

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: the cost curve ──────────────────────────────────────────────── */
    const ga = VZ.panelBox(g, 58, 40, 210, 310, "cost ‖r‖² vs iteration (log)");
    const xs = d3.scaleLinear().domain([0, Math.max(S.hist.length - 1, 1)]).range([0, 210]);
    const lo = Math.max(1e-2, d3.min(S.hist, h => h.c) * 0.5), hi = d3.max(S.hist, h => h.c) * 2;
    const ys = d3.scaleLog().domain([lo, hi]).range([310, 0]).clamp(true);
    VZ.gridY(ga, ys, 210, 5);
    ga.append("g").attr("class", "axis").attr("transform", "translate(0,310)").call(d3.axisBottom(xs).ticks(Math.min(S.hist.length, 8)));
    ga.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(6, "~e"));
    VZ.poly(ga, S.hist.map((h, i) => [xs(i), ys(h.c)]), { stroke: VC.accent, w: 2, close: false });
    S.hist.forEach((h, i) => ga.append("circle").attr("cx", xs(i)).attr("cy", ys(h.c)).attr("r", 3)
      .attr("fill", i === it ? VC.a2 : VC.accent));
    ga.append("line").attr("x1", 0).attr("x2", 210).attr("y1", ys(S.ctrue)).attr("y2", ys(S.ctrue))
      .attr("stroke", VC.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "5 3");
    ga.append("text").attr("x", 4).attr("y", ys(S.ctrue) - 5).attr("font-size", 9.5).attr("fill", VC.good)
      .text("cost at the true parameters " + VZ.fmt(S.ctrue, 2));
    ga.append("text").attr("x", 0).attr("y", 328).attr("font-size", 9.5).attr("fill", VC.muted)
      .text("iteration · λ = " + (st.lam === 0 ? "0" : st.lam.toExponential(1)));

    /* ── B: the residuals in one camera ─────────────────────────────────── */
    const gb = VZ.panelBox(g, 300, 40, 200, 310, "camera 2 · observed ○, predicted •, residual ×10", { fill: VC.bg });
    const r = BA.resid(S.pr, st.p);
    const cam = 2;
    const us = [], vs = [];
    S.pr.obs.forEach(o => { if (o.j === cam) { us.push(o.u); vs.push(o.v); } });
    const pad = 40;
    const xb = d3.scaleLinear().domain([d3.min(us) - pad, d3.max(us) + pad]).range([0, 200]);
    const yb = d3.scaleLinear().domain([d3.min(vs) - pad, d3.max(vs) + pad]).range([0, 310]);
    const cp = VZ.clip(svg, "ba-clip", 0, 0, 200, 310);
    const inner = gb.append("g").attr("clip-path", cp);
    let maxRes = 0;
    S.pr.obs.forEach((o, k) => {
      if (o.j !== cam) return;
      const du = r[2 * k], dv = r[2 * k + 1];
      maxRes = Math.max(maxRes, Math.hypot(du, dv));
      const px = xb(o.u), py = yb(o.v);
      const qx = xb(o.u + 10 * du), qy = yb(o.v + 10 * dv);
      inner.append("line").attr("x1", px).attr("y1", py).attr("x2", qx).attr("y2", qy)
        .attr("stroke", VC.bad).attr("stroke-width", 1.1);
      inner.append("circle").attr("cx", px).attr("cy", py).attr("r", 3).attr("fill", "none")
        .attr("stroke", VC.good).attr("stroke-width", 1.2);
      inner.append("circle").attr("cx", qx).attr("cy", qy).attr("r", 1.8).attr("fill", VC.a2);
    });
    gb.append("text").attr("x", 0).attr("y", 326).attr("font-size", 9.5).attr("fill", VC.muted)
      .text("largest residual in this view: " + VZ.fmt(maxRes, 3) + " px");

    /* ── C: the reconstruction, plan view ───────────────────────────────── */
    const gc = VZ.panelBox(g, 542, 40, 200, 310, "plan view · truth vs current estimate");
    const xs2 = d3.scaleLinear().domain([-5.2, 5.2]).range([0, 200]);
    const ys2 = d3.scaleLinear().domain([-5.2, 5.2]).range([310, 0]);
    VZ.gridY(gc, ys2, 200, 5); VZ.gridX(gc, xs2, 310, 5);
    gc.append("g").attr("class", "axis").attr("transform", "translate(0,310)").call(d3.axisBottom(xs2).ticks(5));
    gc.append("g").attr("class", "axis").call(d3.axisLeft(ys2).ticks(5));
    for (let i = 0; i < S.pr.N; i++) {
      const b = 6 * S.pr.M + 3 * i;
      gc.append("circle").attr("cx", xs2(S.pr.P[i][0])).attr("cy", ys2(S.pr.P[i][2])).attr("r", 2.6)
        .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.1);
      gc.append("circle").attr("cx", xs2(VZ.clamp(st.p[b], -5.2, 5.2))).attr("cy", ys2(VZ.clamp(st.p[b + 2], -5.2, 5.2)))
        .attr("r", 2).attr("fill", VC.a2);
    }
    for (let j = 0; j < S.pr.M; j++) {
      const w = [st.p[6 * j], st.p[6 * j + 1], st.p[6 * j + 2]];
      const t = [st.p[6 * j + 3], st.p[6 * j + 4], st.p[6 * j + 5]];
      const C = VZ.neg(VZ.mv(VZ.T(BA.rodr(w)), t));
      gc.append("path").attr("d", "M0,0 l-6,-10 l12,0 Z")
        .attr("transform", `translate(${xs2(VZ.clamp(C[0], -5.2, 5.2))},${ys2(VZ.clamp(C[2], -5.2, 5.2))})`)
        .attr("fill", VC.a2).attr("fill-opacity", 0.9);
      gc.append("path").attr("d", "M0,0 l-6,-10 l12,0 Z")
        .attr("transform", `translate(${xs2(S.pr.cams[j].C[0])},${ys2(S.pr.cams[j].C[2])})`)
        .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 1.1);
    }
    VZ.legend(gc, [{ color: VC.good, label: "truth" }, { color: VC.a2, label: "estimate" }], 4, 12, { gap: 13, font: 9.5 });

    const rms = Math.sqrt(st.c / m);
    const p = S.pr.p0.length, expect = noise * noise * (m - (p - 7));
    ro.html(
      `${S.pr.M} cameras, ${S.pr.N} points, ${S.pr.obs.length} observations → <b>${m}</b> residuals and <b>${p}</b> parameters. ` +
      `Initial error <b>×${VZ.fmt(pert, 1)}</b>, image noise <b>${VZ.fmt(noise, 1)} px</b>.<br>` +
      (gn && S.hist.some(h => h.note === "singular")
        ? `<b>Gauss–Newton cannot take a single step.</b> JᵀJ is exactly singular — a 7-dimensional null space — so the ` +
        `normal equations have no unique solution and the inversion fails outright. This is not a numerical accident; it ` +
        `is the gauge freedom of §29, and Levenberg–Marquardt's λ·diag term is what removes it.`
        : `iteration <b>${it}</b> of ${S.hist.length - 1}: cost <b>${VZ.fmt(st.c, 4)}</b>, RMS reprojection <b>${VZ.fmt(rms, 5)} px</b>, ` +
        `λ = ${st.lam === 0 ? "0" : st.lam.toExponential(1)}<br>` +
        `the true parameters score <b>${VZ.fmt(S.ctrue, 4)}</b> (${VZ.fmt(Math.sqrt(S.ctrue / m), 4)} px RMS) — the converged estimate scores ` +
        `<b>${VZ.fmt(d3.min(S.hist, h => h.c), 4)}</b>, i.e. <b>${d3.min(S.hist, h => h.c) < S.ctrue ? "below" : "above"}</b> the truth, because a fitted ` +
        `model absorbs some of the noise. Theory says E‖r‖² = σ²(m − p) = <b>${VZ.fmt(expect, 1)}</b>.`)
    );
  }

  ["ba-i"].forEach(id => { });
  [["ba-i", v => "" + v], ["ba-p", v => VZ.fmt(v, 1)], ["ba-n", v => VZ.fmt(v, 1) + " px"]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  el("ba-g").addEventListener("change", draw);
  draw();
})();

/* ══════════ 21 · #spar-svg — sparsity and the Schur complement ══════════ */
(function () {
  const svg = d3.select("#spar-svg");
  if (svg.empty()) return;
  const W = 760, H = 446;
  const ro = d3.select("#spar-readout");
  const el = id => document.getElementById(id);

  function build(M, N, track) {
    /* camera j sees a contiguous window of `track` points, sliding across */
    const vis = [];
    for (let j = 0; j < M; j++) {
      const c = (N - track) * j / Math.max(M - 1, 1);
      const row = [];
      for (let i = 0; i < N; i++) row.push(i >= Math.floor(c) && i < Math.floor(c) + track ? 1 : 0);
      vis.push(row);
    }
    const obs = [];
    for (let j = 0; j < M; j++) for (let i = 0; i < N; i++) if (vis[j][i]) obs.push([j, i]);
    /* a structural Jacobian: random blocks in the right places is enough for a
       sparsity study, and keeps the figure independent of the geometry */
    const r = VZ.rng(3);
    const n = 3 * N + 6 * M, m = 2 * obs.length;
    const J = [];
    for (let k = 0; k < m; k++) J.push(new Float64Array(n));
    obs.forEach(([j, i], k) => {
      for (let a = 0; a < 2; a++) {
        for (let b = 0; b < 3; b++) J[2 * k + a][3 * i + b] = VZ.randn(r);
        for (let b = 0; b < 6; b++) J[2 * k + a][3 * N + 6 * j + b] = VZ.randn(r);
      }
    });
    /* A = JᵀJ, lightly damped so A_pp is invertible even for a point in one view */
    const A = [];
    for (let a = 0; a < n; a++) A.push(new Float64Array(n));
    for (let k = 0; k < m; k++) {
      const Jk = J[k];
      const nz = [];
      for (let a = 0; a < n; a++) if (Jk[a] !== 0) nz.push(a);
      nz.forEach(a => nz.forEach(b => { A[a][b] += Jk[a] * Jk[b]; }));
    }
    for (let a = 0; a < n; a++) A[a][a] += 1e-3;
    /* the Schur complement on the camera block */
    const nc = 6 * M;
    const Acc = [], Apc = [];
    for (let a = 0; a < nc; a++) { Acc.push(new Float64Array(nc)); }
    for (let a = 0; a < nc; a++) for (let b = 0; b < nc; b++) Acc[a][b] = A[3 * N + a][3 * N + b];
    const S = Acc.map(row => Array.from(row));
    for (let i = 0; i < N; i++) {
      const blk = [[A[3 * i][3 * i], A[3 * i][3 * i + 1], A[3 * i][3 * i + 2]],
      [A[3 * i + 1][3 * i], A[3 * i + 1][3 * i + 1], A[3 * i + 1][3 * i + 2]],
      [A[3 * i + 2][3 * i], A[3 * i + 2][3 * i + 1], A[3 * i + 2][3 * i + 2]]];
      const inv = VZ.inv3(blk);
      if (!inv) continue;
      /* the 3 × nc coupling row for this point */
      const B = [0, 1, 2].map(a => Array.from({ length: nc }, (_, b) => A[3 * i + a][3 * N + b]));
      const IB = VZ.mul(inv, B);                       // 3 × nc
      for (let a = 0; a < nc; a++) for (let b = 0; b < nc; b++) {
        let s = 0;
        for (let k2 = 0; k2 < 3; k2++) s += B[k2][a] * IB[k2][b];
        S[a][b] -= s;
      }
    }
    const blockNZ = (Mx, bs, nb) => {
      const P = [];
      for (let a = 0; a < nb; a++) {
        const row = [];
        for (let b = 0; b < nb; b++) {
          let mx = 0;
          for (let p = 0; p < bs; p++) for (let q = 0; q < bs; q++) mx = Math.max(mx, Math.abs(Mx[bs * a + p][bs * b + q]));
          row.push(mx > 1e-8 ? 1 : 0);
        }
        P.push(row);
      }
      return P;
    };
    const before = blockNZ(Acc, 6, M), after = blockNZ(S, 6, M);
    const share = [];
    for (let a = 0; a < M; a++) {
      const row = [];
      for (let b = 0; b < M; b++) { let s = 0; for (let i = 0; i < N; i++) if (vis[a][i] && vis[b][i]) s = 1; row.push(s); }
      share.push(row);
    }
    let match = true;
    for (let a = 0; a < M; a++) for (let b = 0; b < M; b++) if (after[a][b] !== share[a][b]) match = false;
    const dens = Mx => { let n2 = 0; Mx.forEach(r2 => r2.forEach(v => { if (Math.abs(v) > 1e-8) n2++; })); return n2 / (Mx.length * Mx.length); };
    return {
      vis: vis, obs: obs, J: J, A: A, Acc: Acc, S: S, before: before, after: after,
      share: share, match: match, M: M, N: N,
      densJ: (function () { let c = 0; J.forEach(r2 => r2.forEach(v => { if (v !== 0) c++; })); return c / (m * n); })(),
      densA: dens(A), densAcc: dens(Acc), densS: dens(S), n: n, m: m, nc: nc
    };
  }

  function draw() {
    const track = +el("sp-w").value, M = +el("sp-m").value, N = +el("sp-n").value;
    const B = build(M, N, Math.min(track, N));

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    g.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: the bipartite graph ─────────────────────────────────────────── */
    const ga = VZ.panelBox(g, 24, 40, 210, 120, "the bipartite visibility graph");
    const px = i => 8 + (210 - 16) * i / Math.max(N - 1, 1);
    const cxp = j => 8 + (210 - 16) * j / Math.max(M - 1, 1);
    B.obs.forEach(([j, i]) => ga.append("line").attr("x1", px(i)).attr("y1", 22).attr("x2", cxp(j)).attr("y2", 98)
      .attr("stroke", VC.line).attr("stroke-width", 0.7));
    for (let i = 0; i < N; i++) ga.append("circle").attr("cx", px(i)).attr("cy", 22).attr("r", 3).attr("fill", VC.violet);
    for (let j = 0; j < M; j++) ga.append("rect").attr("x", cxp(j) - 4).attr("y", 94).attr("width", 8).attr("height", 8).attr("fill", VC.a2);
    ga.append("text").attr("x", 0).attr("y", 12).attr("font-size", 9).attr("fill", VC.violet).text(N + " points");
    ga.append("text").attr("x", 0).attr("y", 116).attr("font-size", 9).attr("fill", VC.a2).text(M + " cameras");

    /* ── B: J sparsity ──────────────────────────────────────────────────── */
    const gb = VZ.panelBox(g, 24, 200, 210, 210, "J sparsity  ·  " + B.m + " × " + B.n);
    VZ.cells(gb, 0, 0, 1, B.n, B.m, (x, y) => B.J[y][x] !== 0 ? VC.accent : VC.panel2);
    gb.append("rect").attr("x", 0).attr("y", 0).attr("width", B.n).attr("height", B.m)
      .attr("fill", "none").attr("stroke", VC.line);
    gb.append("line").attr("x1", 3 * N).attr("x2", 3 * N).attr("y1", 0).attr("y2", B.m)
      .attr("stroke", VC.a2).attr("stroke-width", 1).attr("stroke-dasharray", "2 2");
    gb.append("text").attr("x", 0).attr("y", B.m + 12).attr("font-size", 9).attr("fill", VC.muted)
      .text("points | cameras  ·  density " + VZ.fmt(100 * B.densJ, 1) + "%");

    /* ── C: the Hessian ─────────────────────────────────────────────────── */
    const cw = Math.min(2.2, 220 / B.n);
    const gc = VZ.panelBox(g, 274, 40, B.n * cw, B.n * cw, "A = JᵀJ  ·  the arrowhead");
    VZ.cells(gc, 0, 0, cw, B.n, B.n, (x, y) => Math.abs(B.A[y][x]) > 1e-8 ? VC.accent : VC.panel2);
    gc.append("rect").attr("x", 3 * N * cw).attr("y", 3 * N * cw).attr("width", B.nc * cw).attr("height", B.nc * cw)
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.4);
    gc.append("text").attr("x", 0).attr("y", B.n * cw + 13).attr("font-size", 9).attr("fill", VC.muted)
      .text("A_pp block-diag · A_cc boxed · density " + VZ.fmt(100 * B.densA, 1) + "%");

    /* ── D: the reduced camera system ───────────────────────────────────── */
    const bw = Math.min(20, 190 / M);
    const gd = VZ.panelBox(g, 520, 40, M * bw, M * bw, "A′_cc after elimination · 6×6 blocks");
    VZ.cells(gd, 0, 0, bw, M, M, (x, y) =>
      B.before[y][x] ? VC.accent : (B.after[y][x] ? VC.bad : VC.panel2));
    VZ.legend(gd, [{ color: VC.accent, label: "already non-zero" }, { color: VC.bad, label: "fill-in" }],
      0, M * bw + 14, { gap: 12, font: 9.5 });

    const ge = VZ.panelBox(g, 520, 40 + M * bw + 56, 210, 150, "the check, and the saving");
    const kv = VZ.kv(ge, 4, 18, { keyW: 128, size: 10.5, lead: 17 });
    kv("full system size", (B.n) + " × " + B.n, VC.ink);
    kv("reduced system size", B.nc + " × " + B.nc, VC.good, true);
    kv("A_cc density before", VZ.fmt(100 * B.densAcc, 1) + "%", VC.ink);
    kv("A′_cc density after", VZ.fmt(100 * B.densS, 1) + "%", VC.ink);
    kv("fill-in = shared point?", B.match ? "YES, exactly" : "MISMATCH", B.match ? VC.good : VC.bad, true);

    const denseFlops = Math.pow(B.n, 3) / 3, schurFlops = N * 27 + B.obs.length * 3 * 6 * 6 + Math.pow(B.nc, 3) / 3;
    ro.html(
      `${M} cameras, ${N} points, each camera seeing a run of ${Math.min(track, N)} points → <b>${B.obs.length}</b> observations, ` +
      `<b>${B.m}</b> residuals, <b>${B.n}</b> parameters<br>` +
      `J is <b>${VZ.fmt(100 * B.densJ, 1)}%</b> dense; A = JᵀJ is <b>${VZ.fmt(100 * B.densA, 1)}%</b> dense; ` +
      `A_cc is <b>${VZ.fmt(100 * B.densAcc, 1)}%</b> dense before elimination and <b>${VZ.fmt(100 * B.densS, 1)}%</b> after<br>` +
      `<b>fill-in check: ${B.match ? "every block created by the elimination corresponds to a pair of cameras sharing a point, and no other block was created — an exact match" : "PATTERN MISMATCH"}</b><br>` +
      `factorising the full system costs about n³/3 ≈ <b>${d3.format(".3~s")(denseFlops)}</b> operations; ` +
      `eliminating the points first costs about <b>${d3.format(".3~s")(schurFlops)}</b>, a saving of ` +
      `<b>${VZ.fmt(denseFlops / schurFlops, 1)}×</b> at this size — and the ratio grows like (N/M)³ as the point count rises`
    );
  }

  [["sp-w", v => "" + v], ["sp-m", v => "" + v], ["sp-n", v => "" + v]].forEach(([id, f]) => {
    el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
    el(id + "v").textContent = f(+el(id).value);
  });
  draw();
})();

/* ══════════ 22 · #drift-svg — drift and loop closure ════════════════════ */
(function () {
  const svg = d3.select("#drift-svg");
  if (svg.empty()) return;
  const W = 760, H = 428;
  const ro = d3.select("#drift-readout");
  const el = id => document.getElementById(id);

  const wrap = a => { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; };
  const rel = (a, b) => {
    const c = Math.cos(a[2]), s = Math.sin(a[2]), dx = b[0] - a[0], dy = b[1] - a[1];
    return [c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])];
  };
  const compose = (a, z) => {
    const c = Math.cos(a[2]), s = Math.sin(a[2]);
    return [a[0] + c * z[0] - s * z[1], a[1] + s * z[0] + c * z[1], wrap(a[2] + z[2])];
  };
  const loopTruth = n => d3.range(n).map(k => {
    const t = 2 * Math.PI * k / n;
    return [6 * Math.cos(t), 6 * Math.sin(t), wrap(t + Math.PI / 2)];
  });

  function build(n, st, sr, bias, seed) {
    const T = loopTruth(n), r = VZ.rng(seed), E = [T[0].slice()], edges = [];
    for (let k = 0; k < n - 1; k++) {
      const z = rel(T[k], T[k + 1]);
      const zn = [z[0] + st * VZ.randn(r), z[1] + st * VZ.randn(r), wrap(z[2] + sr * VZ.randn(r) + bias)];
      edges.push({ i: k, j: k + 1, z: zn });
      E.push(compose(E[k], zn));
    }
    const zc = rel(T[n - 1], T[0]);
    edges.push({ i: n - 1, j: 0, loop: true, z: [zc[0] + st * VZ.randn(r), zc[1] + st * VZ.randn(r), wrap(zc[2] + sr * VZ.randn(r))] });
    return { T: T, E: E, edges: edges, n: n };
  }
  const resid = (P, edges) => {
    const out = [];
    edges.forEach(e => { const r0 = rel(P[e.i], P[e.j]); out.push(r0[0] - e.z[0], r0[1] - e.z[1], wrap(r0[2] - e.z[2])); });
    return out;
  };
  /* Gauss–Newton on the pose graph, with pose 0 held fixed — that is the gauge */
  function optimise(g, iters) {
    const n = g.n, np = 3 * n;
    let P = g.E.map(p => p.slice());
    for (let it = 0; it < iters; it++) {
      const m = 3 * g.edges.length, J = [];
      for (let i = 0; i < m; i++) J.push(new Float64Array(np));
      const r0 = resid(P, g.edges), h = 1e-6;
      for (let k = 3; k < np; k++) {
        const Pp = P.map(p => p.slice());
        Pp[Math.floor(k / 3)][k % 3] += h;
        const rp = resid(Pp, g.edges);
        for (let i = 0; i < m; i++) J[i][k] = (rp[i] - r0[i]) / h;
      }
      const A = []; for (let a = 0; a < np; a++) A.push(new Float64Array(np));
      const b = new Float64Array(np);
      for (let i = 0; i < m; i++) {
        const Ji = J[i];
        for (let a = 0; a < np; a++) {
          const ja = Ji[a]; if (ja === 0) continue;
          b[a] -= ja * r0[i];
          for (let c = 0; c < np; c++) A[a][c] += ja * Ji[c];
        }
      }
      for (let a = 0; a < 3; a++) A[a][a] = 1;
      for (let a = 0; a < np; a++) A[a][a] += 1e-6;
      const Ai = VZ.invN(A.map(r2 => Array.from(r2)));
      if (!Ai) break;
      const dp = VZ.mv(Ai, Array.from(b));
      for (let k = 3; k < np; k++) P[Math.floor(k / 3)][k % 3] += dp[k];
      for (let k = 0; k < n; k++) P[k][2] = wrap(P[k][2]);
    }
    return P;
  }

  /* the growth-law measurement, cached because it is a few thousand walks */
  let LAWS = null;
  function laws() {
    if (LAWS) return LAWS;
    const NS = [10, 20, 40, 80, 160, 320];
    const walk = (n, st, sr, bias, seed) => {
      const r = VZ.rng(seed);
      let T = [0, 0, 0], E = [0, 0, 0];
      for (let k = 0; k < n; k++) {
        T = compose(T, [1, 0, 0]);
        E = compose(E, [1 + st * VZ.randn(r), st * VZ.randn(r), sr * VZ.randn(r) + bias]);
      }
      return Math.hypot(E[0] - T[0], E[1] - T[1]);
    };
    const series = [
      { lb: "translation noise only", c: VC.good, f: (n, s) => walk(n, 0.02, 0, 0, s), pred: 0.5 },
      { lb: "rotation noise", c: VC.a2, f: (n, s) => walk(n, 0.02, 0.005, 0, s), pred: 1.5 },
      { lb: "rotation bias", c: VC.bad, f: (n, s) => walk(n, 0.02, 0.005, 0.002, s), pred: 2 }
    ].map(S => {
      const pts = NS.map(n => {
        let acc = 0; const M = 300;
        for (let t = 0; t < M; t++) { const d = S.f(n, t + 1); acc += d * d; }
        return [n, Math.sqrt(acc / M)];
      });
      const xs = pts.map(p => Math.log(p[0])), ys = pts.map(p => Math.log(p[1]));
      const mx = d3.mean(xs), my = d3.mean(ys);
      let num = 0, den = 0;
      xs.forEach((x, i) => { num += (x - mx) * (ys[i] - my); den += (x - mx) * (x - mx); });
      return Object.assign({}, S, { pts: pts, slope: num / den });
    });
    LAWS = series;
    return LAWS;
  }

  function draw() {
    const sr = +el("dr-r").value, bias = +el("dr-b").value, st = +el("dr-t").value;
    const n = +el("dr-n").value, close = el("dr-c").checked;
    const g = build(n, st, sr, bias, 7);
    const P = close ? optimise(g, 8) : null;
    const err = Q => Math.sqrt(Q.reduce((s, p, k) => s + (p[0] - g.T[k][0]) ** 2 + (p[1] - g.T[k][1]) ** 2, 0) / n);
    const eo = err(g.E), ec = P ? err(P) : NaN;
    const gap = Math.hypot(g.E[n - 1][0] - g.T[n - 1][0], g.E[n - 1][1] - g.T[n - 1][1]);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const gr = fr.g;
    gr.append("rect").attr("width", W).attr("height", H).attr("fill", VC.panel);

    /* ── A: the trajectory ──────────────────────────────────────────────── */
    const all = g.T.concat(g.E).concat(P || []);
    const R = Math.max(8, d3.max(all, p => Math.max(Math.abs(p[0]), Math.abs(p[1]))) * 1.12);
    const ga = VZ.panelBox(gr, 30, 40, 250, 320, "plan view of the loop");
    const xs = d3.scaleLinear().domain([-R, R]).range([0, 250]);
    const ys = d3.scaleLinear().domain([-R * 320 / 250, R * 320 / 250]).range([320, 0]);
    VZ.gridY(ga, ys, 250, 5); VZ.gridX(ga, xs, 320, 5);
    ga.append("g").attr("class", "axis").attr("transform", "translate(0,320)").call(d3.axisBottom(xs).ticks(5));
    ga.append("g").attr("class", "axis").call(d3.axisLeft(ys).ticks(5));
    const path = Q => Q.map(p => [xs(p[0]), ys(p[1])]);
    VZ.poly(ga, path(g.T).concat([path(g.T)[0]]), { stroke: VC.good, w: 1.8, close: false, dash: "5 3" });
    VZ.poly(ga, path(g.E), { stroke: VC.bad, w: 1.8, close: false });
    if (P) VZ.poly(ga, path(P).concat([path(P)[0]]), { stroke: VC.accent, w: 1.8, close: false });
    ga.append("line").attr("x1", xs(g.E[n - 1][0])).attr("y1", ys(g.E[n - 1][1]))
      .attr("x2", xs(g.T[0][0])).attr("y2", ys(g.T[0][1]))
      .attr("stroke", VC.violet).attr("stroke-width", 1.4).attr("stroke-dasharray", "2 2");
    ga.append("circle").attr("cx", xs(g.T[0][0])).attr("cy", ys(g.T[0][1])).attr("r", 4)
      .attr("fill", "none").attr("stroke", VC.violet).attr("stroke-width", 1.5);
    VZ.legend(ga, [{ color: VC.good, label: "truth", dash: "5 3" }, { color: VC.bad, label: "dead reckoning" }]
      .concat(P ? [{ color: VC.accent, label: "after loop closure" }] : []), 4, 336, { gap: 12, font: 9.5 });

    /* ── B: error along the trajectory ──────────────────────────────────── */
    const gb = VZ.panelBox(gr, 322, 40, 190, 320, "position error vs pose index");
    const xb = d3.scaleLinear().domain([0, n - 1]).range([0, 190]);
    const eArr = g.E.map((p, k) => Math.hypot(p[0] - g.T[k][0], p[1] - g.T[k][1]));
    const cArr = P ? P.map((p, k) => Math.hypot(p[0] - g.T[k][0], p[1] - g.T[k][1])) : null;
    const yb = d3.scaleLinear().domain([0, Math.max(d3.max(eArr), 1e-6) * 1.15]).range([320, 0]);
    VZ.gridY(gb, yb, 190, 5);
    gb.append("g").attr("class", "axis").attr("transform", "translate(0,320)").call(d3.axisBottom(xb).ticks(5));
    gb.append("g").attr("class", "axis").call(d3.axisLeft(yb).ticks(5));
    VZ.poly(gb, eArr.map((v, k) => [xb(k), yb(v)]), { stroke: VC.bad, w: 1.8, close: false });
    if (cArr) VZ.poly(gb, cArr.map((v, k) => [xb(k), yb(v)]), { stroke: VC.accent, w: 1.8, close: false });
    gb.append("text").attr("x", 0).attr("y", 336).attr("font-size", 9.5).attr("fill", VC.muted)
      .text("open-loop RMS " + VZ.fmt(eo, 4) + (P ? "  →  closed " + VZ.fmt(ec, 4) : ""));

    /* ── C: the growth laws ─────────────────────────────────────────────── */
    const L = laws();
    const gc = VZ.panelBox(gr, 556, 40, 180, 320, "end-point error vs n (log–log)");
    const xc = d3.scaleLog().domain([8, 400]).range([0, 180]);
    const yc = d3.scaleLog().domain([0.05, 300]).range([320, 0]).clamp(true);
    gc.append("g").attr("class", "axis").attr("transform", "translate(0,320)").call(d3.axisBottom(xc).ticks(4, "~s"));
    gc.append("g").attr("class", "axis").call(d3.axisLeft(yc).ticks(5, "~e"));
    L.forEach(S => {
      VZ.poly(gc, S.pts.map(p => [xc(p[0]), yc(VZ.clamp(p[1], 0.05, 300))]), { stroke: S.c, w: 1.8, close: false });
      S.pts.forEach(p => gc.append("circle").attr("cx", xc(p[0])).attr("cy", yc(VZ.clamp(p[1], 0.05, 300))).attr("r", 2.4).attr("fill", S.c));
    });
    VZ.legend(gc, L.map(S => ({ color: S.c, label: S.lb + "  " + VZ.fmt(S.slope, 2) })), 4, 336, { gap: 12, font: 9 });
    gc.append("text").attr("x", 0).attr("y", 380).attr("font-size", 9).attr("fill", VC.muted)
      .text("predicted ½ · 3⁄2 · 2");

    ro.html(
      `${n} poses on a closed loop; per step <span class="keep">σ</span>_t = <b>${VZ.fmt(st, 3)}</b>, <span class="keep">σ</span>_θ = <b>${VZ.fmt(sr, 3)} rad</b>, ` +
      `rotation bias <b>${VZ.fmt(bias, 3)} rad</b><br>` +
      `open-loop RMS position error <b>${VZ.fmt(eo, 4)}</b>; the last pose misses the first by <b>${VZ.fmt(gap, 4)}</b><br>` +
      (P ? `after one loop-closure edge and 8 Gauss–Newton iterations: RMS <b>${VZ.fmt(ec, 4)}</b>, a factor of ` +
        `<b>${VZ.fmt(eo / Math.max(ec, 1e-9), 1)}×</b>. The middle panel shows where the improvement went: <b>everywhere</b>, ` +
        `not only at the end — the closure constraint is distributed backwards over the whole loop.`
        : `Tick "close the loop" to add one constraint between the last pose and the first.`) + `<br>` +
      `growth exponents measured over 300 walks each: ` +
      L.map(S => `${S.lb} <b>${VZ.fmt(S.slope, 3)}</b> (predicted ${S.pred})`).join(" · ")
    );
  }

  [["dr-r", v => VZ.fmt(v, 3)], ["dr-b", v => VZ.fmt(v, 3)], ["dr-t", v => VZ.fmt(v, 3)], ["dr-n", v => "" + v]]
    .forEach(([id, f]) => {
      el(id).addEventListener("input", () => { el(id + "v").textContent = f(+el(id).value); draw(); });
      el(id + "v").textContent = f(+el(id).value);
    });
  el("dr-c").addEventListener("change", draw);
  draw();
})();
