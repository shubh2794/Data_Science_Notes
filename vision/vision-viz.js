/* vision-viz.js — shared D3 + numeric helpers for the Computer Vision series (vision/*).
   Loaded after ../data.js → ../notes.js and before each part's own "<part>.viz.js".
   Nothing here draws anything by itself; it is a toolbox the part pages call.

   It is the vision analogue of math/statistics/stats-viz.js. Two globals:

   VC — palette, matching the custom properties in notes.css
        (accent, a2, good, bad, ink, muted, line, grid, panel, panel2, bg, violet, teal)

   VZ — namespace:
     · random      VZ.rng(seed) → mulberry32; VZ.randn(r); VZ.poisson(lam, r)
     · scalars     VZ.clamp, VZ.lerp, VZ.linspace, VZ.deg, VZ.rad, VZ.fmt, VZ.sig
     · vectors     VZ.add, VZ.sub, VZ.scale, VZ.dot, VZ.cross, VZ.norm, VZ.unit, VZ.neg
     · matrices    VZ.mat(rows), VZ.mul(A,B), VZ.mv(A,v), VZ.T(A), VZ.eye(n),
                   VZ.det3, VZ.inv3, VZ.inv4, VZ.mul4, VZ.mv4
                   — general dense products; the linear algebra itself is taught in
                     math/linear-algebra-*.html, this is only the arithmetic
     · homogeneous VZ.hom(p), VZ.dehom(x), VZ.applyH(H, p) (3×3 on a 2D point),
                   VZ.applyH4(M, p) (4×4 on a 3D point), VZ.lineThrough(p, q),
                   VZ.meet(l, m), VZ.normLine(l)
     · 2D groups   VZ.T2(tx,ty), VZ.R2(th,tx,ty), VZ.S2(s,th,tx,ty),
                   VZ.A2(a,b,c,d,tx,ty), VZ.H2(h)  — all returned as 3×3
     · 3D rotation VZ.Rx, VZ.Ry, VZ.Rz, VZ.skew(n), VZ.rodrigues(axis, th),
                   VZ.eulerZYX(a,b,g), VZ.eulerFromZYX(R), VZ.axisAngleFromR(R),
                   VZ.qFromAxisAngle(axis, th), VZ.qToR(q), VZ.qMul, VZ.qConj,
                   VZ.qNorm, VZ.qFromR(R), VZ.slerp(q0, q1, a), VZ.orthonormalise(R)
                   — quaternions are stored (x, y, z, w), w last, unit length
     · cameras     VZ.K({f, fx, fy, cx, cy, s, a}) → 3×3
                   VZ.camera({K, R, t}) → {P (3×4), project(p) → [u, v, z], K, R, t, C}
                   VZ.lookAt(eye, target, up) → R with rows (right, down, forward)
                   VZ.fovFromF(f, W) and VZ.fFromFov(fovDeg, W) — degrees, pixels
                   VZ.distort(x, y, k) and VZ.undistort(x, y, k) — normalised coords,
                     k = {k1, k2, k3, p1, p2}; undistort is the fixed-point inverse
     · 3-D preview VZ.view3(opts) → a small perspective viewer for the 3D panels:
                     {project(p) → {x, y, z}, eye, R, opts}
                   VZ.painter(faces, view) — back-to-front depth sort
                   VZ.axes3(g, view, len, labels) — draw a world coordinate frame
                   VZ.gridPlane(view, n, step, z) → array of screen-space segments
                   VZ.cubeMesh(cx, cy, cz, s) → {verts, edges, faces}
     · drawing     VZ.frame(sel, W, H, m), VZ.axisB, VZ.axisL, VZ.gridX, VZ.gridY,
                   VZ.legend(g, items, x, y), VZ.panel(g, x, y, w, h, title),
                   VZ.clip(svg, id, x, y, w, h), VZ.poly(g, pts, opt), VZ.arrow(g, ...)

   House rules: every consuming page supplies its own <svg width height viewBox role
   aria-label>; these helpers only ever append into an svg that already exists.       */

const VC = {
  accent: "#5b9cff", a2: "#ffb454", good: "#4ade80", bad: "#f87171",
  ink: "#e6e9ef", muted: "#9aa3b2", line: "#2a2f3a",
  grid: "#1b2130", panel: "#171a23", panel2: "#1e222d", bg: "#0f1117",
  violet: "#c084fc", teal: "#2dd4bf", rose: "#fb7185", lime: "#a3e635"
};

const VZ = (function () {

  /* ── scalars ───────────────────────────────────────────────────────────── */
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const deg = r => r * 180 / Math.PI;
  const rad = d => d * Math.PI / 180;
  function linspace(a, b, k) {
    const out = [];
    for (let i = 0; i < k; i++) out.push(a + (b - a) * (k === 1 ? 0 : i / (k - 1)));
    return out;
  }
  function fmt(x, d) {
    if (!isFinite(x)) return "—";
    const dd = (d === undefined) ? 2 : d;
    const s = x.toFixed(dd);
    return (parseFloat(s) === 0) ? (0).toFixed(dd) : s;      // never print "-0.00"
  }
  const sig = (x, n) => isFinite(x) ? Number(x.toPrecision(n || 3)).toString() : "—";

  /* ── seeded randomness, so every reader sees the same picture ──────────── */
  function rng(seed) {                       // mulberry32
    let a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(r) {                        // Box–Muller, one draw
    let u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  /* Knuth for small means, a normal approximation above 30 — accurate enough for
     photon counts, where the whole point is that the variance equals the mean. */
  function poisson(lam, r) {
    if (lam <= 0) return 0;
    if (lam > 30) return Math.max(0, Math.round(lam + Math.sqrt(lam) * randn(r)));
    const L = Math.exp(-lam);
    let k = 0, p = 1;
    do { k++; p *= r(); } while (p > L);
    return k - 1;
  }

  /* ── small vectors, as plain arrays ────────────────────────────────────── */
  const add = (a, b) => a.map((v, i) => v + b[i]);
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const neg = a => a.map(v => -v);
  const scale = (a, s) => a.map(v => v * s);
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const norm = a => Math.sqrt(dot(a, a));
  function unit(a) { const n = norm(a); return n > 1e-15 ? scale(a, 1 / n) : a.slice(); }
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

  /* ── dense matrices, as arrays of row arrays ───────────────────────────── */
  const mat = rows => rows.map(r => r.slice());
  function eye(n) {
    const A = [];
    for (let i = 0; i < n; i++) { A.push(new Array(n).fill(0)); A[i][i] = 1; }
    return A;
  }
  function T(A) {                            // transpose
    const m = A.length, n = A[0].length, B = [];
    for (let j = 0; j < n; j++) { B.push([]); for (let i = 0; i < m; i++) B[j].push(A[i][j]); }
    return B;
  }
  function mul(A, B) {                       // (m×k)(k×n)
    const m = A.length, k = B.length, n = B[0].length, C = [];
    for (let i = 0; i < m; i++) {
      const row = new Array(n).fill(0);
      for (let p = 0; p < k; p++) { const a = A[i][p]; if (a === 0) continue; for (let j = 0; j < n; j++) row[j] += a * B[p][j]; }
      C.push(row);
    }
    return C;
  }
  const mv = (A, v) => A.map(row => row.reduce((s, a, j) => s + a * v[j], 0));
  const det3 = A =>
    A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
    - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
    + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  function inv3(A) {
    const d = det3(A);
    if (Math.abs(d) < 1e-14) return null;
    const c = [
      [A[1][1] * A[2][2] - A[1][2] * A[2][1], A[0][2] * A[2][1] - A[0][1] * A[2][2], A[0][1] * A[1][2] - A[0][2] * A[1][1]],
      [A[1][2] * A[2][0] - A[1][0] * A[2][2], A[0][0] * A[2][2] - A[0][2] * A[2][0], A[0][2] * A[1][0] - A[0][0] * A[1][2]],
      [A[1][0] * A[2][1] - A[1][1] * A[2][0], A[0][1] * A[2][0] - A[0][0] * A[2][1], A[0][0] * A[1][1] - A[0][1] * A[1][0]]
    ];
    return c.map(r => r.map(v => v / d));
  }
  /* Gauss–Jordan with partial pivoting — used for the 4×4 camera matrices where a
     closed-form adjugate is unreadable and the matrix is never large. */
  function invN(A) {
    const n = A.length;
    const M = A.map((r, i) => r.concat(eye(n)[i]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (Math.abs(M[p][c]) < 1e-14) return null;
      const tmp = M[c]; M[c] = M[p]; M[p] = tmp;
      const pv = M[c][c];
      for (let j = 0; j < 2 * n; j++) M[c][j] /= pv;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c];
        if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
      }
    }
    return M.map(r => r.slice(n));
  }
  const inv4 = invN;

  /* ── homogeneous coordinates ───────────────────────────────────────────── */
  const hom = p => p.concat([1]);
  function dehom(x) {
    const w = x[x.length - 1];
    if (Math.abs(w) < 1e-14) return null;             // an ideal point: no finite image
    return x.slice(0, x.length - 1).map(v => v / w);
  }
  function applyH(H, p) { return dehom(mv(H, hom(p))); }
  function applyH4(M, p) { return dehom(mv(M, hom(p))); }
  const lineThrough = (p, q) => cross(hom(p), hom(q));
  const meet = (l, m) => cross(l, m);
  function normLine(l) {                     // (n̂ₓ, n̂ᵧ, d) with ‖n̂‖ = 1
    const n = Math.hypot(l[0], l[1]);
    return n < 1e-14 ? l.slice() : l.map(v => v / n);
  }

  /* ── the 2D transformation groups, all as 3×3 homogeneous matrices ─────── */
  const T2 = (tx, ty) => [[1, 0, tx], [0, 1, ty], [0, 0, 1]];
  function R2(th, tx, ty) {
    const c = Math.cos(th), s = Math.sin(th);
    return [[c, -s, tx || 0], [s, c, ty || 0], [0, 0, 1]];
  }
  function S2(s, th, tx, ty) {
    const c = s * Math.cos(th), d = s * Math.sin(th);
    return [[c, -d, tx || 0], [d, c, ty || 0], [0, 0, 1]];
  }
  const A2 = (a, b, c, d, tx, ty) => [[a, b, tx || 0], [c, d, ty || 0], [0, 0, 1]];
  const H2 = h => [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8] === undefined ? 1 : h[8]]];

  /* ── 3D rotations ──────────────────────────────────────────────────────── */
  function Rx(t) { const c = Math.cos(t), s = Math.sin(t); return [[1, 0, 0], [0, c, -s], [0, s, c]]; }
  function Ry(t) { const c = Math.cos(t), s = Math.sin(t); return [[c, 0, s], [0, 1, 0], [-s, 0, c]]; }
  function Rz(t) { const c = Math.cos(t), s = Math.sin(t); return [[c, -s, 0], [s, c, 0], [0, 0, 1]]; }
  const skew = n => [[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]];

  /* Rodrigues: R = I + sin θ [n̂]× + (1 − cos θ) [n̂]×²  — verified against the
     quaternion form to ~1e-16 in the page's own numerical check. */
  function rodrigues(axis, th) {
    const n = unit(axis), Kx = skew(n), K2 = mul(Kx, Kx);
    const s = Math.sin(th), c = 1 - Math.cos(th), I = eye(3);
    return I.map((row, i) => row.map((v, j) => v + s * Kx[i][j] + c * K2[i][j]));
  }
  /* the ZYX ("yaw–pitch–roll") convention used throughout the series:
     R = Rz(α) Ry(β) Rx(γ), applied right-to-left, so roll happens first. */
  const eulerZYX = (a, b, g) => mul(mul(Rz(a), Ry(b)), Rx(g));
  function eulerFromZYX(R) {                 // returns {a, b, g, locked}
    const sy = -R[2][0], cb = Math.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0]);
    if (cb < 1e-8) {
      /* Gimbal lock. β = ±90°, and only the COMBINATION α − γ (at β = +90°) or
         α + γ (at β = −90°) is determined; the split between them is free. We report
         the whole of it in α and set γ = 0. Both branches reduce to the same formula:
         at β = +90° the matrix is [[0, sin(γ−α), cos(γ−α)], [0, cos(γ−α), −sin(γ−α)],
         [−1, 0, 0]], and at β = −90° it is the mirror of that, so in each case
         −atan2(R₀₁, R₁₁) is exactly the surviving combination.                      */
      return { a: -Math.atan2(R[0][1], R[1][1]), b: Math.asin(clamp(sy, -1, 1)), g: 0, locked: true };
    }
    return {
      a: Math.atan2(R[1][0], R[0][0]),
      b: Math.atan2(sy, cb),
      g: Math.atan2(R[2][1], R[2][2]),
      locked: false
    };
  }
  function axisAngleFromR(R) {
    const tr = R[0][0] + R[1][1] + R[2][2];
    const th = Math.acos(clamp((tr - 1) / 2, -1, 1));
    if (th < 1e-9) return { axis: [0, 0, 1], angle: 0 };
    if (Math.PI - th < 1e-6) {               // θ ≈ π: use the symmetric part, ±axis both valid
      const d = [Math.sqrt(Math.max(0, (R[0][0] + 1) / 2)), Math.sqrt(Math.max(0, (R[1][1] + 1) / 2)), Math.sqrt(Math.max(0, (R[2][2] + 1) / 2))];
      if (R[0][1] < 0) d[1] = -d[1];
      if (R[0][2] < 0) d[2] = -d[2];
      return { axis: unit(d), angle: Math.PI };
    }
    const k = 1 / (2 * Math.sin(th));
    return { axis: [(R[2][1] - R[1][2]) * k, (R[0][2] - R[2][0]) * k, (R[1][0] - R[0][1]) * k], angle: th };
  }

  /* quaternions, stored (x, y, z, w) with w LAST and ‖q‖ = 1 */
  function qFromAxisAngle(axis, th) {
    const n = unit(axis), s = Math.sin(th / 2);
    return [n[0] * s, n[1] * s, n[2] * s, Math.cos(th / 2)];
  }
  function qToR(q) {
    const [x, y, z, w] = qNorm(q);
    return [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ];
  }
  function qMul(a, b) {                      // Hamilton product; R(a·b) = R(a)R(b)
    const va = [a[0], a[1], a[2]], vb = [b[0], b[1], b[2]], wa = a[3], wb = b[3];
    const v = add(add(cross(va, vb), scale(vb, wa)), scale(va, wb));
    return [v[0], v[1], v[2], wa * wb - dot(va, vb)];
  }
  const qConj = q => [-q[0], -q[1], -q[2], q[3]];
  function qNorm(q) { const n = Math.hypot(q[0], q[1], q[2], q[3]); return n < 1e-15 ? [0, 0, 0, 1] : q.map(v => v / n); }
  function qFromR(R) {                       // Shepperd's branch-safe form
    const t = R[0][0] + R[1][1] + R[2][2];
    let q;
    if (t > 0) {
      const s = Math.sqrt(t + 1) * 2;
      q = [(R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s, 0.25 * s];
    } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
      const s = Math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]) * 2;
      q = [0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s, (R[2][1] - R[1][2]) / s];
    } else if (R[1][1] > R[2][2]) {
      const s = Math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]) * 2;
      q = [(R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s, (R[0][2] - R[2][0]) / s];
    } else {
      const s = Math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]) * 2;
      q = [(R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s, (R[1][0] - R[0][1]) / s];
    }
    return qNorm(q);
  }
  /* Spherical linear interpolation on the unit 3-sphere. The antipodal flip is what
     makes slerp take the SHORT way round — without it the double cover shows up as
     a 360° detour, which the page's figure lets the reader switch off and watch. */
  function slerp(q0, q1, a, allowLong) {
    let A = qNorm(q0), B = qNorm(q1);
    let d = A[0] * B[0] + A[1] * B[1] + A[2] * B[2] + A[3] * B[3];
    if (d < 0 && !allowLong) { B = B.map(v => -v); d = -d; }
    d = clamp(d, -1, 1);
    const th = Math.acos(d);
    if (th < 1e-6) return qNorm(A.map((v, i) => v + a * (B[i] - v)));
    const s = Math.sin(th);
    const c0 = Math.sin((1 - a) * th) / s, c1 = Math.sin(a * th) / s;
    return qNorm(A.map((v, i) => c0 * v + c1 * B[i]));
  }
  /* nearest rotation matrix, by one step of Björck–Bowie iteration — enough to stop
     a matrix that has been nudged by a slider from drifting off SO(3) visibly */
  function orthonormalise(R) {
    const M = mul(T(R), R);
    const N = M.map((r, i) => r.map((v, j) => 1.5 * (i === j ? 1 : 0) - 0.5 * v));
    return mul(R, N);
  }

  /* ── cameras ───────────────────────────────────────────────────────────── */
  function K(o) {
    const f = (o.f === undefined) ? 500 : o.f;
    const fx = (o.fx === undefined) ? f : o.fx;
    const fy = (o.fy === undefined) ? (o.a === undefined ? f : o.a * f) : o.fy;
    return [[fx, o.s || 0, o.cx || 0], [0, fy, o.cy || 0], [0, 0, 1]];
  }
  function camera(o) {
    const Km = o.K, R = o.R || eye(3), t = o.t || [0, 0, 0];
    const Rt = R.map((row, i) => row.concat([t[i]]));
    const P = mul(Km, Rt);
    const Ri = T(R);                                   // R orthonormal ⇒ R⁻¹ = Rᵀ
    const C = neg(mv(Ri, t));                          // camera centre in world coords
    return {
      K: Km, R: R, t: t, P: P, C: C,
      project: function (p) {
        const pc = add(mv(R, p), t);
        const x = mv(Km, pc);
        return [x[0] / x[2], x[1] / x[2], pc[2]];      // u, v, and the depth that made it
      },
      camPoint: function (p) { return add(mv(R, p), t); }
    };
  }
  function lookAt(eye_, target, up) {
    const fwd = unit(sub(target, eye_));
    const u = up || [0, 0, 1];
    let right = cross(fwd, u);
    if (norm(right) < 1e-9) right = cross(fwd, [0, 1, 0]);
    right = unit(right);
    const down = cross(fwd, right);
    return [right, down, fwd];                         // rows: x right, y down, z forward
  }
  const fovFromF = (f, W) => 2 * deg(Math.atan(W / (2 * f)));
  const fFromFov = (fovDeg, W) => (W / 2) / Math.tan(rad(fovDeg) / 2);

  /* Brown–Conrady: normalised ideal (x, y) → normalised observed (x̂, ŷ). */
  function distort(x, y, k) {
    const k1 = k.k1 || 0, k2 = k.k2 || 0, k3 = k.k3 || 0, p1 = k.p1 || 0, p2 = k.p2 || 0;
    const r2 = x * x + y * y, rad_ = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    return [x * rad_ + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
    y * rad_ + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y];
  }
  /* The inverse has no closed form, so it is a fixed point: x ← (x̂ − tangential)/radial.
     It converges quickly for mild distortion and DIVERGES once the radial map stops
     being monotone, which happens at r > 1/√(−3κ₁) for κ₁ < 0 — see the page. */
  function undistort(xd, yd, k, iters) {
    const n = iters || 20;
    let x = xd, y = yd;
    for (let i = 0; i < n; i++) {
      const r2 = x * x + y * y;
      const rad_ = 1 + (k.k1 || 0) * r2 + (k.k2 || 0) * r2 * r2 + (k.k3 || 0) * r2 * r2 * r2;
      const dx = 2 * (k.p1 || 0) * x * y + (k.p2 || 0) * (r2 + 2 * x * x);
      const dy = (k.p1 || 0) * (r2 + 2 * y * y) + 2 * (k.p2 || 0) * x * y;
      if (!isFinite(rad_) || Math.abs(rad_) < 1e-9) break;
      x = (xd - dx) / rad_; y = (yd - dy) / rad_;
    }
    return [x, y];
  }

  /* ── a small perspective viewer for the 3D panels ──────────────────────── */
  /* World convention used across the series: X right, Y forward (into the scene),
     Z up. The preview camera orbits the target by (yaw about Z, pitch above the
     ground plane) at distance dist, and projects with the very pinhole model the
     page derives — the picture and the theory use one piece of code.            */
  function view3(o) {
    const t = o.target || [0, 0, 0];
    const yaw = (o.yaw === undefined) ? rad(-60) : o.yaw;
    const pitch = (o.pitch === undefined) ? rad(22) : o.pitch;
    const dist = (o.dist === undefined) ? 14 : o.dist;
    const f = (o.f === undefined) ? 460 : o.f;
    const cx = (o.cx === undefined) ? 0 : o.cx, cy = (o.cy === undefined) ? 0 : o.cy;
    const eye_ = [t[0] + dist * Math.cos(pitch) * Math.cos(yaw),
    t[1] + dist * Math.cos(pitch) * Math.sin(yaw),
    t[2] + dist * Math.sin(pitch)];
    const R = lookAt(eye_, t, [0, 0, 1]);
    return {
      eye: eye_, R: R, f: f, opts: o,
      project: function (p) {
        const d = sub(p, eye_);
        const pc = mv(R, d);
        const z = Math.max(pc[2], 1e-4);
        return { x: cx + f * pc[0] / z, y: cy + f * pc[1] / z, z: pc[2] };
      }
    };
  }
  const painter = (faces, view) => faces.slice().sort((a, b) => {
    const da = a.verts.reduce((s, v) => s + view.project(v).z, 0) / a.verts.length;
    const db = b.verts.reduce((s, v) => s + view.project(v).z, 0) / b.verts.length;
    return db - da;                                    // far first
  });
  function gridPlane(view, n, step, z) {
    const out = [], h = n * step / 2, zz = z || 0;
    for (let i = 0; i <= n; i++) {
      const u = -h + i * step;
      out.push([view.project([u, -h, zz]), view.project([u, h, zz])]);
      out.push([view.project([-h, u, zz]), view.project([h, u, zz])]);
    }
    return out;
  }
  function cubeMesh(cx, cy, cz, s) {
    const h = s / 2, V = [];
    for (const dz of [-h, h]) for (const dy of [-h, h]) for (const dx of [-h, h]) V.push([cx + dx, cy + dy, cz + dz]);
    /* index order: bit0 = x, bit1 = y, bit2 = z */
    const E = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    const F = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]];
    return { verts: V, edges: E, faces: F };
  }
  function axes3(g, view, len, opt) {
    const o = Object.assign({ o: [0, 0, 0], w: 1.6, labels: ["X", "Y", "Z"], colors: [VC.bad, VC.good, VC.accent] }, opt || {});
    const L = len || 2, p0 = view.project(o.o);
    [[L, 0, 0], [0, L, 0], [0, 0, L]].forEach((d, i) => {
      const p1 = view.project(add(o.o, d));
      g.append("line").attr("x1", p0.x).attr("y1", p0.y).attr("x2", p1.x).attr("y2", p1.y)
        .attr("stroke", o.colors[i]).attr("stroke-width", o.w);
      if (o.labels) g.append("text").attr("x", p1.x + 4).attr("y", p1.y - 3)
        .attr("font-size", 10).attr("fill", o.colors[i]).text(o.labels[i]);
    });
    return g;
  }

  /* ── drawing ───────────────────────────────────────────────────────────── */
  function frame(sel, W, H, m) {
    const mm = Object.assign({ l: 46, r: 16, t: 14, b: 34 }, m || {});
    const svg = (typeof sel === "string") ? d3.select(sel) : sel;
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${mm.l},${mm.t})`);
    return { svg: svg, g: g, m: mm, iw: W - mm.l - mm.r, ih: H - mm.t - mm.b };
  }
  function axisB(g, x, ih, ticks, label, fmtFn) {
    const ax = g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`)
      .call(fmtFn ? d3.axisBottom(x).ticks(ticks || 6).tickFormat(fmtFn) : d3.axisBottom(x).ticks(ticks || 6));
    if (label) g.append("text").attr("x", x.range()[1]).attr("y", ih + 31).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", VC.muted).text(label);
    return ax;
  }
  function axisL(g, y, ticks, label, fmtFn) {
    const ax = g.append("g").attr("class", "axis")
      .call(fmtFn ? d3.axisLeft(y).ticks(ticks || 5).tickFormat(fmtFn) : d3.axisLeft(y).ticks(ticks || 5));
    if (label) g.append("text").attr("x", 0).attr("y", -6).attr("text-anchor", "start")
      .attr("font-size", 11).attr("fill", VC.muted).text(label);
    return ax;
  }
  function gridY(g, y, iw, ticks) {
    g.append("g").attr("class", "gridlines").selectAll("line").data(y.ticks(ticks || 5)).join("line")
      .attr("x1", 0).attr("x2", iw).attr("y1", d => y(d)).attr("y2", d => y(d)).attr("stroke", VC.grid);
  }
  function gridX(g, x, ih, ticks) {
    g.append("g").attr("class", "gridlines").selectAll("line").data(x.ticks(ticks || 5)).join("line")
      .attr("y1", 0).attr("y2", ih).attr("x1", d => x(d)).attr("x2", d => x(d)).attr("stroke", VC.grid);
  }
  function legend(g, items, x, y, opts) {
    const o = Object.assign({ gap: 15, size: 9, font: 10.5, vertical: true, step: 96 }, opts || {});
    const gl = g.append("g").attr("transform", `translate(${x},${y})`);
    items.forEach((it, i) => {
      const dx = o.vertical ? 0 : i * o.step, dy = o.vertical ? i * o.gap : 0;
      if (it.dash) {
        gl.append("line").attr("x1", dx).attr("x2", dx + o.size).attr("y1", dy).attr("y2", dy)
          .attr("stroke", it.color).attr("stroke-width", 2).attr("stroke-dasharray", it.dash);
      } else {
        gl.append("rect").attr("x", dx).attr("y", dy - o.size / 2).attr("width", o.size)
          .attr("height", o.size).attr("rx", 2).attr("fill", it.color)
          .attr("fill-opacity", it.op === undefined ? 0.9 : it.op);
      }
      gl.append("text").attr("x", dx + o.size + 6).attr("y", dy + 3.5)
        .attr("font-size", o.font).attr("fill", VC.muted).text(it.label);
    });
    return gl;
  }
  function panel(g, x, y, w, h, title) {
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    if (title) gg.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11.5)
      .attr("fill", VC.ink).attr("font-weight", 600).text(title);
    return { g: gg, w: w, h: h };
  }
  /* A rectangular clip region. NOTE the coordinate space: the rect is expressed in the
     USER SPACE OF THE ELEMENT THAT REFERENCES THE CLIP, not in page coordinates. Inside
     a <g> that has already been translated, the rect therefore starts at (0, 0), not at
     the translation. Getting this wrong clips the whole drawing away silently. */
  function clip(svg, id, x, y, w, h) {
    svg.append("defs").append("clipPath").attr("id", id)
      .append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h);
    return "url(#" + id + ")";
  }
  function poly(g, pts, opt) {
    const o = Object.assign({ fill: "none", stroke: VC.accent, w: 1.6, op: 1, fillOp: 0.18, dash: null, close: true }, opt || {});
    const d = pts.map((p, i) => (i ? "L" : "M") + p[0] + "," + p[1]).join(" ") + (o.close ? " Z" : "");
    const el = g.append("path").attr("d", d).attr("fill", o.fill).attr("fill-opacity", o.fill === "none" ? 0 : o.fillOp)
      .attr("stroke", o.stroke).attr("stroke-width", o.w).attr("stroke-opacity", o.op);
    if (o.dash) el.attr("stroke-dasharray", o.dash);
    return el;
  }
  function arrow(g, x1, y1, x2, y2, opt) {
    const o = Object.assign({ color: VC.a2, w: 1.6, head: 6, dash: null, op: 1 }, opt || {});
    const el = g.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
      .attr("stroke", o.color).attr("stroke-width", o.w).attr("stroke-opacity", o.op);
    if (o.dash) el.attr("stroke-dasharray", o.dash);
    const a = Math.atan2(y2 - y1, x2 - x1), h = o.head;
    g.append("path").attr("d", `M${x2},${y2} L${x2 - h * Math.cos(a - 0.4)},${y2 - h * Math.sin(a - 0.4)} L${x2 - h * Math.cos(a + 0.4)},${y2 - h * Math.sin(a + 0.4)} Z`)
      .attr("fill", o.color).attr("fill-opacity", o.op);
    return el;
  }

  return {
    clamp, lerp, deg, rad, linspace, fmt, sig,
    rng, randn, poisson,
    add, sub, neg, scale, dot, norm, unit, cross,
    mat, eye, T, mul, mv, det3, inv3, invN, inv4,
    hom, dehom, applyH, applyH4, lineThrough, meet, normLine,
    T2, R2, S2, A2, H2,
    Rx, Ry, Rz, skew, rodrigues, eulerZYX, eulerFromZYX, axisAngleFromR,
    qFromAxisAngle, qToR, qMul, qConj, qNorm, qFromR, slerp, orthonormalise,
    K, camera, lookAt, fovFromF, fFromFov, distort, undistort,
    view3, painter, gridPlane, cubeMesh, axes3,
    frame, axisB, axisL, gridX, gridY, legend, panel, clip, poly, arrow
  };
})();
