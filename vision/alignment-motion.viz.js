/* alignment-motion.viz.js — figures for vision/alignment-motion.html (part 6 of 7).
   Loaded after ../data.js → ../notes.js → vision-viz.js.

   Everything page-local lives in the AM namespace. The shared toolbox (VZ) supplies the
   arithmetic — matrices, homogeneous coordinates, the 2D groups, Jacobi eigen, Gaussian
   kernels, separable filtering, and the drawing helpers — and is NOT duplicated here.
   AM adds only what this page needs and vision-viz.js does not have:

     · AM.lstsq(A, b)          overdetermined linear least squares, normal equations
     · AM.nullvec(A)           the smallest right singular vector, via Jacobi on AᵀA
     · AM.condA(A)             σ₁/σ₈ of a DLT design matrix — the null-space sensitivity
     · AM.hnorm(P)             Hartley's isotropic normalising similarity
     · AM.dlt(src, dst, norm)  the direct linear transform, with or without normalisation
     · AM.fit2d(kind, s, d)    translation / Euclidean / similarity / affine, closed form
     · AM.tensor, AM.lkStep       the structure tensor and one LK step, on the
                                 analytic scenes of §21–§23 (AM.sceneI / AM.sceneG)
     · AM.bilinear, AM.grads, AM.pyrDown, AM.pyramid, AM.lkTrans, AM.coarseToFine,
       AM.hornSchunck, AM.blockMatch1D, AM.flowColor   the raster dense-motion kit
     · AM.h4pt, AM.degenerate4, AM.refineH, AM.refinePose, AM.transferErr
     · AM.drag(...)            a uniform draggable-handle helper

   Verified in python (numpy/scipy) before being written: see the page prose for the
   numbers each figure must reproduce.                                                  */

const AM = (function () {

  /* ── linear algebra this page needs on top of VZ ─────────────────────────── */

  /* Overdetermined least squares by normal equations. n is tiny here (≤ 8), and the
     design matrices are well scaled, so squaring the condition number is affordable;
     §07's DLT is the one place where it would not be, and that uses nullvec instead. */
  function lstsq(A, b) {
    const n = A[0].length, AT = VZ.T(A);
    const N = VZ.mul(AT, A), r = VZ.mv(AT, b);
    const Ni = VZ.invN(N);
    if (!Ni) return new Array(n).fill(0);
    return VZ.mv(Ni, r);
  }

  /* The right singular vector of the SMALLEST singular value = the eigenvector of AᵀA
     with the smallest eigenvalue. VZ.jacobiEig returns eigenvalues ASCENDING, so it is
     index 0. This is the homogeneous least-squares solution: min ‖Ah‖ s.t. ‖h‖ = 1. */
  function nullvec(A) {
    const M = VZ.mul(VZ.T(A), A);
    const e = VZ.jacobiEig(M, 90);
    return e.vectors[0].slice();
  }

  /* Singular values of A, as √(eigenvalues of AᵀA), descending. */
  function svals(A) {
    const e = VZ.jacobiEig(VZ.mul(VZ.T(A), A), 90);
    return e.values.map(v => Math.sqrt(Math.max(0, v))).reverse();
  }

  /* The conditioning number that matters for a null-space problem is σ₁/σ₈: how big the
     data is against how weakly the eighth direction is constrained. σ₉ is the answer
     itself and is meant to be ~0, so σ₁/σ₉ would measure nothing. */
  function condA(A) {
    const s = svals(A);
    const lo = s[7];
    return (lo > 1e-300) ? s[0] / lo : Infinity;
  }

  /* Hartley's normalising similarity: translate the centroid to the origin and scale so
     that the mean distance from it is √2 — i.e. the average point sits at (1, 1). */
  function hnorm(P) {
    let cx = 0, cy = 0;
    for (const p of P) { cx += p[0]; cy += p[1]; }
    cx /= P.length; cy /= P.length;
    let d = 0;
    for (const p of P) d += Math.hypot(p[0] - cx, p[1] - cy);
    d /= P.length;
    const s = (d > 1e-12) ? Math.SQRT2 / d : 1;
    return [[s, 0, -s * cx], [0, s, -s * cy], [0, 0, 1]];
  }

  /* The DLT design matrix: two rows per correspondence, from x̃ × (H x) = 0. */
  function dltRows(src, dst) {
    const A = [];
    for (let i = 0; i < src.length; i++) {
      const x = src[i][0], y = src[i][1], u = dst[i][0], v = dst[i][1];
      A.push([-x, -y, -1, 0, 0, 0, u * x, u * y, u]);
      A.push([0, 0, 0, -x, -y, -1, v * x, v * y, v]);
    }
    return A;
  }

  /* The direct linear transform. normalise = true applies Hartley conditioning first and
     undoes it afterwards; the returned cond is the conditioning of the system ACTUALLY
     solved, which is the whole point of the comparison in §07. */
  function dlt(src, dst, normalise) {
    if (src.length < 4) return null;
    let S = src, D = dst, T1 = null, T2 = null;
    if (normalise) {
      T1 = hnorm(src); T2 = hnorm(dst);
      S = src.map(p => VZ.applyH(T1, p));
      D = dst.map(p => VZ.applyH(T2, p));
    }
    const A = dltRows(S, D);
    const h = nullvec(A);
    let H = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]];
    if (normalise) {
      const T2i = VZ.inv3(T2);
      if (!T2i) return null;
      H = VZ.mul(VZ.mul(T2i, H), T1);
    }
    const w = H[2][2];
    if (Math.abs(w) > 1e-14) H = H.map(r => r.map(v => v / w));
    return { H: H, cond: condA(A), sv: svals(A) };
  }

  /* ── the linear 2D models, each as a closed form ─────────────────────────── */
  /* kind ∈ "trans" | "eucl" | "sim" | "aff". Returns a 3×3 so everything downstream
     (drawing, composing, warping) can treat all four identically. */
  function fit2d(kind, src, dst) {
    const N = src.length;
    if (kind === "trans") {
      let tx = 0, ty = 0;
      for (let i = 0; i < N; i++) { tx += dst[i][0] - src[i][0]; ty += dst[i][1] - src[i][1]; }
      return VZ.T2(tx / N, ty / N);
    }
    if (kind === "aff") {
      const A = [], b = [];
      for (let i = 0; i < N; i++) {
        A.push([src[i][0], src[i][1], 1, 0, 0, 0]); b.push(dst[i][0]);
        A.push([0, 0, 0, src[i][0], src[i][1], 1]); b.push(dst[i][1]);
      }
      const p = lstsq(A, b);
      return [[p[0], p[1], p[2]], [p[3], p[4], p[5]], [0, 0, 1]];
    }
    /* Euclidean and similarity share the centroid-and-cross-product closed form; the
       only difference is whether the recovered scale is used or forced to 1. */
    let cx = 0, cy = 0, dx = 0, dy = 0;
    for (let i = 0; i < N; i++) { cx += src[i][0]; cy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
    cx /= N; cy /= N; dx /= N; dy /= N;
    let sd = 0, sc = 0, ss = 0;
    for (let i = 0; i < N; i++) {
      const ax = src[i][0] - cx, ay = src[i][1] - cy, bx = dst[i][0] - dx, by = dst[i][1] - dy;
      sd += ax * bx + ay * by;          // Σ dot
      sc += ax * by - ay * bx;          // Σ cross
      ss += ax * ax + ay * ay;
    }
    const th = Math.atan2(sc, sd);
    const s = (kind === "sim" && ss > 1e-12) ? Math.hypot(sd, sc) / ss : 1;
    const c = s * Math.cos(th), q = s * Math.sin(th);
    return [[c, -q, dx - (c * cx - q * cy)], [q, c, dy - (q * cx + c * cy)], [0, 0, 1]];
  }

  function rms(H, src, dst) {
    let s = 0;
    for (let i = 0; i < src.length; i++) {
      const p = VZ.applyH(H, src[i]);
      if (!p) return Infinity;
      s += (p[0] - dst[i][0]) ** 2 + (p[1] - dst[i][1]) ** 2;
    }
    return Math.sqrt(s / src.length);
  }

  /* decompose a 3×3 that is known to be a similarity */
  function decompSim(H) {
    return { s: Math.hypot(H[0][0], H[1][0]), th: VZ.deg(Math.atan2(H[1][0], H[0][0])), t: [H[0][2], H[1][2]] };
  }

  /* ── a uniform draggable handle ──────────────────────────────────────────── */
  /* pts is an array of [x, y] in WORLD units; to/from convert world ↔ screen. */
  function drag(g, pts, to, from, onchange, opt) {
    const o = Object.assign({ r: 6, fill: VC.a2, stroke: VC.bg, klass: "dragpt" }, opt || {});
    const sel = g.selectAll("circle." + o.klass).data(pts.map((p, i) => i)).join("circle")
      .attr("class", o.klass)
      .attr("cx", i => to(pts[i])[0]).attr("cy", i => to(pts[i])[1])
      .attr("r", o.r).attr("fill", o.fill).attr("fill-opacity", 0.85)
      .attr("stroke", o.stroke).attr("stroke-width", 1.5).style("cursor", "grab");
    sel.call(d3.drag()
      .on("drag", function (ev, i) {
        const w = from([ev.x, ev.y]);
        pts[i][0] = w[0]; pts[i][1] = w[1];
        onchange(i);
      }));
    return sel;
  }

  return {
    lstsq: lstsq, nullvec: nullvec, svals: svals, condA: condA,
    hnorm: hnorm, dltRows: dltRows, dlt: dlt,
    fit2d: fit2d, rms: rms, decompSim: decompSim, drag: drag
  };
})();

/* ═══ FIG 1 · §04 — the model ladder ═════════════════════════════════════════
   Six correspondences generated by a known similarity (s = 1.20, θ = 15°,
   t = (30, −12)), then perturbed. All four linear models are fitted to the SAME
   data on every redraw and their RMS residuals compared, so under- and
   over-parameterisation are visible side by side rather than argued about.     */
(function () {
  const svg = d3.select("#lsq-svg");
  if (svg.empty()) return;
  const W = 760, H = 440;

  const SRC = [[60, 80], [220, 70], [300, 190], [140, 260], [70, 200], [250, 300]];
  const TRUE = { s: 1.20, th: VZ.rad(15), t: [30, -12] };
  const HT = VZ.S2(TRUE.s, TRUE.th, TRUE.t[0], TRUE.t[1]);

  /* one fixed table of unit normals, so the noise slider scales a FIXED pattern
     rather than re-rolling the dice on every drag */
  const r0 = VZ.rng(20260906);
  const NZ = SRC.map(() => [VZ.randn(r0), VZ.randn(r0)]);
  const UOFF = SRC.map(() => [0, 0]);          // whatever the reader dragged

  const kA = 0.60;
  const A2S = p => [14 + (p[0] - 20) * kA, 46 + (p[1] - 40) * kA];
  const B2S = p => [268 + (p[0] - 40) * kA, 46 + (p[1] - 30) * kA];
  const S2B = q => [(q[0] - 268) / kA + 40, (q[1] - 46) / kA + 30];

  const MODELS = [["trans", "translation", 2], ["eucl", "Euclidean", 3], ["sim", "similarity", 4], ["aff", "affine", 6]];

  const ui = {
    m: document.getElementById("ls-m"), n: document.getElementById("ls-n"),
    nv: document.getElementById("ls-nv"), o: document.getElementById("ls-o"),
    ov: document.getElementById("ls-ov"), all: document.getElementById("ls-all"),
    out: document.getElementById("lsq-readout")
  };

  function measured(sig, out) {
    return SRC.map((p, i) => {
      const t = VZ.applyH(HT, p);
      let x = t[0] + sig * NZ[i][0] + UOFF[i][0];
      let y = t[1] + sig * NZ[i][1] + UOFF[i][1];
      if (i === 0) { x += out * 0.78; y -= out * 0.63; }     // a definite, directed blunder
      return [x, y];
    });
  }

  function draw() {
    const kind = ui.m.value, sig = +ui.n.value, out = +ui.o.value, showAll = ui.all.checked;
    ui.nv.textContent = VZ.fmt(sig, 2) + " px";
    ui.ov.textContent = out + " px";

    const M = measured(sig, out);
    const fits = {}, err = {};
    MODELS.forEach(m => { fits[m[0]] = AM.fit2d(m[0], SRC, M); err[m[0]] = AM.rms(fits[m[0]], SRC, M); });
    const Hf = fits[kind];

    const f = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = f.g;

    /* ---- panel A: image 1 -------------------------------------------------- */
    VZ.panelBox(g, 14, 34, 236, 300, "image 1 · the source points");
    const gA = g.append("g");
    SRC.forEach((p, i) => {
      const q = A2S(p);
      gA.append("circle").attr("cx", q[0]).attr("cy", q[1]).attr("r", 5)
        .attr("fill", VC.accent).attr("fill-opacity", 0.9).attr("stroke", VC.bg);
      gA.append("text").attr("x", q[0] + 8).attr("y", q[1] + 4).attr("font-size", 9.5)
        .attr("fill", VC.muted).text(i + 1);
    });
    g.append("text").attr("x", 14).attr("y", 352).attr("font-size", 10.5).attr("fill", VC.muted)
      .text("fixed. the six points of the worked example in §04.");

    /* ---- panel B: image 2 -------------------------------------------------- */
    VZ.panelBox(g, 268, 34, 272, 300, "image 2 · measured ○ vs predicted ■   (residuals ×5)");
    const gB = g.append("g");
    M.forEach((m, i) => {
      const pr = VZ.applyH(Hf, SRC[i]);
      const qm = B2S(m), qp = B2S(pr);
      const bad = (i === 0 && out > 0);
      /* residual, magnified five times about the measured point so sub-pixel
         differences are visible at this scale */
      gB.append("line").attr("x1", qm[0]).attr("y1", qm[1])
        .attr("x2", qm[0] + 5 * (qp[0] - qm[0])).attr("y2", qm[1] + 5 * (qp[1] - qm[1]))
        .attr("stroke", bad ? VC.bad : VC.a2).attr("stroke-width", 1.6).attr("stroke-opacity", 0.95);
      gB.append("rect").attr("x", qp[0] - 3.5).attr("y", qp[1] - 3.5).attr("width", 7).attr("height", 7)
        .attr("fill", VC.good).attr("fill-opacity", 0.9);
    });
    AM.drag(gB, UOFF.map((u, i) => M[i]), B2S, S2B, function (i) {
      const t = VZ.applyH(HT, SRC[i]);
      const base = [t[0] + sig * NZ[i][0] + (i === 0 ? out * 0.78 : 0),
                    t[1] + sig * NZ[i][1] - (i === 0 ? out * 0.63 : 0)];
      const cur = M[i];
      UOFF[i][0] = cur[0] - base[0]; UOFF[i][1] = cur[1] - base[1];
      draw();
    }, { r: 6, fill: "none", stroke: VC.a2 });
    gB.selectAll("circle.dragpt").attr("stroke-width", 2).attr("fill", VC.bg).attr("fill-opacity", 0.25);
    g.append("text").attr("x", 268).attr("y", 352).attr("font-size", 10.5).attr("fill", VC.muted)
      .text("drag any measured point — every fit is recomputed.");

    /* ---- panel C: RMS residual of all four models -------------------------- */
    VZ.panelBox(g, 566, 34, 182, 300, "RMS residual · log scale");
    const y = d3.scaleLog().domain([0.02, 400]).range([322, 56]).clamp(true);
    const gC = g.append("g");
    [0.05, 0.5, 5, 50, 400].forEach(v => {
      gC.append("line").attr("x1", 566).attr("x2", 748).attr("y1", y(v)).attr("y2", y(v))
        .attr("stroke", VC.grid);
      gC.append("text").attr("x", 570).attr("y", y(v) - 3).attr("font-size", 9).attr("fill", VC.muted)
        .text(v + " px");
    });
    MODELS.forEach((m, i) => {
      const e = Math.max(err[m[0]], 0.021), x = 580 + i * 42;
      const on = (m[0] === kind);
      if (!showAll && !on) return;
      gC.append("rect").attr("x", x).attr("y", y(e)).attr("width", 30).attr("height", Math.max(1, 322 - y(e)))
        .attr("fill", on ? VC.accent : VC.muted).attr("fill-opacity", on ? 0.9 : 0.32);
      gC.append("text").attr("x", x + 15).attr("y", y(e) - 5).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", on ? VC.ink : VC.muted).text(VZ.fmt(err[m[0]], 2));
      gC.append("text").attr("x", x + 15).attr("y", 336).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("fill", VC.muted).text(m[2] + " dof");
    });

    /* ---- readout ----------------------------------------------------------- */
    const d = AM.decompSim(Hf);
    const name = MODELS.find(m => m[0] === kind)[1];
    let s = `<b>${name}</b> · RMS residual <b>${VZ.fmt(err[kind], 3)} px</b>`;
    if (kind === "sim" || kind === "eucl") {
      s += ` · ŝ = ${VZ.fmt(d.s, 5)} (true 1.20000) · <span class="keep">θ</span>̂ = ${VZ.fmt(d.th, 4)}° (true 15.0000°)` +
           ` · t̂ = (${VZ.fmt(d.t[0], 2)}, ${VZ.fmt(d.t[1], 2)}) (true 30.00, −12.00)`;
    } else if (kind === "trans") {
      s += ` · t̂ = (${VZ.fmt(Hf[0][2], 2)}, ${VZ.fmt(Hf[1][2], 2)}) — the mean displacement, and the only thing 2 dof can say`;
    } else {
      s += ` · Â = [[${VZ.fmt(Hf[0][0], 4)}, ${VZ.fmt(Hf[0][1], 4)}], [${VZ.fmt(Hf[1][0], 4)}, ${VZ.fmt(Hf[1][1], 4)}]]` +
           ` · t̂ = (${VZ.fmt(Hf[0][2], 2)}, ${VZ.fmt(Hf[1][2], 2)})`;
    }
    s += `<br />all four: ` + MODELS.map(m => `${m[1]} ${VZ.fmt(err[m[0]], 3)}`).join(" · ") + " px";
    if (out > 0) s += ` — the ${out} px blunder on point 1 has moved <i>every</i> fit; least squares has no defence (§09, §10).`;
    ui.out.innerHTML = s;
  }

  [ui.m, ui.n, ui.o, ui.all].forEach(el => {
    el.addEventListener("input", draw); el.addEventListener("change", draw);
  });
  draw();
})();

/* ═══ FIG 2 · §05 — Procrustes, stage by stage ═══════════════════════════════
   Centre, rotate, scale. The determinant guard is shown by letting the reader
   MIRROR the target set: the unconstrained maximiser then becomes a reflection
   whose residual is lower than any rotation's, which is exactly the trap.      */
(function () {
  const svg = d3.select("#proc-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;

  /* a shape whose orientation is unmistakable: a house outline plus two interior marks */
  const BASE = [[-42, -30], [42, -30], [42, 8], [0, 44], [-42, 8], [-16, -6], [18, -8]];
  const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]];
  const TRUE = { s: 1.35, th: VZ.rad(40), t: [0, 0] };
  const cS = [112, 190], cT = [300, 190], O = [206, 190];

  const rn = VZ.rng(7717);
  const NZ = BASE.map(() => [VZ.randn(rn), VZ.randn(rn)]);
  const PULL = BASE.map(() => [0, 0]);          // reader drags, stored as offsets

  const ui = {
    st: document.getElementById("pr-stage"), sc: document.getElementById("pr-scale"),
    rf: document.getElementById("pr-refl"), n: document.getElementById("pr-n"),
    nv: document.getElementById("pr-nv"), out: document.getElementById("proc-readout")
  };

  const P2S = p => [14 + p[0], 34 + p[1]];
  const S2P = q => [q[0] - 14, q[1] - 34];

  function sets(sig, refl) {
    const src = BASE.map(b => [b[0] + cS[0], b[1] + cS[1]]);
    const c = Math.cos(TRUE.th) * TRUE.s, s = Math.sin(TRUE.th) * TRUE.s;
    const tgt = BASE.map((b, i) => {
      const bx = refl ? -b[0] : b[0], by = b[1];
      return [c * bx - s * by + cT[0] + sig * NZ[i][0] + PULL[i][0],
              s * bx + c * by + cT[1] + sig * NZ[i][1] + PULL[i][1]];
    });
    return { src: src, tgt: tgt };
  }

  function procrustes(src, tgt) {
    const N = src.length;
    let ax = 0, ay = 0, bx = 0, by = 0;
    src.forEach(p => { ax += p[0]; ay += p[1]; }); tgt.forEach(p => { bx += p[0]; by += p[1]; });
    ax /= N; ay /= N; bx /= N; by /= N;
    const y = src.map(p => [p[0] - ax, p[1] - ay]), z = tgt.map(p => [p[0] - bx, p[1] - by]);
    /* C = Σ zᵢ yᵢᵀ  (2 × 2) */
    let C = [[0, 0], [0, 0]], sy = 0, sz = 0;
    for (let i = 0; i < N; i++) {
      C[0][0] += z[i][0] * y[i][0]; C[0][1] += z[i][0] * y[i][1];
      C[1][0] += z[i][1] * y[i][0]; C[1][1] += z[i][1] * y[i][1];
      sy += y[i][0] * y[i][0] + y[i][1] * y[i][1];
      sz += z[i][0] * z[i][0] + z[i][1] * z[i][1];
    }
    const ev = VZ.jacobiEig(VZ.mul(VZ.T(C), C), 60);
    const sv = ev.values.map(v => Math.sqrt(Math.max(0, v))).reverse();   // σ₁ ≥ σ₂
    const detC = C[0][0] * C[1][1] - C[0][1] * C[1][0];
    const d = detC < 0 ? -1 : 1;
    /* the 2D closed form is already the SO(2) optimum */
    const th = Math.atan2(C[1][0] - C[0][1], C[0][0] + C[1][1]);
    const trR = sv[0] + d * sv[1], trO = sv[0] + sv[1];
    const s = sy > 1e-12 ? trR / sy : 1;
    return {
      cy: [ax, ay], cz: [bx, by], y: y, z: z, C: C, sv: sv, det: d, th: th, s: s,
      ssdR: sz - trR * trR / sy, ssdO: sz - trO * trO / sy, sy: sy, sz: sz
    };
  }

  function draw() {
    const stage = ui.st.value, useScale = ui.sc.checked, refl = ui.rf.checked, sig = +ui.n.value;
    ui.nv.textContent = VZ.fmt(sig, 1) + " px";
    const { src, tgt } = sets(sig, refl);
    const P = procrustes(src, tgt);
    const s = useScale ? P.s : 1, c = s * Math.cos(P.th), q = s * Math.sin(P.th);

    /* where each set is drawn at this stage */
    function place(i) {
      const yv = P.y[i], zv = P.z[i];
      if (stage === "raw") return [src[i], tgt[i]];
      if (stage === "centred") return [[yv[0] + O[0], yv[1] + O[1]], [zv[0] + O[0], zv[1] + O[1]]];
      const rot = [Math.cos(P.th) * yv[0] - Math.sin(P.th) * yv[1], Math.sin(P.th) * yv[0] + Math.cos(P.th) * yv[1]];
      const full = [c * yv[0] - q * yv[1], q * yv[0] + c * yv[1]];
      const a = (stage === "rot") ? rot : full;
      return [[a[0] + O[0], a[1] + O[1]], [zv[0] + O[0], zv[1] + O[1]]];
    }

    const f = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = f.g;
    VZ.panelBox(g, 14, 34, 420, 340, "source ● → target ○   ·   " +
      ({ raw: "raw", centred: "centred", rot: "centred + rotated", scaled: "centred + rotated + scaled" })[stage]);

    const A = [], B = [];
    for (let i = 0; i < BASE.length; i++) { const pr = place(i); A.push(pr[0]); B.push(pr[1]); }

    /* correspondence lines first, so the marks sit on top */
    for (let i = 0; i < A.length; i++) {
      const p = P2S(A[i]), qd = P2S(B[i]);
      g.append("line").attr("x1", p[0]).attr("y1", p[1]).attr("x2", qd[0]).attr("y2", qd[1])
        .attr("stroke", VC.muted).attr("stroke-width", 1).attr("stroke-opacity", 0.5).attr("stroke-dasharray", "2 3");
    }
    EDGES.forEach(e => {
      const p = P2S(A[e[0]]), qd = P2S(A[e[1]]);
      g.append("line").attr("x1", p[0]).attr("y1", p[1]).attr("x2", qd[0]).attr("y2", qd[1])
        .attr("stroke", VC.accent).attr("stroke-width", 1.8).attr("stroke-opacity", 0.8);
      const r = P2S(B[e[0]]), t = P2S(B[e[1]]);
      g.append("line").attr("x1", r[0]).attr("y1", r[1]).attr("x2", t[0]).attr("y2", t[1])
        .attr("stroke", VC.a2).attr("stroke-width", 1.8).attr("stroke-opacity", 0.8);
    });
    A.forEach(p => { const u = P2S(p); g.append("circle").attr("cx", u[0]).attr("cy", u[1]).attr("r", 4).attr("fill", VC.accent); });
    /* centroid crosses, only where they still mean something */
    if (stage === "raw") {
      [[P.cy, VC.accent], [P.cz, VC.a2]].forEach(([cc, col]) => {
        const u = P2S(cc);
        g.append("path").attr("d", `M${u[0] - 7},${u[1]} H${u[0] + 7} M${u[0]},${u[1] - 7} V${u[0] * 0 + u[1] + 7}`)
          .attr("stroke", col).attr("stroke-width", 1.6);
      });
    }
    /* the target points are the draggable ones */
    const gd = g.append("g");
    AM.drag(gd, B, P2S, S2P, function (i) {
      /* B[i] is a placed copy; convert the screen move back into a change of tgt[i] */
      const stageOff = (stage === "raw") ? [0, 0] : [O[0] - P.cz[0], O[1] - P.cz[1]];
      PULL[i][0] += (B[i][0] - stageOff[0]) - tgt[i][0];
      PULL[i][1] += (B[i][1] - stageOff[1]) - tgt[i][1];
      draw();
    }, { r: 5.5, fill: "none", stroke: VC.a2 });
    gd.selectAll("circle.dragpt").attr("stroke-width", 2).attr("fill", VC.bg).attr("fill-opacity", 0.2);

    /* ---- right column ------------------------------------------------------ */
    VZ.panelBox(g, 452, 34, 294, 340, "the cross-covariance and what it gives");
    const gr = g.append("g");
    VZ.matText(gr, P.C, 468, 78, { dp: 1, pad: 9, size: 10.5, label: "C = ∑ᵢ z̃ᵢ yᵢᵀ" });
    const kv = VZ.kv(gr, 468, 148, { keyW: 168, size: 11 });
    kv("σ₁, σ₂", VZ.sig(P.sv[0], 5) + ",  " + VZ.sig(P.sv[1], 4));
    kv("det(U Vᵀ)", P.det > 0 ? "+1  (a rotation)" : "−1  (would reflect)", P.det > 0 ? VC.good : VC.bad, true);
    kv("recovered angle", VZ.fmt(VZ.deg(P.th), 4) + "°", VC.ink);
    kv("recovered scale", VZ.fmt(P.s, 5) + (useScale ? "" : "  (forced to 1)"), VC.ink);
    kv("∑‖yᵢ‖²", VZ.sig(P.sy, 6));
    kv("SSD, best rotation", VZ.fmt(P.ssdR, 2), P.det > 0 ? VC.good : VC.ink, true);
    kv("SSD, best reflection", VZ.fmt(P.ssdO, 2), P.det < 0 ? VC.bad : VC.muted);
    gr.append("text").attr("x", 468).attr("y", 300).attr("font-size", 10.5).attr("fill", VC.muted)
      .attr("xml:space", "preserve").text(P.det > 0
        ? "det = +1: rotation and reflection agree; the guard is idle."
        : "det = −1: the reflection fits BETTER. Without the");
    if (P.det < 0) {
      gr.append("text").attr("x", 468).attr("y", 315).attr("font-size", 10.5).attr("fill", VC.bad)
        .attr("xml:space", "preserve").text("guard the solver would return it — and it is not");
      gr.append("text").attr("x", 468).attr("y", 330).attr("font-size", 10.5).attr("fill", VC.bad)
        .attr("xml:space", "preserve").text("a rigid motion. The guard costs one determinant.");
    }

    ui.out.innerHTML =
      `true generator: s = 1.35000, <span class="keep">θ</span> = 40.0000°` +
      ` · recovered: ŝ = <b>${VZ.fmt(P.s, 5)}</b>, <span class="keep">θ</span>̂ = <b>${VZ.fmt(VZ.deg(P.th), 4)}°</b>` +
      ` · <span class="keep">σ</span>₁ = ${VZ.sig(P.sv[0], 5)}, <span class="keep">σ</span>₂ = ${VZ.sig(P.sv[1], 4)}` +
      ` · det(UVᵀ) = ${P.det > 0 ? "+1" : "−1"}` +
      `<br />SSD over the best <b>rotation</b> = ${VZ.fmt(P.ssdR, 3)}; over the best <b>reflection</b> = ${VZ.fmt(P.ssdO, 3)}. ` +
      (P.det > 0
        ? "With unmirrored data these coincide, because the optimal orthogonal map already has determinant +1."
        : "With mirrored data the reflection wins by " + VZ.fmt(P.ssdR - P.ssdO, 1) + " — which is precisely why the constrained solution must be forced, not discovered.");
  }

  [ui.st, ui.sc, ui.rf, ui.n].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ═══ FIG 3 · §06 — four correspondences, one homography ═════════════════════
   The DLT is re-solved on every pixel of drag, the grid re-warped, and the
   design matrix's spectrum printed. Switching normalisation off moves the
   condition number by five decades and the drawn grid by nothing visible —
   the point being that conditioning fails long before the answer does.        */
(function () {
  const svg = d3.select("#dlt-svg");
  if (svg.empty()) return;
  const W = 760, H = 470, IW = 640, IH = 480, k = 0.52;

  const SRC = [[100, 100], [500, 120], [520, 400], [80, 380]];
  const HTRUE = [[0.90, 0.15, 40], [-0.10, 1.05, -25], [0.0002, 0.0003, 1]];
  const DST0 = SRC.map(p => VZ.applyH(HTRUE, p));
  let DST = DST0.map(p => p.slice());

  const rn = VZ.rng(4242);
  const EXS = d3.range(12).map(() => [40 + rn() * 560, 40 + rn() * 400]);
  const EXN = d3.range(12).map(() => [VZ.randn(rn), VZ.randn(rn)]);

  const A2S = p => [18 + p[0] * k, 44 + p[1] * k];
  const B2S = p => [390 + p[0] * k, 44 + p[1] * k];
  const S2B = q => [(q[0] - 390) / k, (q[1] - 44) / k];

  const ui = {
    n: document.getElementById("dl-n"), kk: document.getElementById("dl-k"),
    kv: document.getElementById("dl-kv"), s: document.getElementById("dl-s"),
    sv: document.getElementById("dl-sv"), grid: document.getElementById("dl-grid"),
    reset: document.getElementById("dl-reset"), out: document.getElementById("dlt-readout")
  };

  const pristine = () => DST.every((p, i) => Math.abs(p[0] - DST0[i][0]) < 1e-9 && Math.abs(p[1] - DST0[i][1]) < 1e-9);

  function draw() {
    const norm = ui.n.value === "hartley", nex = +ui.kk.value, sig = +ui.s.value;
    ui.kv.textContent = nex; ui.sv.textContent = VZ.fmt(sig, 2) + " px";

    /* the four handles alone define the map the extras are generated FROM */
    const base = AM.dlt(SRC, DST, true);
    const src = SRC.slice(), dst = DST.slice();
    for (let i = 0; i < nex; i++) {
      const p = EXS[i], q = base ? VZ.applyH(base.H, p) : p;
      if (!q) continue;
      src.push(p); dst.push([q[0] + sig * EXN[i][0], q[1] + sig * EXN[i][1]]);
    }
    const fit = AM.dlt(src, dst, norm);
    const Hf = fit ? fit.H : VZ.eye(3);

    const f = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = f.g;

    /* ---- panel A ----------------------------------------------------------- */
    VZ.panelBox(g, 18, 44, IW * k, IH * k, "image 1 · source");
    const gA = g.append("g");
    for (let x = 0; x <= IW; x += 80) {
      const a = A2S([x, 0]), b = A2S([x, IH]);
      gA.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", b[0]).attr("y2", b[1]).attr("stroke", VC.grid);
    }
    for (let y = 0; y <= IH; y += 80) {
      const a = A2S([0, y]), b = A2S([IW, y]);
      gA.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", b[0]).attr("y2", b[1]).attr("stroke", VC.grid);
    }
    VZ.poly(gA, SRC.map(A2S), { stroke: VC.accent, w: 1.6, fill: VC.accent, fillOp: 0.10 });
    SRC.forEach((p, i) => {
      const q = A2S(p);
      gA.append("circle").attr("cx", q[0]).attr("cy", q[1]).attr("r", 4.5).attr("fill", VC.accent);
      gA.append("text").attr("x", q[0] + 7).attr("y", q[1] - 5).attr("font-size", 9.5).attr("fill", VC.muted).text(i + 1);
    });
    for (let i = 0; i < nex; i++) {
      const q = A2S(EXS[i]);
      gA.append("circle").attr("cx", q[0]).attr("cy", q[1]).attr("r", 2.4).attr("fill", VC.teal);
    }

    /* ---- panel B ----------------------------------------------------------- */
    VZ.panelBox(g, 390, 44, IW * k, IH * k, "image 2 · target — drag the four handles");
    const gB = g.append("g");
    const cid = VZ.clip(svg, "am-dlt-clip", 390, 44, IW * k, IH * k);
    const gW = gB.append("g").attr("clip-path", cid);
    if (ui.grid.checked) {
      const seg = 16;
      for (let x = 0; x <= IW; x += 80) {
        const pts = [];
        for (let j = 0; j <= seg; j++) { const q = VZ.applyH(Hf, [x, IH * j / seg]); if (q) pts.push(B2S(q)); }
        if (pts.length > 1) VZ.poly(gW, pts, { stroke: VC.grid, w: 1, close: false });
      }
      for (let y = 0; y <= IH; y += 80) {
        const pts = [];
        for (let j = 0; j <= seg; j++) { const q = VZ.applyH(Hf, [IW * j / seg, y]); if (q) pts.push(B2S(q)); }
        if (pts.length > 1) VZ.poly(gW, pts, { stroke: VC.grid, w: 1, close: false });
      }
      /* the image-1 border, warped: the "keystone" that says projective */
      const bd = [];
      [[0, 0], [IW, 0], [IW, IH], [0, IH], [0, 0]].forEach((c, ci, arr) => {
        if (ci === 0) return;
        const a = arr[ci - 1];
        for (let j = 0; j <= 12; j++) {
          const q = VZ.applyH(Hf, [VZ.lerp(a[0], c[0], j / 12), VZ.lerp(a[1], c[1], j / 12)]);
          if (q) bd.push(B2S(q));
        }
      });
      if (bd.length > 2) VZ.poly(gW, bd, { stroke: VC.accent, w: 1.4, close: true, fill: VC.accent, fillOp: 0.05 });
    }
    for (let i = 0; i < nex; i++) {
      const m = dst[4 + i], pr = VZ.applyH(Hf, EXS[i]);
      const qm = B2S(m);
      gB.append("circle").attr("cx", qm[0]).attr("cy", qm[1]).attr("r", 2.6).attr("fill", VC.teal);
      if (pr) {
        const qp = B2S(pr);
        gB.append("line").attr("x1", qm[0]).attr("y1", qm[1])
          .attr("x2", qm[0] + 6 * (qp[0] - qm[0])).attr("y2", qm[1] + 6 * (qp[1] - qm[1]))
          .attr("stroke", VC.bad).attr("stroke-width", 1).attr("stroke-opacity", 0.8);
      }
    }
    AM.drag(gB, DST, B2S, S2B, draw, { r: 6.5 });

    /* ---- bottom: the matrix and the spectrum ------------------------------- */
    VZ.panelBox(g, 18, 330, 300, 118, "Ĥ, scaled so that h₂₂ = 1");
    VZ.matText(g, Hf, 34, 366, { dp: 4, pad: 10, size: 11, lead: 16 });
    VZ.panelBox(g, 336, 330, 410, 118, "the design matrix A · singular values, log scale");
    const gs = g.append("g");
    const sv = fit ? fit.sv : [];
    const y = d3.scaleLog().domain([1e-9, 1e7]).range([432, 348]).clamp(true);
    [1e-6, 1e-3, 1, 1e3, 1e6].forEach(v => {
      gs.append("line").attr("x1", 348).attr("x2", 736).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", VC.grid);
      gs.append("text").attr("x", 350).attr("y", y(v) - 2).attr("font-size", 8.5).attr("fill", VC.muted)
        .text(v >= 1 ? d3.format(".0e")(v) : d3.format(".0e")(v));
    });
    sv.forEach((s, i) => {
      const x = 400 + i * 36;
      gs.append("circle").attr("cx", x).attr("cy", y(Math.max(s, 1e-9))).attr("r", 4)
        .attr("fill", i === 7 ? VC.a2 : (i === 8 ? VC.muted : VC.accent)).attr("fill-opacity", 0.9);
      gs.append("text").attr("x", x).attr("y", 442).attr("text-anchor", "middle").attr("font-size", 8.5)
        .attr("fill", i === 7 ? VC.a2 : VC.muted).text("σ" + (i + 1));
    });

    /* ---- readout ----------------------------------------------------------- */
    const res = AM.rms(Hf, src, dst);
    const Hi = VZ.inv3(Hf);
    let ident = NaN;
    if (Hi) { const I = VZ.mul(Hf, Hi); ident = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) ident = Math.max(ident, Math.abs(I[i][j] - (i === j ? 1 : 0))); }
    let s = `${norm ? "Hartley-normalised" : "unnormalised"} DLT on <b>${src.length}</b> correspondences` +
            ` · cond(A) = <span class="keep">σ</span>₁/<span class="keep">σ</span>₈ = <b>${fit ? d3.format(".4~e")(fit.cond) : "—"}</b>` +
            ` · RMS residual ${VZ.fmt(res, 4)} px · max |Ĥ Ĥ⁻¹ − I| = ${ident.toExponential(1)}`;
    if (pristine() && nex === 0) {
      let e = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) e = Math.max(e, Math.abs(Hf[i][j] - HTRUE[i][j]));
      s += `<br />the worked example, untouched: max |Ĥ − H_true| = <b>${e.toExponential(2)}</b>. ` +
           `The unnormalised system has cond ≈ 1.04 × 10⁶ and the normalised one ≈ 3.40; both still recover H exactly in double precision, which is the honest version of this story (§07).`;
    } else {
      s += `<br />the four handles have been moved or extra points added, so the generating transformation is no longer H_true and that comparison is retired.`;
    }
    ui.out.innerHTML = s;
  }

  ui.reset.addEventListener("click", () => { DST = DST0.map(p => p.slice()); ui.kk.value = 0; draw(); });
  [ui.n, ui.kk, ui.s, ui.grid].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ═══ FIG 4 · §07 — what normalisation does to the design matrix ═════════════
   The spectra are drawn RELATIVE to σ₁, so the eighth point's height is exactly
   1/cond and the two systems can be compared on one axis whatever the image
   size. The nearly-collinear configuration is the control: both systems degrade
   together there, because that failure is geometric, not numerical.            */
(function () {
  const svg = d3.select("#norm-svg");
  if (svg.empty()) return;
  const W = 760, H = 430;
  const HTRUE = [[0.90, 0.15, 40], [-0.10, 1.05, -25], [0.0002, 0.0003, 1]];

  const rn = VZ.rng(9091);
  const U = d3.range(40).map(() => [rn(), rn(), VZ.randn(rn), VZ.randn(rn)]);

  const ui = {
    c: document.getElementById("nm-c"), w: document.getElementById("nm-w"), wv: document.getElementById("nm-wv"),
    o: document.getElementById("nm-o"), ov: document.getElementById("nm-ov"),
    n: document.getElementById("nm-n"), nv: document.getElementById("nm-nv"),
    out: document.getElementById("norm-readout")
  };

  function config(kind, iw, ih, N) {
    const P = [];
    for (let i = 0; i < N; i++) {
      const a = U[i][0], b = U[i][1];
      if (kind === "spread") P.push([0.05 * iw + 0.90 * iw * a, 0.05 * ih + 0.90 * ih * b]);
      else if (kind === "corner") P.push([0.76 * iw + 0.20 * iw * a, 0.76 * ih + 0.20 * ih * b]);
      else if (kind === "line") P.push([0.08 * iw + 0.84 * iw * a, 0.20 * ih + 0.60 * ih * a + 0.004 * ih * U[i][2]]);
      else P.push([0.5 * iw + 0.055 * iw * Math.cos(2 * Math.PI * a), 0.5 * ih + 0.055 * iw * Math.sin(2 * Math.PI * a)]);
    }
    return P;
  }

  function draw() {
    const kind = ui.c.value, iw = +ui.w.value, off = +ui.o.value, N = +ui.n.value;
    const ih = Math.round(0.75 * iw);
    ui.wv.textContent = iw + " px"; ui.ov.textContent = off; ui.nv.textContent = N;

    /* H_true is written for a 640-wide image; conjugate it by the scaling similarity
       so that the projective character is preserved at every image size */
    const sc = iw / 640, S = [[sc, 0, 0], [0, sc, 0], [0, 0, 1]], Si = VZ.inv3(S);
    const Hs = VZ.mul(VZ.mul(S, HTRUE), Si);

    const loc = config(kind, iw, ih, N);
    const src = loc.map(p => [p[0] + off, p[1] + off]);
    const dst = loc.map((p, i) => {
      const q = VZ.applyH(Hs, p);
      return [q[0] + off + U[i][2], q[1] + off + U[i][3]];      // σ = 1 px measurement noise
    });

    const rawA = AM.dltRows(src, dst);
    const svR = AM.svals(rawA), condR = svR[7] > 0 ? svR[0] / svR[7] : Infinity;
    const fit = AM.dlt(src, dst, true);
    const svN = fit ? fit.sv : svR, condN = fit ? fit.cond : Infinity;
    const T1 = AM.hnorm(src);
    const nrm = src.map(p => VZ.applyH(T1, p));

    const f = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = f.g;

    /* ---- panel A: raw ------------------------------------------------------ */
    VZ.panelBox(g, 14, 40, 250, 296, "raw pixel coordinates");
    const bx = d3.scaleLinear().domain([off, off + iw]).range([22, 256]);
    const by = d3.scaleLinear().domain([off, off + ih]).range([56, 56 + 234 * (ih / iw)]);
    const gA = g.append("g");
    gA.append("rect").attr("x", bx(off)).attr("y", by(off)).attr("width", bx(off + iw) - bx(off))
      .attr("height", by(off + ih) - by(off)).attr("fill", VC.panel2).attr("stroke", VC.line);
    let cx = 0, cy = 0; src.forEach(p => { cx += p[0]; cy += p[1]; }); cx /= N; cy /= N;
    let md = 0; src.forEach(p => md += Math.hypot(p[0] - cx, p[1] - cy)); md /= N;
    gA.append("circle").attr("cx", bx(cx)).attr("cy", by(cy)).attr("r", Math.max(1, (bx(off + md) - bx(off))))
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.7);
    src.forEach(p => gA.append("circle").attr("cx", bx(p[0])).attr("cy", by(p[1])).attr("r", 2.6)
      .attr("fill", VC.accent).attr("fill-opacity", 0.9));
    gA.append("path").attr("d", `M${bx(cx) - 6},${by(cy)} H${bx(cx) + 6} M${bx(cx)},${by(cy) - 6} V${by(cy) + 6}`)
      .attr("stroke", VC.a2).attr("stroke-width", 1.6);
    g.append("text").attr("x", 22).attr("y", 330).attr("font-size", 10).attr("fill", VC.muted)
      .text(`centroid (${Math.round(cx)}, ${Math.round(cy)}) · mean radius ${Math.round(md)} px`);

    /* ---- panel B: normalised ---------------------------------------------- */
    VZ.panelBox(g, 288, 40, 200, 296, "after T · centroid 0, mean radius √2");
    const nx = d3.scaleLinear().domain([-3.2, 3.2]).range([298, 478]);
    const ny = d3.scaleLinear().domain([-3.2, 3.2]).range([58, 238]);
    const gB = g.append("g");
    [1, Math.SQRT2].forEach((r, i) => {
      gB.append("circle").attr("cx", nx(0)).attr("cy", ny(0)).attr("r", nx(r) - nx(0))
        .attr("fill", "none").attr("stroke", i ? VC.a2 : VC.grid).attr("stroke-dasharray", i ? "3 3" : null);
    });
    gB.append("line").attr("x1", nx(-3.2)).attr("x2", nx(3.2)).attr("y1", ny(0)).attr("y2", ny(0)).attr("stroke", VC.grid);
    gB.append("line").attr("y1", ny(-3.2)).attr("y2", ny(3.2)).attr("x1", nx(0)).attr("x2", nx(0)).attr("stroke", VC.grid);
    nrm.forEach(p => gB.append("circle").attr("cx", nx(VZ.clamp(p[0], -3.2, 3.2))).attr("cy", ny(VZ.clamp(p[1], -3.2, 3.2)))
      .attr("r", 2.8).attr("fill", VC.good).attr("fill-opacity", 0.9));
    g.append("text").attr("x", 298).attr("y", 262).attr("font-size", 10).attr("fill", VC.muted)
      .text("every configuration lands looking like this —");
    g.append("text").attr("x", 298).attr("y", 276).attr("font-size", 10).attr("fill", VC.muted)
      .text("which is the entire purpose of the step.");
    g.append("text").attr("x", 298).attr("y", 300).attr("font-size", 10).attr("fill", VC.muted)
      .text(`s = ${VZ.sig(T1[0][0], 4)}  (√2 / mean radius)`);

    /* ---- panel C: the two spectra, relative to σ₁ -------------------------- */
    VZ.panelBox(g, 512, 40, 234, 296, "σᵢ / σ₁ · log scale");
    const y = d3.scaleLog().domain([1e-12, 1]).range([320, 62]).clamp(true);
    const gC = g.append("g");
    [1, 1e-3, 1e-6, 1e-9, 1e-12].forEach(v => {
      gC.append("line").attr("x1", 520).attr("x2", 738).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", VC.grid);
      gC.append("text").attr("x", 522).attr("y", y(v) - 3).attr("font-size", 8.5).attr("fill", VC.muted)
        .text(d3.format(".0e")(v));
    });
    [[svR, VC.bad, 566, "raw"], [svN, VC.good, 672, "normalised"]].forEach(([sv, col, x0, lab]) => {
      const s1 = sv[0] || 1;
      sv.forEach((s, i) => {
        if (i > 7) return;
        gC.append("circle").attr("cx", x0).attr("cy", y(Math.max(s / s1, 1e-12))).attr("r", i === 7 ? 5 : 3)
          .attr("fill", col).attr("fill-opacity", i === 7 ? 1 : 0.55);
      });
      gC.append("line").attr("x1", x0).attr("x2", x0).attr("y1", y(1)).attr("y2", y(Math.max(sv[7] / s1, 1e-12)))
        .attr("stroke", col).attr("stroke-width", 1).attr("stroke-opacity", 0.45);
      gC.append("text").attr("x", x0).attr("y", 334).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", col).text(lab);
    });
    VZ.legend(gC, [{ color: VC.bad, label: "unnormalised" }, { color: VC.good, label: "Hartley" }], 522, 348, { gap: 14 });

    ui.out.innerHTML =
      `<b>cond(A) = <span class="keep">σ</span>₁/<span class="keep">σ</span>₈</b> — unnormalised <b style="color:${VC.bad}">${d3.format(".4~e")(condR)}</b>` +
      ` · Hartley-normalised <b style="color:${VC.good}">${VZ.fmt(condN, 3)}</b> · ratio ${d3.format(".3~e")(condR / condN)}` +
      `<br />${N} points, ${iw} × ${ih} image, coordinates offset by ${off}. ` +
      (kind === "line"
        ? `Nearly collinear — the one configuration normalisation cannot rescue. The normalised condition number climbs to ${VZ.fmt(condN, 1)}, about ${Math.round(condN / 4.1)}× its value on spread points, because this degeneracy is <i>geometric</i>: a line carries 2 degrees of freedom, not 8, and no change of units can manufacture the missing constraints (§03). Reject such samples; do not try to condition them away.`
        : "Notice how little the normalised number moves as the image grows and the coordinates are offset, and how far the raw one climbs. That stability is what you are buying.");
  }

  [ui.c, ui.w, ui.o, ui.n].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ── extra AM helpers: non-linear refinement of a homography ─────────────────
   The forward Jacobian is the one printed in §08; the extra rows of the
   symmetric cost are differentiated numerically, because H⁻¹'s dependence on h
   is not worth writing out and eight extra function evaluations cost nothing. */
Object.assign(AM, {
  hFromVec: function (h) { return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]]; },
  hToVec: function (H) { const w = H[2][2]; return [H[0][0] / w, H[0][1] / w, H[0][2] / w, H[1][0] / w, H[1][1] / w, H[1][2] / w, H[2][0] / w, H[2][1] / w]; },

  /* r(h): forward residuals, and the symmetric ones appended when sym is set */
  hResid: function (h, src, dst, sym) {
    const H = AM.hFromVec(h), r = [];
    for (let i = 0; i < src.length; i++) {
      const p = VZ.applyH(H, src[i]);
      if (!p) { r.push(1e6, 1e6); continue; }
      r.push(p[0] - dst[i][0], p[1] - dst[i][1]);
    }
    if (sym) {
      const Hi = VZ.inv3(H);
      for (let i = 0; i < src.length; i++) {
        const q = Hi ? VZ.applyH(Hi, dst[i]) : null;
        if (!q) { r.push(1e6, 1e6); continue; }
        r.push(q[0] - src[i][0], q[1] - src[i][1]);
      }
    }
    return r;
  },

  hJac: function (h, src, dst, sym) {
    const H = AM.hFromVec(h), J = [];
    for (let i = 0; i < src.length; i++) {
      const x = src[i][0], y = src[i][1];
      const w = h[6] * x + h[7] * y + 1;
      const u = (h[0] * x + h[1] * y + h[2]) / w, v = (h[3] * x + h[4] * y + h[5]) / w;
      J.push([x / w, y / w, 1 / w, 0, 0, 0, -u * x / w, -u * y / w]);
      J.push([0, 0, 0, x / w, y / w, 1 / w, -v * x / w, -v * y / w]);
    }
    if (sym) {
      /* central differences on the backward block only */
      const n2 = 2 * src.length, rows = [];
      for (let k = 0; k < n2; k++) rows.push(new Array(8).fill(0));
      for (let j = 0; j < 8; j++) {
        const e = Math.max(1e-7, Math.abs(h[j]) * 1e-6);
        const hp = h.slice(), hm = h.slice();
        hp[j] += e; hm[j] -= e;
        const rp = AM.hResid(hp, src, dst, true).slice(n2);
        const rm = AM.hResid(hm, src, dst, true).slice(n2);
        for (let k = 0; k < n2; k++) rows[k][j] = (rp[k] - rm[k]) / (2 * e);
      }
      rows.forEach(r => J.push(r));
    }
    return J;
  },

  /* Levenberg–Marquardt. Returns the refined H and the objective at every step. */
  refineH: function (H0, src, dst, opt) {
    const o = Object.assign({ iters: 12, sym: false, lam: 1e-3 }, opt || {});
    let h = AM.hToVec(H0), lam = o.lam;
    const cost = hh => { const r = AM.hResid(hh, src, dst, o.sym); return r.reduce((s, v) => s + v * v, 0); };
    let E = cost(h);
    const hist = [E];
    for (let it = 0; it < o.iters; it++) {
      const J = AM.hJac(h, src, dst, o.sym), r = AM.hResid(h, src, dst, o.sym);
      const JT = VZ.T(J), N = VZ.mul(JT, J), g = VZ.mv(JT, r);
      let ok = false;
      for (let tries = 0; tries < 8 && !ok; tries++) {
        const M = N.map((row, i) => row.map((v, j) => v + (i === j ? lam * (N[i][i] || 1) : 0)));
        const Mi = VZ.invN(M);
        if (!Mi) { lam *= 10; continue; }
        const d = VZ.mv(Mi, g);
        const hn = h.map((v, i) => v - d[i]);
        const En = cost(hn);
        if (isFinite(En) && En < E) { h = hn; E = En; lam = Math.max(1e-9, lam * 0.4); ok = true; }
        else lam *= 10;
      }
      hist.push(E);
      if (!ok) break;
    }
    return { H: AM.hFromVec(h), hist: hist };
  },

  /* mean transfer error of H against a reference map, over a grid of the image */
  transferErr: function (H, Href, w, h, n) {
    let s = 0, k = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const p = [w * (i + 0.5) / n, h * (j + 0.5) / n];
      const a = VZ.applyH(H, p), b = VZ.applyH(Href, p);
      if (!a || !b) continue;
      s += Math.hypot(a[0] - b[0], a[1] - b[1]); k++;
    }
    return k ? s / k : NaN;
  }
});

/* ═══ FIG 5 · §08 — algebraic against geometric ══════════════════════════════
   Image 1 is shaded by the homogeneous scale w, which IS the weight the
   algebraic error applies. At zero projective strength the shading is flat and
   refinement does nothing; as it rises the shading spreads and the gap opens.  */
(function () {
  const svg = d3.select("#refine-svg");
  if (svg.empty()) return;
  const W = 760, H = 450, IW = 640, IH = 480, k = 0.40;

  const rn = VZ.rng(60613);
  const PT = d3.range(40).map(() => [30 + rn() * 580, 30 + rn() * 420]);
  const TRIALS = 25;               // the readout is an ENSEMBLE mean, not one draw
  const N1 = d3.range(TRIALS).map(() => d3.range(40).map(() => [VZ.randn(rn), VZ.randn(rn)]));
  const N2 = d3.range(TRIALS).map(() => d3.range(40).map(() => [VZ.randn(rn), VZ.randn(rn)]));

  const ui = {
    p: document.getElementById("rf-p"), pv: document.getElementById("rf-pv"),
    s: document.getElementById("rf-s"), sv: document.getElementById("rf-sv"),
    n: document.getElementById("rf-n"), nv: document.getElementById("rf-nv"),
    b: document.getElementById("rf-b"), e: document.getElementById("rf-e"),
    out: document.getElementById("refine-readout")
  };

  function draw() {
    const str = +ui.p.value / 100, sig = +ui.s.value, n = +ui.n.value;
    const both = ui.b.checked, sym = ui.e.value === "sym";
    ui.pv.textContent = Math.round(str * 100) + "%";
    ui.sv.textContent = VZ.fmt(sig, 2) + " px";
    ui.nv.textContent = n;

    const HT = [[0.90, 0.15, 40], [-0.10, 1.05, -25], [-0.00110 * str, 0.00090 * str, 1]];
    function trial(t) {
      const src = [], dst = [];
      for (let i = 0; i < n; i++) {
        const p = VZ.applyH(HT, PT[i]);
        src.push(both ? [PT[i][0] + sig * N1[t][i][0], PT[i][1] + sig * N1[t][i][1]] : PT[i].slice());
        dst.push([p[0] + sig * N2[t][i][0], p[1] + sig * N2[t][i][1]]);
      }
      const d0 = AM.dlt(src, dst, true);
      const H0 = d0 ? d0.H : VZ.eye(3);
      const rr = AM.refineH(H0, src, dst, { iters: 12, sym: sym });
      return { src: src, dst: dst, H0: H0, ref: rr,
               eD: AM.transferErr(H0, HT, IW, IH, 10), eR: AM.transferErr(rr.H, HT, IW, IH, 10) };
    }
    const T0 = trial(0);
    const src = T0.src, dst = T0.dst, H0 = T0.H0, ref = T0.ref;
    let sD = 0, sR = 0, cnt = 0;
    for (let t = 0; t < TRIALS; t++) {
      const T = (t === 0) ? T0 : trial(t);
      if (isFinite(T.eD) && isFinite(T.eR)) { sD += T.eD; sR += T.eR; cnt++; }
    }
    const eD = sD / cnt, eR = sR / cnt;

    const f = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = f.g;

    /* ---- panel A: the algebraic weight w ---------------------------------- */
    VZ.panelBox(g, 14, 40, IW * k, IH * k, "image 1, shaded by w = h₂₀x + h₂₁y + 1");
    const GW = 32, GH = 24;
    let lo = Infinity, hi = -Infinity;
    const wv = [];
    for (let j = 0; j < GH; j++) {
      wv.push([]);
      for (let i = 0; i < GW; i++) {
        const x = IW * (i + 0.5) / GW, y = IH * (j + 0.5) / GH;
        const v = HT[2][0] * x + HT[2][1] * y + 1;
        wv[j].push(v); if (v < lo) lo = v; if (v > hi) hi = v;
      }
    }
    const cs = d3.scaleLinear().domain([Math.min(lo, 0.98), Math.max(hi, 1.02)]).range([0, 1]);
    VZ.cells(g, 14, 40, IW * k / GW, GW, GH,
      (x, y) => d3.interpolateCividis(VZ.clamp(cs(wv[y][x]), 0, 1)));
    const gA = g.append("g");
    for (let i = 0; i < n; i++) {
      gA.append("circle").attr("cx", 14 + src[i][0] * k).attr("cy", 40 + src[i][1] * k).attr("r", 2.4)
        .attr("fill", VC.a2).attr("fill-opacity", 0.95);
    }
    g.append("text").attr("x", 14).attr("y", 254).attr("font-size", 10).attr("fill", VC.muted)
      .text(`w ranges ${VZ.fmt(lo, 3)} … ${VZ.fmt(hi, 3)}  (ratio ${VZ.fmt(hi / lo, 2)}×)`);
    g.append("text").attr("x", 14).attr("y", 268).attr("font-size", 10).attr("fill", VC.muted)
      .text("the algebraic error weights each point by w².");

    /* ---- panel B: image 2 -------------------------------------------------- */
    VZ.panelBox(g, 292, 40, IW * k, IH * k, "image 2 · DLT ■  refined ▲  measured ○");
    const gB = g.append("g");
    const B = p => [292 + p[0] * k, 40 + p[1] * k];
    for (let i = 0; i < n; i++) {
      const m = B(dst[i]), a = VZ.applyH(H0, src[i]), b = VZ.applyH(ref.H, src[i]);
      gB.append("circle").attr("cx", m[0]).attr("cy", m[1]).attr("r", 3)
        .attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 1);
      if (a) { const q = B(a); gB.append("line").attr("x1", m[0]).attr("y1", m[1])
        .attr("x2", m[0] + 8 * (q[0] - m[0])).attr("y2", m[1] + 8 * (q[1] - m[1]))
        .attr("stroke", VC.bad).attr("stroke-width", 1).attr("stroke-opacity", 0.75); }
      if (b) { const q = B(b); gB.append("line").attr("x1", m[0]).attr("y1", m[1])
        .attr("x2", m[0] + 8 * (q[0] - m[0])).attr("y2", m[1] + 8 * (q[1] - m[1]))
        .attr("stroke", VC.good).attr("stroke-width", 1).attr("stroke-opacity", 0.75); }
    }
    VZ.legend(gB, [{ color: VC.bad, label: "DLT residual ×8" }, { color: VC.good, label: "refined residual ×8" }], 298, 250, { gap: 14 });

    /* ---- panel C: convergence --------------------------------------------- */
    VZ.panelBox(g, 570, 40, 176, 232, "objective vs iteration");
    const hist = ref.hist, m0 = d3.max(hist), m1 = Math.max(d3.min(hist), 1e-9);
    const yy = d3.scaleLog().domain([Math.max(m1 * 0.9, 1e-9), Math.max(m0 * 1.1, m1 * 1.2)]).range([258, 60]).clamp(true);
    const xx = d3.scaleLinear().domain([0, Math.max(1, hist.length - 1)]).range([584, 736]);
    const gC = g.append("g");
    yy.ticks(4).forEach(v => {
      gC.append("line").attr("x1", 584).attr("x2", 736).attr("y1", yy(v)).attr("y2", yy(v)).attr("stroke", VC.grid);
      gC.append("text").attr("x", 586).attr("y", yy(v) - 2).attr("font-size", 8.5).attr("fill", VC.muted).text(d3.format(".3~s")(v));
    });
    VZ.poly(gC, hist.map((v, i) => [xx(i), yy(v)]), { stroke: VC.accent, w: 1.8, close: false });
    hist.forEach((v, i) => gC.append("circle").attr("cx", xx(i)).attr("cy", yy(v)).attr("r", 2.4).attr("fill", VC.accent));
    gC.append("text").attr("x", 584).attr("y", 272).attr("font-size", 9.5).attr("fill", VC.muted)
      .text(`iteration 0 → ${hist.length - 1}`);

    const kvp = VZ.kv(g, 570, 300, { keyW: 118, size: 10.5, lead: 15 });
    kvp("DLT transfer error", VZ.fmt(eD, 4) + " px", VC.bad);
    kvp("refined", VZ.fmt(eR, 4) + " px", eR < eD ? VC.good : VC.bad);
    kvp("change in error", (eR < eD ? "" : "+") + VZ.fmt(100 * (eR - eD) / eD, 2) + " %", eR < eD ? VC.good : VC.bad, true);
    kvp("cost, start → end", d3.format(".4~s")(hist[0]) + " → " + d3.format(".4~s")(hist[hist.length - 1]));
    kvp("averaged over", TRIALS + " noise draws");

    ui.out.innerHTML =
      `w spans ${VZ.fmt(lo, 3)}…${VZ.fmt(hi, 3)} (ratio ${VZ.fmt(hi / lo, 2)}×) · mean over ${TRIALS} noise draws: normalised DLT transfer error <b>${VZ.fmt(eD, 4)} px</b>` +
      ` → ${sym ? "symmetric" : "forward"} Gauss–Newton <b>${VZ.fmt(eR, 4)} px</b> — a change of <b style="color:${eR < eD ? VC.good : VC.bad}">${eR < eD ? "" : "+"}${VZ.fmt(100 * (eR - eD) / eD, 2)}%</b> in the transfer error` +
      `<br />` + (both && !sym
        ? "Noise in image 1 <i>and</i> the forward error selected: the model insists image 1's coordinates are exact when they are not, so refinement typically makes the estimate worse than the DLT it started from. Switch to the symmetric error."
        : (str < 0.15
          ? "At near-zero projective strength w ≈ 1 everywhere, so the algebraic error already is the geometric one and refinement has nothing to remove."
          : "The shading in the left panel is the weight the DLT silently applied. Refinement's job is to take it away."));
  }

  [ui.p, ui.s, ui.n, ui.b, ui.e].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ── extra AM helpers: minimal-sample homography and a degeneracy test ────── */
Object.assign(AM, {
  /* The 4-point homography as an 8 × 8 linear solve with h₂₂ = 1. Much cheaper than
     the SVD/Jacobi route and exact for a minimal sample; returns null when h₂₂ = 0,
     which is the one configuration this parameterisation cannot express. */
  h4pt: function (s, d) {
    const A = [], b = [];
    for (let i = 0; i < 4; i++) {
      const x = s[i][0], y = s[i][1], u = d[i][0], v = d[i][1];
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
      A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
    }
    const Ai = VZ.invN(A);
    if (!Ai) return null;
    const h = VZ.mv(Ai, b);
    if (!h.every(isFinite)) return null;
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
  },

  /* Degenerate ⟺ some triple is nearly collinear, in EITHER image. The triangle
     area is compared against the sample's own scale, so the test is scale-free. */
  degenerate4: function (P, tol) {
    const t = (tol === undefined) ? 0.002 : tol;   // ≈ 5% rejection on uniform points, measured
    let sc = 0;
    for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) sc = Math.max(sc, Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]));
    if (sc < 1e-9) return true;
    const tri = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    for (const [a, b, c] of tri) {
      const ar = Math.abs((P[b][0] - P[a][0]) * (P[c][1] - P[a][1]) - (P[b][1] - P[a][1]) * (P[c][0] - P[a][0])) / 2;
      if (ar < t * sc * sc) return true;
    }
    return false;
  }
});

/* ═══ FIG 6 · §09 — RANSAC fitting a homography ══════════════════════════════
   The estimator is Part 4's; what is on show here is the homography-specific
   inside of the loop — the minimal sample of four, the degeneracy rejection,
   and the final refit, whose value is printed rather than asserted.            */
(function () {
  const svg = d3.select("#ransac-svg");
  if (svg.empty()) return;
  const W = 760, H = 470, IW = 640, IH = 480, k = 0.40, NPT = 120;

  const HTRUE = [[0.90, 0.15, 40], [-0.10, 1.05, -25], [-0.00055, 0.00045, 1]];
  let seed = 31337, src = [], dstIn = [], dstOut = [], NZ = [];
  let st = null;

  const ui = {
    o: document.getElementById("rs-o"), ov: document.getElementById("rs-ov"),
    t: document.getElementById("rs-t"), tv: document.getElementById("rs-tv"),
    s: document.getElementById("rs-s"), sv: document.getElementById("rs-sv"),
    deg: document.getElementById("rs-deg"), ad: document.getElementById("rs-ad"),
    step: document.getElementById("rs-step"), run: document.getElementById("rs-run"),
    nw: document.getElementById("rs-new"), out: document.getElementById("ransac-readout")
  };

  function makeData() {
    const r = VZ.rng(seed);
    src = []; dstIn = []; dstOut = []; NZ = [];
    for (let i = 0; i < NPT; i++) {
      const p = [24 + r() * 592, 24 + r() * 432];
      src.push(p);
      dstIn.push(VZ.applyH(HTRUE, p));
      dstOut.push([24 + r() * 592, 24 + r() * 432]);
      NZ.push([VZ.randn(r), VZ.randn(r)]);
    }
  }

  function dataset() {
    const frac = +ui.o.value / 100, sig = +ui.s.value;
    const nOut = Math.round(frac * NPT);
    const dst = [], isOut = [];
    for (let i = 0; i < NPT; i++) {
      const bad = i < nOut;                 // a fixed prefix, so the slider is monotone
      isOut.push(bad);
      dst.push(bad ? dstOut[i].slice() : [dstIn[i][0] + sig * NZ[i][0], dstIn[i][1] + sig * NZ[i][1]]);
    }
    return { dst: dst, isOut: isOut, nOut: nOut };
  }

  function reset() {
    st = { n: 0, degRej: 0, best: [], bestH: null, hist: [], N: 1e9, rng: VZ.rng(seed ^ 0x5bd1), done: false };
    const p = 0.99, w0 = 0.2;
    st.N = Math.ceil(Math.log(1 - p) / Math.log(1 - Math.pow(w0, 4)));
  }

  function sample(D, t) {
    const r = st.rng, idx = [];
    while (idx.length < 4) { const j = Math.floor(r() * NPT); if (idx.indexOf(j) < 0) idx.push(j); }
    st.n++;
    const S = idx.map(i => src[i]), T = idx.map(i => D.dst[i]);
    st.last = idx;
    if (ui.deg.checked && (AM.degenerate4(S) || AM.degenerate4(T))) { st.degRej++; st.lastDeg = true; return; }
    st.lastDeg = false;
    const Hh = AM.h4pt(S, T);
    if (!Hh) return;
    const inl = [];
    for (let i = 0; i < NPT; i++) {
      const q = VZ.applyH(Hh, src[i]);
      if (q && Math.hypot(q[0] - D.dst[i][0], q[1] - D.dst[i][1]) < t) inl.push(i);
    }
    if (inl.length > st.best.length) {
      st.best = inl; st.bestH = Hh; st.bestSample = idx.slice();
      if (ui.ad.checked) {
        const w = Math.max(0.05, inl.length / NPT);
        st.N = Math.max(4, Math.ceil(Math.log(0.01) / Math.log(1 - Math.pow(w, 4))));
      }
    }
    st.hist.push([st.n, st.best.length]);
  }

  function draw() {
    const t = +ui.t.value, sig = +ui.s.value;
    ui.ov.textContent = ui.o.value + "%"; ui.tv.textContent = VZ.fmt(t, 1) + " px";
    ui.sv.textContent = VZ.fmt(sig, 2) + " px";
    const D = dataset();
    const inSet = new Set(st.best);

    const f = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = f.g;

    VZ.panelBox(g, 14, 44, IW * k, IH * k, "image 1 · every putative match");
    VZ.panelBox(g, 292, 44, IW * k, IH * k, `image 2 · residuals against the best H so far (t = ${VZ.fmt(t, 1)} px)`);
    const A = p => [14 + p[0] * k, 44 + p[1] * k], B = p => [292 + p[0] * k, 44 + p[1] * k];

    for (let i = 0; i < NPT; i++) {
      const on = inSet.has(i);
      const a = A(src[i]);
      g.append("circle").attr("cx", a[0]).attr("cy", a[1]).attr("r", on ? 2.6 : 2.2)
        .attr("fill", on ? VC.good : VC.bad).attr("fill-opacity", on ? 0.9 : 0.55);
      const m = B(D.dst[i]);
      if (st.bestH) {
        const q = VZ.applyH(st.bestH, src[i]);
        if (q) {
          const p2 = B(q);
          g.append("line").attr("x1", p2[0]).attr("y1", p2[1]).attr("x2", m[0]).attr("y2", m[1])
            .attr("stroke", on ? VC.good : VC.bad).attr("stroke-width", on ? 1.4 : 0.9)
            .attr("stroke-opacity", on ? 0.95 : 0.4);
        }
      }
      g.append("circle").attr("cx", m[0]).attr("cy", m[1]).attr("r", 1.8)
        .attr("fill", on ? VC.good : VC.bad).attr("fill-opacity", on ? 0.85 : 0.4);
    }
    /* the four points of the sample just drawn */
    (st.last || []).forEach(i => {
      const a = A(src[i]), b = B(D.dst[i]);
      [[a, 14, 44], [b, 292, 44]].forEach(([q]) => {
        g.append("circle").attr("cx", q[0]).attr("cy", q[1]).attr("r", 6)
          .attr("fill", "none").attr("stroke", st.lastDeg ? VC.bad : VC.a2).attr("stroke-width", 1.8);
      });
    });

    /* ---- consensus history ------------------------------------------------- */
    VZ.panelBox(g, 14, 278, 534, 168, "best consensus size vs samples drawn");
    const gx = d3.scaleLinear().domain([0, Math.max(30, st.n)]).range([26, 540]);
    const gy = d3.scaleLinear().domain([0, NPT]).range([432, 292]);
    const gh = g.append("g");
    [0, 30, 60, 90, 120].forEach(v => {
      gh.append("line").attr("x1", 26).attr("x2", 540).attr("y1", gy(v)).attr("y2", gy(v)).attr("stroke", VC.grid);
      gh.append("text").attr("x", 18).attr("y", gy(v) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted).text(v);
    });
    gh.append("line").attr("x1", 26).attr("x2", 540).attr("y1", gy(NPT - D.nOut)).attr("y2", gy(NPT - D.nOut))
      .attr("stroke", VC.a2).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.2);
    gh.append("text").attr("x", 534).attr("y", gy(NPT - D.nOut) - 4).attr("text-anchor", "end")
      .attr("font-size", 9).attr("fill", VC.a2).text(`${NPT - D.nOut} true inliers`);
    if (st.hist.length > 1) VZ.poly(gh, st.hist.map(d => [gx(d[0]), gy(d[1])]), { stroke: VC.accent, w: 1.8, close: false });
    if (isFinite(st.N) && st.N < gx.domain()[1] * 4) {
      const xN = gx(Math.min(st.N, gx.domain()[1]));
      gh.append("line").attr("x1", xN).attr("x2", xN).attr("y1", 292).attr("y2", 432)
        .attr("stroke", VC.violet).attr("stroke-dasharray", "3 3");
      gh.append("text").attr("x", xN + 4).attr("y", 300).attr("font-size", 9).attr("fill", VC.violet).text("N = " + st.N);
    }
    gh.append("text").attr("x", 26).attr("y", 444).attr("font-size", 9.5).attr("fill", VC.muted).text("samples drawn →");

    /* ---- stats column ------------------------------------------------------ */
    VZ.panelBox(g, 570, 44, 176, 402, "the run");
    const kv = VZ.kv(g, 582, 76, { keyW: 112, size: 10.5, lead: 15.5 });
    kv("samples drawn", String(st.n));
    kv("rejected degenerate", String(st.degRej), st.degRej ? VC.a2 : VC.muted);
    kv("best consensus", `${st.best.length} / ${NPT}`, VC.good, true);
    kv("true inliers", `${NPT - D.nOut} / ${NPT}`, VC.a2);
    kv("ŵ", VZ.fmt(st.best.length / NPT, 3));
    kv("N now required", isFinite(st.N) ? String(st.N) : "—", VC.violet);
    const wTrue = (NPT - D.nOut) / NPT;
    kv("N at the true w", String(wTrue > 0.02 ? Math.ceil(Math.log(0.01) / Math.log(1 - Math.pow(wTrue, 4))) : "—"));

    let eMin = NaN, eRef = NaN, nRef = 0, nAfter = 0;
    if (st.bestH && st.best.length >= 4) {
      eMin = AM.transferErr(st.bestH, HTRUE, IW, IH, 8);
      const S2 = st.best.map(i => src[i]), D2 = st.best.map(i => D.dst[i]);
      const d0 = AM.dlt(S2, D2, true);
      if (d0) {
        const rr = AM.refineH(d0.H, S2, D2, { iters: 8, sym: true });
        eRef = AM.transferErr(rr.H, HTRUE, IW, IH, 8); nRef = S2.length;
        for (let i = 0; i < NPT; i++) {
          const q = VZ.applyH(rr.H, src[i]);
          if (q && Math.hypot(q[0] - D.dst[i][0], q[1] - D.dst[i][1]) < t) nAfter++;
        }
      }
    }
    const kv2 = VZ.kv(g, 582, 244, { keyW: 112, size: 10.5, lead: 15.5 });
    kv2("H from 4 points", isFinite(eMin) ? VZ.fmt(eMin, 3) + " px" : "—", VC.bad);
    kv2(`H refit on ${nRef}`, isFinite(eRef) ? VZ.fmt(eRef, 3) + " px" : "—", VC.good, true);
    kv2("refit is better by", (isFinite(eMin) && isFinite(eRef) && eRef > 0) ? VZ.fmt(eMin / eRef, 2) + "×" : "—", VC.good, true);
    kv2("consensus, re-scored", `${nAfter} / ${NPT}`, nAfter >= st.best.length ? VC.good : VC.a2);
    g.append("text").attr("x", 582).attr("y", 320).attr("font-size", 10).attr("fill", VC.muted).text("transfer error against the map");
    g.append("text").attr("x", 582).attr("y", 334).attr("font-size", 10).attr("fill", VC.muted).text("that generated the inliers.");
    g.append("text").attr("x", 582).attr("y", 362).attr("font-size", 10).attr("fill", VC.muted).text("The minimal-sample H is fitted");
    g.append("text").attr("x", 582).attr("y", 376).attr("font-size", 10).attr("fill", VC.muted).text("to exactly 4 noisy points and");
    g.append("text").attr("x", 582).attr("y", 390).attr("font-size", 10).attr("fill", VC.muted).text("averages nothing. Never keep it.");

    ui.out.innerHTML =
      `${st.n} samples · ${st.degRej} rejected as degenerate before fitting · best consensus <b>${st.best.length}</b> of ${NPT}` +
      ` (true inliers ${NPT - D.nOut}) · adaptive N = <b>${isFinite(st.N) ? st.N : "—"}</b>` +
      (isFinite(eMin) && isFinite(eRef)
        ? `<br />transfer error: minimal-sample H <b style="color:${VC.bad}">${VZ.fmt(eMin, 3)} px</b> → refit on the whole consensus set <b style="color:${VC.good}">${VZ.fmt(eRef, 3)} px</b>, a factor of <b>${VZ.fmt(eMin / eRef, 2)}</b>. Re-scoring against the refitted H recovers <b>${nAfter}</b> inliers where the minimal sample found ${st.best.length} — which is why the refit is repeated once or twice, not applied only at the end.`
        : `<br />draw some samples to start. At ${ui.o.value}% outliers the formula asks for ${wTrue > 0.02 ? Math.ceil(Math.log(0.01) / Math.log(1 - Math.pow(wTrue, 4))) : "—"} samples for a 99% chance of a clean one.`);
  }

  function step() { draw0(); sample(dataset(), +ui.t.value); draw(); }
  function draw0() { if (!st) reset(); }

  ui.step.addEventListener("click", () => { draw0(); sample(dataset(), +ui.t.value); draw(); });
  ui.run.addEventListener("click", () => {
    draw0();
    const D = dataset(), t = +ui.t.value;
    let guard = 0;
    while (st.n < Math.min(st.N, 1200) && guard++ < 1200) sample(D, t);
    draw();
  });
  ui.nw.addEventListener("click", () => { seed = (seed * 1103515245 + 12345) >>> 0; makeData(); reset(); draw(); });
  [ui.o, ui.t, ui.s, ui.deg, ui.ad].forEach(el => {
    el.addEventListener("input", () => { reset(); draw(); });
    el.addEventListener("change", () => { reset(); draw(); });
  });

  makeData(); reset();
  /* start with a short run so the figure is never blank on arrival */
  (function () { const D = dataset(), t = +ui.t.value; let gd = 0; while (st.n < Math.min(st.N, 800) && gd++ < 800) sample(D, t); })();
  draw();
})();

/* ═══ FIG 7 · §10 — influence, weight, and the fit ═══════════════════════════
   A 1-D line fit is enough to make the point and keeps the picture readable.
   The marker area is the IRLS weight, so the down-weighting is not described,
   it is drawn.                                                                */
(function () {
  const svg = d3.select("#mest-svg");
  if (svg.empty()) return;
  const W = 760, H = 430;

  const LOSS = {
    l2:    { rho: (r, a) => r * r, psi: (r, a) => 2 * r,                       w: (r, a) => 2 },
    l1:    { rho: (r, a) => Math.abs(r), psi: (r, a) => Math.sign(r),          w: (r, a) => 1 / Math.max(1e-6, Math.abs(r)) },
    huber: { rho: (r, a) => Math.abs(r) <= a ? 0.5 * r * r : a * Math.abs(r) - 0.5 * a * a,
             psi: (r, a) => Math.abs(r) <= a ? r : a * Math.sign(r),
             w:   (r, a) => Math.abs(r) <= a ? 1 : a / Math.max(1e-6, Math.abs(r)) },
    gm:    { rho: (r, a) => r * r / (1 + r * r / (a * a)),
             psi: (r, a) => 2 * r / Math.pow(1 + r * r / (a * a), 2),
             w:   (r, a) => 2 / Math.pow(1 + r * r / (a * a), 2) },
    tukey: { rho: (r, a) => Math.abs(r) <= a ? (a * a / 6) * (1 - Math.pow(1 - r * r / (a * a), 3)) : a * a / 6,
             psi: (r, a) => Math.abs(r) <= a ? r * Math.pow(1 - r * r / (a * a), 2) : 0,
             w:   (r, a) => Math.abs(r) <= a ? Math.pow(1 - r * r / (a * a), 2) : 0 }
  };
  const NAME = { l2: "L2", l1: "L1", huber: "Huber", gm: "Geman–McClure", tukey: "Tukey biweight" };

  const rn = VZ.rng(1181);
  const N = 20, XS = d3.range(N).map(i => 0.5 + 9 * i / (N - 1));
  const TRUE = { m: 0.55, c: 1.2 };
  const NZ = XS.map(() => VZ.randn(rn) * 0.35);
  const OUTX = [2.2, 7.4, 4.0, 8.6, 1.2, 5.6, 9.2, 3.1];
  const OUTY = [7.6, 7.1, 8.2, 1.0, 8.0, 0.6, 8.4, 8.1];
  let drag = [5.0, 8.2];                   // the draggable one

  const ui = {
    l: document.getElementById("me-l"), a: document.getElementById("me-a"), av: document.getElementById("me-av"),
    i: document.getElementById("me-i"), iv: document.getElementById("me-iv"),
    o: document.getElementById("me-o"), ov: document.getElementById("me-ov"),
    out: document.getElementById("mest-readout")
  };

  function data(nout) {
    const P = XS.map((x, i) => [x, TRUE.m * x + TRUE.c + NZ[i], false]);
    P.push([drag[0], drag[1], true]);
    for (let i = 0; i < nout; i++) P.push([OUTX[i], OUTY[i], true]);
    return P;
  }

  function wls(P, wts) {
    let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    P.forEach((p, i) => { const w = wts ? wts[i] : 1; sw += w; sx += w * p[0]; sy += w * p[1]; sxx += w * p[0] * p[0]; sxy += w * p[0] * p[1]; });
    const den = sw * sxx - sx * sx;
    if (Math.abs(den) < 1e-12) return [0, 0];
    return [(sw * sxy - sx * sy) / den, (sxx * sy - sx * sxy) / den];
  }

  function irls(P, kind, a, passes) {
    let fit = wls(P, null), wts = P.map(() => 1), sig = 1;
    for (let it = 0; it < passes; it++) {
      const r = P.map(p => p[1] - (fit[0] * p[0] + fit[1]));
      const med = d3.median(r.map(Math.abs)) || 1e-3;
      sig = Math.max(1e-3, 1.4826 * med);
      wts = r.map(v => LOSS[kind].w(v / sig, a));
      fit = wls(P, wts);
    }
    return { fit: fit, w: wts, sig: sig };
  }

  function draw() {
    const kind = ui.l.value, a = +ui.a.value, passes = +ui.i.value, nout = +ui.o.value;
    ui.av.textContent = VZ.fmt(a, 2) + " ";
    ui.iv.textContent = passes; ui.ov.textContent = nout;
    /* keep the σ mark on the label without re-templating it every draw */
    ui.av.innerHTML = VZ.fmt(a, 2) + ' <span class="keep">σ</span>';

    const P = data(nout);
    const ls = wls(P, null);
    const R = irls(P, kind, a, passes);

    const f = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = f.g;

    /* ---- left: ρ, ψ, w ----------------------------------------------------- */
    const names = [["ρ(r)  the loss", "rho"], ["ψ(r)  the influence", "psi"], ["w(r)  the IRLS weight", "w"]];
    names.forEach((nm, row) => {
      const y0 = 44 + row * 126;
      VZ.panelBox(g, 14, y0, 320, 104, nm[0]);
      const x = d3.scaleLinear().domain([-6, 6]).range([22, 328]);
      const vals = VZ.linspace(-6, 6, 241).map(r => [r, LOSS[kind][nm[1]](r, a)]);
      const l2v = VZ.linspace(-6, 6, 241).map(r => [r, LOSS.l2[nm[1]](r, a)]);
      const ext = d3.extent(vals.map(d => d[1]).concat(nm[1] === "rho" ? [0] : []));
      const lo = Math.min(ext[0], nm[1] === "w" ? 0 : -Math.abs(ext[1]) * 0.15);
      const y = d3.scaleLinear().domain([lo, Math.max(ext[1], 1e-6) * 1.12]).range([y0 + 98, y0 + 8]);
      g.append("line").attr("x1", 22).attr("x2", 328).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", VC.grid);
      g.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", y0 + 6).attr("y2", y0 + 98).attr("stroke", VC.grid);
      [-a, a].forEach(v => g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", y0 + 6).attr("y2", y0 + 98)
        .attr("stroke", VC.a2).attr("stroke-dasharray", "2 3").attr("stroke-opacity", 0.7));
      if (kind !== "l2") {
        const cl = l2v.filter(d => y(d[1]) >= y0 + 4 && y(d[1]) <= y0 + 100);
        if (cl.length > 1) VZ.poly(g, cl.map(d => [x(d[0]), y(d[1])]), { stroke: VC.muted, w: 1, close: false, dash: "3 3" });
      }
      VZ.poly(g, vals.map(d => [x(d[0]), y(VZ.clamp(d[1], y.domain()[0], y.domain()[1]))]), { stroke: VC.accent, w: 2, close: false });
      g.append("text").attr("x", x(a) + 3).attr("y", y0 + 18).attr("font-size", 9).attr("fill", VC.a2).text("a");
    });
    g.append("text").attr("x", 14).attr("y", 424).attr("font-size", 10).attr("fill", VC.muted)
      .text("dashed grey = least squares, for comparison · r in units of the robust scale σ̂");

    /* ---- right: the fit ---------------------------------------------------- */
    VZ.panelBox(g, 360, 44, 386, 330, `${NAME[kind]} against least squares · marker area = IRLS weight`);
    const x = d3.scaleLinear().domain([0, 10]).range([374, 736]);
    const y = d3.scaleLinear().domain([-0.5, 9.5]).range([360, 58]);
    const gp = g.append("g");
    [0, 2, 4, 6, 8].forEach(v => {
      gp.append("line").attr("x1", 374).attr("x2", 736).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", VC.grid);
      gp.append("line").attr("y1", 58).attr("y2", 360).attr("x1", x(v)).attr("x2", x(v)).attr("stroke", VC.grid);
    });
    const line = (fit, col, dash) => VZ.poly(gp, [[x(0), y(fit[1])], [x(10), y(fit[0] * 10 + fit[1])]],
      { stroke: col, w: 2, close: false, dash: dash });
    line([TRUE.m, TRUE.c], VC.muted, "4 4");
    line(ls, VC.bad, null);
    line(R.fit, VC.good, null);
    const wmax = d3.max(R.w) || 1;
    P.forEach((p, i) => {
      const rad = 2 + 5 * Math.sqrt(Math.max(0, R.w[i]) / wmax);
      gp.append("circle").attr("cx", x(p[0])).attr("cy", y(p[1])).attr("r", rad)
        .attr("fill", p[2] ? VC.bad : VC.accent).attr("fill-opacity", 0.55)
        .attr("stroke", p[2] ? VC.bad : VC.accent).attr("stroke-width", 1);
    });
    VZ.legend(gp, [{ color: VC.muted, label: "the true line", dash: "4 4" },
                   { color: VC.bad, label: "least squares" },
                   { color: VC.good, label: NAME[kind] + " · IRLS" }], 384, 76, { gap: 14 });

    const gd = gp.append("g");
    AM.drag(gd, [drag], p => [x(p[0]), y(p[1])], q => [VZ.clamp(x.invert(q[0]), 0, 10), VZ.clamp(y.invert(q[1]), -0.5, 9.5)],
      () => draw(), { r: 7, fill: "none", stroke: VC.a2 });
    gd.selectAll("circle.dragpt").attr("stroke-width", 2);

    const wDrag = R.w[N];
    ui.out.innerHTML =
      `true line y = 0.550x + 1.200 · least squares <b style="color:${VC.bad}">y = ${VZ.fmt(ls[0], 3)}x + ${VZ.fmt(ls[1], 3)}</b>` +
      ` · ${NAME[kind]} after ${passes} IRLS passes <b style="color:${VC.good}">y = ${VZ.fmt(R.fit[0], 3)}x + ${VZ.fmt(R.fit[1], 3)}</b>` +
      `<br />robust scale <span class="keep">σ</span>̂ = 1.4826 · median|r| = ${VZ.fmt(R.sig, 4)} · the dragged outlier now carries weight <b>${VZ.fmt(wDrag / wmax, 4)}</b> of the maximum` +
      (kind === "l2" ? " — which is to say, full weight, because least squares has no down-weighting mechanism at all."
        : (wDrag / wmax < 0.02 ? " — effectively excluded, without any hard threshold ever being applied."
          : " — reduced but still heard; lower the tuning constant to silence it further."));
  }

  [ui.l, ui.a, ui.i, ui.o].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ═══ FIG 8 · §11 — the plane-induced homography ═════════════════════════════
   The 3-D panel is only there to make the geometry legible; the claim being
   tested is the number in the readout, which is the largest discrepancy between
   projecting a point directly and transferring it through H. It never leaves
   the 10⁻¹³ range, whatever the controls are set to.                          */
(function () {
  const svg = d3.select("#plane-svg");
  if (svg.empty()) return;
  const W = 760, H = 430, f = 600, cx = 320, cy = 240, k = 0.30;

  const ui = {
    b: document.getElementById("pl-b"), bv: document.getElementById("pl-bv"),
    r: document.getElementById("pl-r"), rv: document.getElementById("pl-rv"),
    t: document.getElementById("pl-t"), tv: document.getElementById("pl-tv"),
    d: document.getElementById("pl-d"), dv: document.getElementById("pl-dv"),
    out: document.getElementById("plane-readout")
  };

  function draw() {
    const b = +ui.b.value / 100, rot = VZ.rad(+ui.r.value), tilt = VZ.rad(+ui.t.value), dpl = +ui.d.value;
    ui.bv.textContent = VZ.fmt(b, 2) + " m"; ui.rv.textContent = ui.r.value + "°";
    ui.tv.textContent = ui.t.value + "°"; ui.dv.textContent = VZ.fmt(dpl, 1) + " m";

    const K = [[f, 0, cx], [0, f, cy], [0, 0, 1]], Ki = VZ.inv3(K);
    const R = VZ.mul(VZ.Ry(rot), VZ.Rz(VZ.rad(3)));
    const t = [-b, 0, 0.3 * b];
    /* The plane is tilted about the y axis but always passes through (0, 0, d) on the
       optical axis, so tilting does not swing it out of the field of view. Its
       perpendicular distance from camera 0 — the d in the formula — is d·cos(tilt). */
    const nrm = VZ.unit([Math.sin(tilt), 0, Math.cos(tilt)]);
    const ctr = [0, 0, dpl], dperp = VZ.dot(nrm, ctr);
    const Ht = VZ.mul(VZ.mul(K, R.map((row, i) => row.map((v, j) => v + t[i] * nrm[j] / dperp))), Ki);

    /* points ON the plane, laid out on a grid in the plane's own frame */
    const e1 = VZ.unit(VZ.cross(nrm, [0, 1, 0])), e2 = VZ.cross(nrm, e1);
    const PTS = [];
    for (let i = -3; i <= 3; i++) for (let j = -2; j <= 2; j++) {
      const u = i * 0.16 * dpl, v = j * 0.13 * dpl;
      PTS.push(VZ.add(ctr, VZ.add(VZ.scale(e1, u), VZ.scale(e2, v))));
    }
    const prj = (Rc, tc, P) => {
      const X = VZ.add(VZ.mv(Rc, P), tc);
      return (X[2] > 0.05) ? [f * X[0] / X[2] + cx, f * X[1] / X[2] + cy, X[2]] : null;
    };
    const im0 = PTS.map(P => prj(VZ.eye(3), [0, 0, 0], P));
    const im1 = PTS.map(P => prj(R, t, P));

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;

    /* ---- 3-D panel --------------------------------------------------------- */
    VZ.panelBox(g, 14, 40, 246, 296, "the scene");
    const v3 = VZ.view3({ target: [0, 0, dpl * 0.55], yaw: VZ.rad(-118), pitch: VZ.rad(20), dist: dpl * 2.1, f: 300, cx: 137, cy: 188 });
    const g3 = g.append("g").attr("clip-path", VZ.clip(svg, "am-pl-clip", 14, 40, 246, 296));
    /* the plane, as a quad */
    const quad = [[-0.58, -0.42], [0.58, -0.42], [0.58, 0.42], [-0.58, 0.42]].map(([u, v]) =>
      VZ.add(ctr, VZ.add(VZ.scale(e1, u * dpl), VZ.scale(e2, v * dpl))));
    VZ.poly(g3, quad.map(p => { const q = v3.project(p); return [q.x, q.y]; }),
      { stroke: VC.accent, w: 1.4, fill: VC.accent, fillOp: 0.12 });
    PTS.forEach(P => { const q = v3.project(P); g3.append("circle").attr("cx", q.x).attr("cy", q.y).attr("r", 1.8).attr("fill", VC.a2); });
    /* the two camera centres, with a stub of their optical axes */
    const C0 = [0, 0, 0], Rt = VZ.T(R), C1 = VZ.neg(VZ.mv(Rt, t));
    [[C0, VZ.eye(3), VC.good, "cam 0"], [C1, Rt, VC.violet, "cam 1"]].forEach(([C, Rr, col, lab]) => {
      const a = v3.project(C), ax = v3.project(VZ.add(C, VZ.mv(Rr, [0, 0, dpl * 0.35])));
      g3.append("line").attr("x1", a.x).attr("y1", a.y).attr("x2", ax.x).attr("y2", ax.y).attr("stroke", col).attr("stroke-width", 1.4);
      g3.append("circle").attr("cx", a.x).attr("cy", a.y).attr("r", 4).attr("fill", col);
      g3.append("text").attr("x", a.x + 6).attr("y", a.y + 12).attr("font-size", 9.5).attr("fill", col).text(lab);
    });
    /* a few rays, to show that a pixel is a ray and the plane picks the depth */
    [0, 6, 17, 28, 34].forEach(i => {
      const a = v3.project(C0), q = v3.project(PTS[i]);
      g3.append("line").attr("x1", a.x).attr("y1", a.y).attr("x2", q.x).attr("y2", q.y)
        .attr("stroke", VC.good).attr("stroke-width", 0.7).attr("stroke-opacity", 0.4);
    });

    /* ---- the two images ---------------------------------------------------- */
    let maxErr = 0, nvis = 0;
    [[im0, 282, "image 0"], [im1, 492, "image 1 · ○ direct   ● via H"]].forEach(([im, x0, lab]) => {
      VZ.panelBox(g, x0, 40, 640 * k, 480 * k, lab);
      im.forEach((p, i) => {
        if (!p) return;
        const sx = x0 + p[0] * k, sy = 40 + p[1] * k;
        if (p[0] < -80 || p[0] > 720 || p[1] < -60 || p[1] > 540) return;
        g.append("circle").attr("cx", sx).attr("cy", sy).attr("r", 2.6)
          .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.2);
      });
    });
    im0.forEach((p, i) => {
      if (!p || !im1[i]) return;
      const q = VZ.applyH(Ht, [p[0], p[1]]);
      if (!q) return;
      const e = Math.hypot(q[0] - im1[i][0], q[1] - im1[i][1]);
      if (im1[i][0] > -80 && im1[i][0] < 720) { maxErr = Math.max(maxErr, e); nvis++; }
      g.append("circle").attr("cx", 492 + q[0] * k).attr("cy", 40 + q[1] * k).attr("r", 1.6).attr("fill", VC.good);
    });

    /* ---- the matrix, decomposed ------------------------------------------- */
    VZ.panelBox(g, 282, 196, 464, 140, "H₁₀ = K₁ ( R + t n̂ᵀ / d ) K₀⁻¹");
    const inner = R.map((row, i) => row.map((v, j) => v + t[i] * nrm[j] / dpl));
    VZ.matText(g, R, 296, 244, { dp: 4, pad: 8, size: 10, label: "R" });
    VZ.matText(g, [[t[0] * nrm[0] / dperp, t[0] * nrm[1] / dperp, t[0] * nrm[2] / dperp],
                   [t[1] * nrm[0] / dperp, t[1] * nrm[1] / dperp, t[1] * nrm[2] / dperp],
                   [t[2] * nrm[0] / dperp, t[2] * nrm[1] / dperp, t[2] * nrm[2] / dperp]],
      420, 244, { dp: 5, pad: 9, size: 10, label: "t n̂ᵀ / d   (rank 1)" });
    VZ.matText(g, Ht.map(r => r.map(v => v / Ht[2][2])), 578, 244, { dp: 4, pad: 9, size: 10, label: "H₁₀, h₂₂ = 1" });
    g.append("text").attr("x", 296).attr("y", 314).attr("font-size", 10).attr("fill", VC.muted)
      .text("the rank-1 term is the only place the plane enters; set t = 0 and it vanishes (§12).");

    ui.out.innerHTML =
      (nvis === 0
        ? `<b>no plane point is visible in both images at this setting</b>`
        : `<b>max transfer error over ${nvis} points on the plane: ${maxErr === 0 ? "0" : maxErr.toExponential(2)} px</b>`) +
      ` — baseline ${VZ.fmt(b, 2)} m, rotation ${ui.r.value}°, plane tilt ${ui.t.value}° (perpendicular distance ${VZ.fmt(dperp, 2)} m).` +
      `<br />Move every control to its extreme and this number does not budge out of the 10⁻¹³ range. A homography relates two views of a <i>plane</i> exactly, for <i>any</i> camera motion. That is a theorem, and §13 shows what happens the moment a point leaves the plane.`;
  }

  [ui.b, ui.r, ui.t, ui.d].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ═══ FIG 9 · §12 — the nodal point ══════════════════════════════════════════
   Pure rotation is exact; "pure" rotation is not. The error is drawn on the
   image and then plotted against depth, where it is a straight line of slope
   −1 sitting just above the prediction f·b/Z.                                 */
(function () {
  const svg = d3.select("#rot-svg");
  if (svg.empty()) return;
  const W = 760, H = 440, cx = 320, cy = 240, k = 0.55, NP = 2500, NDRAW = 520;

  const rn = VZ.rng(2024);
  const RAW = d3.range(NP).map(() => [rn(), rn(), rn()]);

  const ui = {
    r: document.getElementById("ro-r"), rv: document.getElementById("ro-rv"),
    o: document.getElementById("ro-o"), ov: document.getElementById("ro-ov"),
    f: document.getElementById("ro-f"), fv: document.getElementById("ro-fv"),
    z: document.getElementById("ro-z"), zv: document.getElementById("ro-zv"),
    out: document.getElementById("rot-readout")
  };

  function draw() {
    const th = VZ.rad(+ui.r.value), b = +ui.o.value / 100, f = +ui.f.value, znear = +ui.z.value;
    ui.rv.textContent = ui.r.value + "°"; ui.ov.textContent = ui.o.value + " cm";
    ui.fv.textContent = f + " px"; ui.zv.textContent = VZ.fmt(znear, 1) + " m";

    const K = [[f, 0, cx], [0, f, cy], [0, 0, 1]], Ki = VZ.inv3(K);
    const R = VZ.Ry(th), Hr = VZ.mul(VZ.mul(K, R), Ki);
    const C = [b, 0, 0], t = VZ.neg(VZ.mv(R, C));
    const zfar = 60;

    const prj = (Rc, tc, P) => {
      const X = VZ.add(VZ.mv(Rc, P), tc);
      return (X[2] > 0.05) ? [f * X[0] / X[2] + cx, f * X[1] / X[2] + cy] : null;
    };

    const rows = [];
    for (let i = 0; i < NP; i++) {
      const Z = znear * Math.pow(zfar / znear, RAW[i][2]);
      const P = [(RAW[i][0] - 0.5) * 2.4 * Z, (RAW[i][1] - 0.5) * 1.6 * Z, Z];
      const a = prj(VZ.eye(3), [0, 0, 0], P), c = prj(R, t, P);
      if (!a || !c) continue;
      if (a[0] < 0 || a[0] > 640 || a[1] < 0 || a[1] > 480) continue;
      if (c[0] < -60 || c[0] > 700 || c[1] < -40 || c[1] > 520) continue;
      const q = VZ.applyH(Hr, a);
      if (!q) continue;
      rows.push({ Z: Z, p: q, m: c, e: Math.hypot(q[0] - c[0], q[1] - c[1]) });
    }

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    const col = d3.scaleSequential(d3.interpolateViridis).domain([Math.log(zfar), Math.log(znear)]);

    VZ.panelBox(g, 14, 44, 640 * k, 480 * k, "image 1 · predicted by K R K⁻¹ → where the point actually lands");
    const gI = g.append("g").attr("clip-path", VZ.clip(svg, "am-rot-clip", 14, 44, 640 * k, 480 * k));
    const step = Math.max(1, Math.ceil(rows.length / NDRAW));
    rows.forEach((d, di) => {
      if (di % step) return;
      const a = [14 + d.p[0] * k, 44 + d.p[1] * k], c = [14 + d.m[0] * k, 44 + d.m[1] * k];
      gI.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", c[0]).attr("y2", c[1])
        .attr("stroke", col(Math.log(d.Z))).attr("stroke-width", 1.1).attr("stroke-opacity", 0.85);
      gI.append("circle").attr("cx", a[0]).attr("cy", a[1]).attr("r", 1.3).attr("fill", col(Math.log(d.Z))).attr("fill-opacity", 0.7);
    });
    /* a depth colour key */
    [znear, 3, 10, 30, zfar].filter(z => z >= znear && z <= zfar).forEach((z, i) => {
      g.append("rect").attr("x", 20 + i * 62).attr("y", 322).attr("width", 12).attr("height", 9).attr("fill", col(Math.log(z)));
      g.append("text").attr("x", 36 + i * 62).attr("y", 330).attr("font-size", 9.5).attr("fill", VC.muted).text(VZ.fmt(z, 0) + " m");
    });

    /* ---- error against depth ---------------------------------------------- */
    VZ.panelBox(g, 410, 44, 336, 300, "transfer error vs depth · log–log");
    const x = d3.scaleLog().domain([Math.max(0.4, znear * 0.85), zfar * 1.15]).range([440, 736]);
    const y = d3.scaleLog().domain([0.005, 600]).range([320, 66]).clamp(true);
    const gp = g.append("g");
    [0.01, 0.1, 1, 10, 100].forEach(v => {
      gp.append("line").attr("x1", 440).attr("x2", 736).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", VC.grid);
      gp.append("text").attr("x", 436).attr("y", y(v) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted)
        .text(v >= 1 ? v : v.toString());
    });
    [1, 3, 10, 30].filter(v => v >= x.domain()[0] && v <= x.domain()[1]).forEach(v => {
      gp.append("line").attr("y1", 66).attr("y2", 320).attr("x1", x(v)).attr("x2", x(v)).attr("stroke", VC.grid);
      gp.append("text").attr("x", x(v)).attr("y", 334).attr("text-anchor", "middle").attr("font-size", 8.5).attr("fill", VC.muted).text(v + " m");
    });
    gp.append("text").attr("x", 736).attr("y", 348).attr("text-anchor", "end").attr("font-size", 10).attr("fill", VC.muted).text("depth Z →");
    gp.append("text").attr("x", 440).attr("y", 60).attr("font-size", 10).attr("fill", VC.muted).text("error, pixels");
    rows.forEach((d, di) => { if (di % step) return;
      gp.append("circle").attr("cx", x(VZ.clamp(d.Z, x.domain()[0], x.domain()[1])))
        .attr("cy", y(Math.max(d.e, 0.005))).attr("r", 1.7).attr("fill", col(Math.log(d.Z))).attr("fill-opacity", 0.8); });
    if (b > 0) {
      const pred = VZ.linspace(Math.log(x.domain()[0]), Math.log(x.domain()[1]), 40)
        .map(u => { const z = Math.exp(u); return [x(z), y(VZ.clamp(f * b / z, 0.005, 600))]; });
      VZ.poly(gp, pred, { stroke: VC.a2, w: 1.6, close: false, dash: "5 4" });
      gp.append("text").attr("x", 446).attr("y", y(VZ.clamp(f * b / x.domain()[0], 0.008, 500)) - 6)
        .attr("font-size", 10).attr("fill", VC.a2).text("f · b / Z");
    }

    /* the measured error in a narrow band around the nearest depth, and around 10 m */
    const band = z0 => {
      const s = rows.filter(d => Math.abs(Math.log(d.Z / z0)) < 0.12);
      return s.length ? d3.mean(s, d => d.e) : NaN;
    };
    const eNear = band(znear), eTen = band(10);
    const mx = rows.length ? d3.max(rows, d => d.e) : 0;

    ui.out.innerHTML =
      `f = ${f} px, ${ui.r.value}° pan, rotation centre offset by <b>${ui.o.value} cm</b>.` +
      ` Measured over ${rows.length} visible points: <b>${isFinite(eNear) ? VZ.fmt(eNear, 2) : "—"} px</b> of error at ${VZ.fmt(znear, 1)} m` +
      (isFinite(eTen) ? `, <b>${VZ.fmt(eTen, 2)} px</b> at 10 m` : "") +
      `, worst point ${VZ.fmt(mx, 1)} px.` +
      `<br />` + (b === 0
        ? "At zero offset the rotation is genuinely about the optical centre and every error is machine zero, at every depth — that is §12's theorem."
        : `The prediction f·b/Z gives ${VZ.fmt(f * b / znear, 2)} px at ${VZ.fmt(znear, 1)} m; the measurement is ${VZ.fmt(Math.abs(100 * (eNear / (f * b / znear) - 1)), 0)}% ${eNear >= f * b / znear ? "above" : "below"} it, the difference coming from the perspective divide and from where in the frame the points happen to fall. Slope −1 on this plot means the error is inversely proportional to depth: distant scenery stitches perfectly while the foreground ghosts, which is exactly what a bad panorama looks like.`);
  }

  [ui.r, ui.o, ui.f, ui.z].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ═══ FIG 10 · §13 — plane plus parallax ═════════════════════════════════════
   The camera moves sideways, so the epipole sits far off-frame and the residual
   field is very nearly a uniform direction with a magnitude that encodes depth.
   That makes the V against inverse depth clean, which is the point: the residual
   is not noise, it is a measurement of the scene.                              */
(function () {
  const svg = d3.select("#parallax-svg");
  if (svg.empty()) return;
  const W = 760, H = 470, f = 600, cx = 320, cy = 240, k = 0.65, NP = 260;

  const rn = VZ.rng(880123);
  const RAW = d3.range(NP).map(() => [rn(), rn(), rn(), rn()]);
  const R = VZ.mul(VZ.Ry(VZ.rad(8)), VZ.Rz(VZ.rad(3)));

  const ui = {
    b: document.getElementById("px-b"), bv: document.getElementById("px-bv"),
    d: document.getElementById("px-d"), dv: document.getElementById("px-dv"),
    s: document.getElementById("px-s"), ff: document.getElementById("px-f"),
    out: document.getElementById("parallax-readout")
  };

  function draw() {
    const b = +ui.b.value / 100, zpi = +ui.d.value, scene = ui.s.value, mode = ui.ff.value;
    ui.bv.textContent = VZ.fmt(b, 2) + " m"; ui.dv.textContent = VZ.fmt(zpi, 1) + " m";

    const K = [[f, 0, cx], [0, f, cy], [0, 0, 1]], Ki = VZ.inv3(K);
    const t = [-b, 0, 0.30 * b], nrm = [0, 0, 1];
    const Hpi = VZ.mul(VZ.mul(K, R.map((row, i) => row.map((v, j) => v + t[i] * nrm[j] / zpi))), Ki);

    const depth = i => {
      const u = RAW[i][2];
      if (scene === "plane") return zpi;
      if (scene === "far") return 60 * Math.pow(10, u);
      if (scene === "two") return (RAW[i][3] < 0.5) ? 5 : 20;
      return 3 * Math.pow(10, u);                        // 3 … 30 m
    };
    const prj = (Rc, tc, P) => {
      const X = VZ.add(VZ.mv(Rc, P), tc);
      return (X[2] > 0.05) ? [f * X[0] / X[2] + cx, f * X[1] / X[2] + cy] : null;
    };
    const src = [], dst = [], Zs = [];
    for (let i = 0; i < NP; i++) {
      const Z = depth(i);
      const P = [(RAW[i][0] - 0.5) * 0.95 * Z, (RAW[i][1] - 0.5) * 0.72 * Z, Z];
      const a = prj(VZ.eye(3), [0, 0, 0], P), c = prj(R, t, P);
      if (!a || !c) continue;
      if (a[0] < 4 || a[0] > 636 || a[1] < 4 || a[1] > 476) continue;
      src.push(a); dst.push(c); Zs.push(Z);
    }

    let Hf = Hpi;
    if (mode === "ls" && src.length >= 6) {
      const d0 = AM.dlt(src, dst, true);
      if (d0) Hf = AM.refineH(d0.H, src, dst, { iters: 8, sym: false }).H;
    }
    const eh = VZ.mv(K, t);
    const epi = (Math.abs(eh[2]) > 1e-9) ? [eh[0] / eh[2], eh[1] / eh[2]] : null;

    const rows = [];
    let maxAng = 0;
    for (let i = 0; i < src.length; i++) {
      const q = VZ.applyH(Hf, src[i]);
      if (!q) continue;
      const r = [dst[i][0] - q[0], dst[i][1] - q[1]], mag = Math.hypot(r[0], r[1]);
      if (epi && mag > 1e-6) {
        const w = [epi[0] - q[0], epi[1] - q[1]], wn = Math.hypot(w[0], w[1]);
        if (wn > 1e-9) {
          const c = Math.abs((r[0] * w[0] + r[1] * w[1]) / (mag * wn));
          maxAng = Math.max(maxAng, VZ.deg(Math.acos(VZ.clamp(c, -1, 1))));
        }
      }
      rows.push({ q: q, m: dst[i], Z: Zs[i], e: mag, iz: 1 / Zs[i] });
    }

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    const zlo = d3.min(rows, d => d.Z) || 1, zhi = d3.max(rows, d => d.Z) || 10;
    const col = d3.scaleSequential(d3.interpolatePlasma).domain([Math.log(zhi), Math.log(zlo)]);

    VZ.panelBox(g, 14, 44, 640 * k, 480 * k, "image 2 · from where H puts the point → to where it really is");
    const gI = g.append("g").attr("clip-path", VZ.clip(svg, "am-px-clip", 14, 44, 640 * k, 480 * k));
    rows.forEach(d => {
      const a = [14 + d.q[0] * k, 44 + d.q[1] * k], c = [14 + d.m[0] * k, 44 + d.m[1] * k];
      if (d.e * k > 3) VZ.arrow(gI, a[0], a[1], c[0], c[1], { color: col(Math.log(d.Z)), w: 1.2, head: 4, op: 0.9 });
      else {
        gI.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", c[0]).attr("y2", c[1])
          .attr("stroke", col(Math.log(d.Z))).attr("stroke-width", 1.2);
      }
      gI.append("circle").attr("cx", a[0]).attr("cy", a[1]).attr("r", 1.5).attr("fill", col(Math.log(d.Z))).attr("fill-opacity", 0.75);
    });
    if (epi) {
      const inside = epi[0] > 0 && epi[0] < 640 && epi[1] > 0 && epi[1] < 480;
      if (inside) {
        const e = [14 + epi[0] * k, 44 + epi[1] * k];
        gI.append("path").attr("d", `M${e[0] - 9},${e[1]} H${e[0] + 9} M${e[0]},${e[1] - 9} V${e[1] + 9}`)
          .attr("stroke", VC.a2).attr("stroke-width", 2);
      } else {
        const ey = VZ.clamp(44 + epi[1] * k, 50, 44 + 480 * k - 6);
        const ex = epi[0] < 0 ? 18 : 14 + 640 * k - 6;
        VZ.arrow(gI, ex + (epi[0] < 0 ? 26 : -26), ey, ex, ey, { color: VC.a2, w: 2, head: 7 });
        g.append("text").attr("x", 18).attr("y", 44 + 480 * k - 8).attr("font-size", 10).attr("fill", VC.a2)
          .text(`epipole off-frame at (${Math.round(epi[0])}, ${Math.round(epi[1])})`);
      }
    }
    [zlo, zpi, zhi].forEach((z, i) => {
      g.append("rect").attr("x", 20 + i * 104).attr("y", 384).attr("width", 12).attr("height", 9).attr("fill", col(Math.log(z)));
      g.append("text").attr("x", 36 + i * 104).attr("y", 392).attr("font-size", 9.5).attr("fill", VC.muted)
        .text(VZ.fmt(z, z < 10 ? 1 : 0) + " m" + (i === 1 ? "  (the plane)" : ""));
    });

    /* ---- residual vs inverse depth ---------------------------------------- */
    VZ.panelBox(g, 456, 44, 290, 250, "‖residual‖ against 1/Z");
    const izmax = 1 / zlo * 1.08, emax = Math.max(2, d3.max(rows, d => d.e) || 2) * 1.1;
    const x = d3.scaleLinear().domain([0, izmax]).range([472, 736]);
    const y = d3.scaleLinear().domain([0, emax]).range([272, 66]);
    const gp = g.append("g");
    y.ticks(4).forEach(v => {
      gp.append("line").attr("x1", 472).attr("x2", 736).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", VC.grid);
      gp.append("text").attr("x", 468).attr("y", y(v) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted).text(VZ.fmt(v, 0));
    });
    gp.append("line").attr("x1", x(1 / zpi)).attr("x2", x(1 / zpi)).attr("y1", 66).attr("y2", 272)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "4 3");
    gp.append("text").attr("x", x(1 / zpi) + 4).attr("y", 76).attr("font-size", 9.5).attr("fill", VC.a2).text("1/Z" + "π");
    rows.forEach(d => gp.append("circle").attr("cx", x(VZ.clamp(d.iz, 0, izmax))).attr("cy", y(VZ.clamp(d.e, 0, emax)))
      .attr("r", 2).attr("fill", col(Math.log(d.Z))).attr("fill-opacity", 0.85));
    /* slope through the origin of (|1/Z − 1/Zπ|, ‖r‖) */
    let sxy = 0, sxx = 0;
    rows.forEach(d => { const u = Math.abs(d.iz - 1 / zpi); sxy += u * d.e; sxx += u * u; });
    const slope = sxx > 1e-14 ? sxy / sxx : 0;
    if (slope > 0) {
      [[0, 1 / zpi], [1 / zpi, izmax]].forEach(([a, c]) => VZ.poly(gp,
        [[x(a), y(VZ.clamp(slope * Math.abs(a - 1 / zpi), 0, emax))], [x(c), y(VZ.clamp(slope * Math.abs(c - 1 / zpi), 0, emax))]],
        { stroke: VC.ink, w: 1.2, close: false, dash: "4 3" }));
    }
    gp.append("text").attr("x", 736).attr("y", 288).attr("text-anchor", "end").attr("font-size", 10).attr("fill", VC.muted).text("1 / Z  (m⁻¹) →");

    const ft = VZ.norm(t) * f;
    const kv = VZ.kv(g, 470, 322, { keyW: 200, size: 11, lead: 16 });
    kv("fitted slope  ‖r‖ / |1/Z − 1/Zπ|", VZ.fmt(slope, 1) + " px·m", VC.ink, true);
    kv("predicted  f · ‖t‖", VZ.fmt(ft, 1) + " px·m", VC.a2);
    kv("max angle to the epipolar line", maxAng < 1e-9 ? "—" : maxAng.toExponential(1) + "°", VC.good, true);
    kv("largest residual", VZ.fmt(d3.max(rows, d => d.e) || 0, 2) + " px", VC.bad);
    kv("points", String(rows.length));

    ui.out.innerHTML =
      (scene === "plane"
        ? `Every point is on the reference plane, so every residual is <b>${(d3.max(rows, d => d.e) || 0).toExponential(2)} px</b>. This is §11's theorem, drawn.`
        : `Largest residual <b style="color:${VC.bad}">${VZ.fmt(d3.max(rows, d => d.e) || 0, 2)} px</b> at ${VZ.fmt(d3.min(rows, d => d.Z), 1)} m, zero at the reference plane, and rising again beyond it. ` +
          `Slope ${VZ.fmt(slope, 1)} px·m against the prediction f·‖t‖ = ${VZ.fmt(ft, 1)} px·m` +
          (maxAng > 1e-9 ? `; every residual lies on the line to the epipole to within <b>${maxAng.toExponential(1)}°</b>.` : ".")) +
      `<br />` + (mode === "ls"
        ? "Fitting H by least squares to <i>all</i> the points does not remove the parallax — it only redistributes it, moving the zero-residual surface to a compromise depth. There is no homography that fits a non-planar scene from two centres; least squares finds the best of a set that does not contain the answer."
        : (b === 0
          ? "At zero baseline the two centres coincide and the residual is machine zero at every depth — §12."
          : "The zero-residual surface sits exactly at the reference plane. Slide the plane and it follows, which is the clearest possible demonstration that a homography encodes one depth and one only."));
  }

  [ui.b, ui.d, ui.s, ui.ff].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ═══ FIG 11 · §15 — compositing surfaces ════════════════════════════════════
   The flat canvas's width is 2f·tan(FOV/2) and diverges at 180°; the cylinder's
   is f·FOV and does not. The tilt control then measures the price of the
   "pure translation" shortcut when the camera is not level.                    */
(function () {
  const svg = d3.select("#surf-svg");
  if (svg.empty()) return;
  const W = 760, H = 450, IW = 640, IH = 480, cx = 320, cy = 240, NF = 5;

  const ui = {
    s: document.getElementById("sf-s"), fov: document.getElementById("sf-fov"), fovv: document.getElementById("sf-fovv"),
    f: document.getElementById("sf-f"), fv: document.getElementById("sf-fv"),
    t: document.getElementById("sf-t"), tv: document.getElementById("sf-tv"),
    out: document.getElementById("surf-readout")
  };

  /* a world direction, expressed in the camera's own frame, onto the canvas */
  function toSurf(kind, c, f) {
    const r = Math.hypot(c[0], c[2]);
    if (kind === "flat") return (c[2] > 1e-3) ? [f * c[0] / c[2], f * c[1] / c[2]] : null;
    if (kind === "cyl") return [f * Math.atan2(c[0], c[2]), r > 1e-9 ? f * c[1] / r : 0];
    return [f * Math.atan2(c[0], c[2]), f * Math.atan2(c[1], r)];
  }
  const canvasW = (kind, fovDeg, f) =>
    kind === "flat" ? (fovDeg >= 178 ? Infinity : 2 * f * Math.tan(VZ.rad(fovDeg) / 2)) : f * VZ.rad(fovDeg);

  function draw() {
    const kind = ui.s.value, fov = +ui.fov.value, f = +ui.f.value, tilt = VZ.rad(+ui.t.value);
    ui.fovv.textContent = fov + "°"; ui.fv.textContent = f + " px"; ui.tv.textContent = ui.t.value + "°";

    const perFrame = 2 * VZ.deg(Math.atan(cx / f));
    const span = Math.max(0, fov - perFrame);
    const yaws = d3.range(NF).map(i => VZ.rad(-span / 2 + span * (NF === 1 ? 0.5 : i / (NF - 1))));
    /* world → camera k: the camera pans about the WORLD vertical and is then tilted
       about its OWN x axis, so R_k = Rx(tilt)·Ry(yaw_k). Writing it the other way round
       makes the tilt a rotation of the world, which cancels between frames and hides
       the very effect this control exists to show. */
    const Rt = VZ.Rx(tilt);
    const cams = yaws.map(a => VZ.mul(Rt, VZ.Ry(a)));
    const camsInv = cams.map(VZ.T);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;

    /* ---- the canvas -------------------------------------------------------- */
    const cw = canvasW(kind, fov, f);
    VZ.panelBox(g, 14, 44, 732, 196,
      `${kind === "flat" ? "flat" : kind === "cyl" ? "cylindrical" : "spherical"} canvas · ` +
      (isFinite(cw) ? `${VZ.fmt(cw / f, 3)} f wide (${Math.round(cw)} px at f = ${f})` : "unbounded"));
    const half = isFinite(cw) ? cw / 2 : 40 * f;
    const sx = d3.scaleLinear().domain([-half, half]).range([20, 740]);
    const scaleXY = (sx(1) - sx(0));
    const sy = v => 142 + v * scaleXY;
    const gc = g.append("g").attr("clip-path", VZ.clip(svg, "am-sf-clip", 14, 44, 732, 196));
    const cols = [VC.accent, VC.a2, VC.good, VC.violet, VC.teal];

    let anyOut = false;
    cams.forEach((Rc, kf) => {
      const Rci = camsInv[kf];
      /* frame outline: walk the sensor border, take each pixel to a world ray,
         then straight back into the camera's own frame — which is the identity —
         so the outline is simply the sensor border mapped by the surface, placed
         at this frame's yaw. Do it via the world so the tilt participates. */
      const bd = [];
      const border = [];
      for (let u = 0; u <= IW; u += 32) border.push([u, 0]);
      for (let v = 0; v <= IH; v += 32) border.push([IW, v]);
      for (let u = IW; u >= 0; u -= 32) border.push([u, IH]);
      for (let v = IH; v >= 0; v -= 32) border.push([0, v]);
      border.forEach(([u, v]) => {
        const dcam = [u - cx, v - cy, f];
        const dworld = VZ.mv(Rci, dcam);                 // camera → world
        const q = toSurf(kind, dworld, f);
        if (!q) { anyOut = true; return; }
        bd.push([sx(q[0]), sy(q[1])]);
      });
      if (bd.length > 3) VZ.poly(gc, bd, { stroke: cols[kf % 5], w: 1.4, fill: cols[kf % 5], fillOp: 0.07 });
      /* a grid of world-vertical lines seen through this frame, to show line bending */
      for (let u = 80; u < IW; u += 160) {
        const pts = [];
        for (let v = 0; v <= IH; v += 20) {
          const q = toSurf(kind, VZ.mv(Rci, [u - cx, v - cy, f]), f);
          if (q) pts.push([sx(q[0]), sy(q[1])]);
        }
        if (pts.length > 1) VZ.poly(gc, pts, { stroke: cols[kf % 5], w: 0.8, close: false, op: 0.55 });
      }
    });

    /* ---- canvas width vs field of view ------------------------------------- */
    VZ.panelBox(g, 14, 268, 348, 162, "canvas width, in units of f");
    const fx = d3.scaleLinear().domain([40, 330]).range([34, 352]);
    const fy = d3.scaleLog().domain([0.5, 60]).range([416, 288]).clamp(true);
    const gw = g.append("g");
    [1, 3, 10, 30].forEach(v => {
      gw.append("line").attr("x1", 34).attr("x2", 352).attr("y1", fy(v)).attr("y2", fy(v)).attr("stroke", VC.grid);
      gw.append("text").attr("x", 30).attr("y", fy(v) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted).text(v + " f");
    });
    [60, 120, 180, 240, 300].forEach(v => {
      gw.append("line").attr("y1", 288).attr("y2", 416).attr("x1", fx(v)).attr("x2", fx(v)).attr("stroke", VC.grid);
      gw.append("text").attr("x", fx(v)).attr("y", 428).attr("text-anchor", "middle").attr("font-size", 8.5).attr("fill", VC.muted).text(v + "°");
    });
    [["flat", VC.bad], ["cyl", VC.good], ["sph", VC.accent]].forEach(([kk, col]) => {
      const pts = [];
      for (let a = 40; a <= 330; a += 2) {
        const v = canvasW(kk, a, f) / f;
        if (!isFinite(v) || v > 60) continue;
        pts.push([fx(a), fy(v)]);
      }
      if (pts.length > 1) VZ.poly(gw, pts, { stroke: col, w: kk === kind ? 2.4 : 1.2, close: false, op: kk === kind ? 1 : 0.5 });
    });
    gw.append("line").attr("x1", fx(fov)).attr("x2", fx(fov)).attr("y1", 288).attr("y2", 416)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "3 3");
    VZ.legend(gw, [{ color: VC.bad, label: "flat  2f·tan(FOV/2)" }, { color: VC.good, label: "cylindrical  f·FOV" },
                   { color: VC.accent, label: "spherical  f·FOV" }], 42, 300, { gap: 13, font: 9.5 });

    /* ---- sampling density -------------------------------------------------- */
    VZ.panelBox(g, 388, 268, 358, 162, "canvas pixels per degree of view");
    const ax = d3.scaleLinear().domain([0, 85]).range([412, 736]);
    const ay = d3.scaleLog().domain([f * VZ.rad(1) * 0.8, f * VZ.rad(1) * 20]).range([416, 288]).clamp(true);
    const ga = g.append("g");
    [1, 2, 5, 10, 20].forEach(m => {
      const v = f * VZ.rad(1) * m;
      ga.append("line").attr("x1", 412).attr("x2", 736).attr("y1", ay(v)).attr("y2", ay(v)).attr("stroke", VC.grid);
      ga.append("text").attr("x", 408).attr("y", ay(v) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted).text(m + "×");
    });
    [0, 30, 60].forEach(v => {
      ga.append("line").attr("y1", 288).attr("y2", 416).attr("x1", ax(v)).attr("x2", ax(v)).attr("stroke", VC.grid);
      ga.append("text").attr("x", ax(v)).attr("y", 428).attr("text-anchor", "middle").attr("font-size", 8.5).attr("fill", VC.muted).text(v + "°");
    });
    const flatD = VZ.linspace(0, 84, 90).map(a => [ax(a), ay(VZ.clamp(f * VZ.rad(1) / Math.pow(Math.cos(VZ.rad(a)), 2), ay.domain()[0], ay.domain()[1]))]);
    VZ.poly(ga, flatD, { stroke: VC.bad, w: kind === "flat" ? 2.4 : 1.2, close: false, op: kind === "flat" ? 1 : 0.5 });
    VZ.poly(ga, [[ax(0), ay(f * VZ.rad(1))], [ax(85), ay(f * VZ.rad(1))]],
      { stroke: VC.good, w: kind === "flat" ? 1.2 : 2.4, close: false, op: kind === "flat" ? 0.5 : 1 });
    ga.append("text").attr("x", 418).attr("y", ay(f * VZ.rad(1)) - 5).attr("font-size", 9.5).attr("fill", VC.good).text("cylindrical / spherical: constant");
    ga.append("text").attr("x", 640).attr("y", 302).attr("font-size", 9.5).attr("fill", VC.bad).text("flat: f·sec²θ");

    /* ---- the level-camera claim, measured ---------------------------------- */
    let resid = 0;
    if (NF > 1) {
      const R0 = cams[0], R1 = cams[1], R0i = camsInv[0];
      const ds = [], pts = [];
      for (let u = 0.35 * IW; u <= IW; u += 40) for (let v = 40; v <= IH - 40; v += 60) {
        const dworld = VZ.mv(R0i, [u - cx, v - cy, f]);
        const c0 = VZ.mv(R0, dworld), c1 = VZ.mv(R1, dworld);
        if (c1[2] <= 1e-3) continue;
        const q0 = toSurf(kind === "flat" ? "cyl" : kind, c0, f), q1 = toSurf(kind === "flat" ? "cyl" : kind, c1, f);
        if (!q0 || !q1) continue;
        ds.push([q1[0] - q0[0], q1[1] - q0[1]]); pts.push(1);
      }
      if (ds.length) {
        const mx = d3.mean(ds, d => d[0]), my = d3.mean(ds, d => d[1]);
        resid = Math.sqrt(d3.mean(ds, d => (d[0] - mx) ** 2 + (d[1] - my) ** 2));
      }
    }

    ui.out.innerHTML =
      `field of view ${fov}°, f = ${f} px · canvas width ` +
      (isFinite(cw) ? `<b>${VZ.fmt(cw / f, 3)} f = ${Math.round(cw)} px</b>` : "<b>unbounded</b>") +
      ` · flat would need ${isFinite(canvasW("flat", fov, f)) ? VZ.fmt(canvasW("flat", fov, f) / f, 2) + " f" : "an infinite canvas"}` +
      `, cylindrical ${VZ.fmt(canvasW("cyl", fov, f) / f, 2)} f` +
      (anyOut ? " — some frames fall behind the flat surface's horizon and cannot be drawn on it at all." : "") +
      `<br />` + (Math.abs(tilt) < 1e-9
        ? "Camera level: after warping to cylindrical coordinates, adjacent frames are related by a <b>pure horizontal translation</b> and the residual after the best shift is " + VZ.fmt(resid, 6) + " px. One parameter for the whole registration."
        : `Camera tilted ${ui.t.value}°: the pure-shift model now leaves <b>${VZ.fmt(resid, 2)} px</b> of residual between adjacent frames. The shortcut is exact only for a level camera, and the residual grows roughly in proportion to the tilt — about ${VZ.fmt(resid / Math.max(1e-6, Math.abs(+ui.t.value)), 2)} px per degree at this focal length and frame spacing.`);
  }

  [ui.s, ui.fov, ui.f, ui.t].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ═══ FIG 12 · §17 — drift, the gap, and what closes it ══════════════════════
   The true focal length is 468 px throughout; the slider is what the pipeline
   ASSUMES. The gap is then read back as an estimate of the truth.             */
(function () {
  const svg = d3.select("#bundle-svg");
  if (svg.empty()) return;
  const W = 760, H = 430, FTRUE = 468, IWH = 320;      // half-width of the sensor

  const rn = VZ.rng(556677);
  const NZ = d3.range(40).map(() => VZ.randn(rn));

  const ui = {
    n: document.getElementById("bu-n"), nv: document.getElementById("bu-nv"),
    f: document.getElementById("bu-f"), fv: document.getElementById("bu-fv"),
    s: document.getElementById("bu-s"), sv: document.getElementById("bu-sv"),
    m: document.getElementById("bu-m"), out: document.getElementById("bundle-readout")
  };

  function draw() {
    const n = +ui.n.value, fA = +ui.f.value, sig = +ui.s.value, mode = ui.m.value;
    ui.nv.textContent = n; ui.fv.textContent = fA + " px"; ui.sv.textContent = VZ.fmt(sig, 2) + "°";

    const dTrue = 2 * Math.PI / n;
    /* what the pipeline infers for each pair, with the assumed focal length */
    const est = [];
    for (let i = 0; i < n; i++) {
      const x = FTRUE * Math.tan(dTrue);
      est.push(Math.atan(x / fA) + VZ.rad(sig) * NZ[i]);
    }
    let total = d3.sum(est);
    const gapRad = 2 * Math.PI - total, gapDeg = VZ.deg(gapRad);
    const fRec = fA * (1 - gapDeg / 360);

    let head = [0], step = est.slice();
    if (mode === "spread") step = est.map(v => v + gapRad / n);
    if (mode === "ba") {
      /* re-register every pair with the recovered focal length */
      step = est.map((v, i) => Math.atan(FTRUE * Math.tan(dTrue) / fRec) + VZ.rad(sig) * NZ[i]);
    }
    for (let i = 0; i < n; i++) head.push(head[i] + step[i]);
    const total2 = head[n];

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;

    /* ---- the ring ---------------------------------------------------------- */
    VZ.panelBox(g, 14, 44, 348, 348, "the ring, seen from above");
    const c0 = [188, 220], Rr = 128;
    const halfFov = Math.atan(IWH / fA);
    const col = d3.scaleSequential(d3.interpolateTurbo).domain([0, n]);
    g.append("circle").attr("cx", c0[0]).attr("cy", c0[1]).attr("r", Rr).attr("fill", "none").attr("stroke", VC.line);
    for (let i = 0; i < n; i++) {
      const a = head[i] - Math.PI / 2;
      const p = [];
      p.push(c0);
      for (let t = -1; t <= 1; t += 0.25) {
        const b = a + t * halfFov;
        p.push([c0[0] + Rr * Math.cos(b), c0[1] + Rr * Math.sin(b)]);
      }
      VZ.poly(g, p, { stroke: col(i), w: 0.9, fill: col(i), fillOp: 0.13, op: 0.85 });
    }
    /* the gap, drawn between the last frame's far edge and the first frame's near edge */
    const aEnd = head[n] - Math.PI / 2, aStart = -Math.PI / 2;
    const arc = d3.arc()({ innerRadius: Rr + 6, outerRadius: Rr + 18, startAngle: aEnd + Math.PI / 2, endAngle: aStart + Math.PI / 2 + 2 * Math.PI });
    g.append("path").attr("d", arc).attr("transform", `translate(${c0[0]},${c0[1]})`)
      .attr("fill", Math.abs(VZ.deg(2 * Math.PI - total2)) < 0.5 ? VC.good : VC.bad).attr("fill-opacity", 0.55);
    g.append("text").attr("x", c0[0]).attr("y", 384).attr("text-anchor", "middle").attr("font-size", 11)
      .attr("fill", Math.abs(VZ.deg(2 * Math.PI - total2)) < 0.5 ? VC.good : VC.bad)
      .text(`turned ${VZ.fmt(VZ.deg(total2), 2)}° · gap ${VZ.fmt(VZ.deg(2 * Math.PI - total2), 2)}°`);

    /* ---- accumulated heading error ---------------------------------------- */
    VZ.panelBox(g, 388, 44, 358, 196, "accumulated heading error, degrees");
    const x = d3.scaleLinear().domain([0, n]).range([412, 736]);
    const errs = head.map((h, i) => VZ.deg(h - i * dTrue));
    const ext = d3.extent(errs), pad = Math.max(1, (ext[1] - ext[0]) * 0.2);
    const y = d3.scaleLinear().domain([Math.min(ext[0] - pad, -1), Math.max(ext[1] + pad, 1)]).range([222, 66]);
    const gp = g.append("g");
    y.ticks(5).forEach(v => {
      gp.append("line").attr("x1", 412).attr("x2", 736).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", v === 0 ? VC.line : VC.grid);
      gp.append("text").attr("x", 408).attr("y", y(v) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted).text(VZ.fmt(v, 0));
    });
    VZ.poly(gp, errs.map((e, i) => [x(i), y(e)]), { stroke: VC.accent, w: 2, close: false });
    errs.forEach((e, i) => gp.append("circle").attr("cx", x(i)).attr("cy", y(e)).attr("r", 2).attr("fill", VC.accent));
    gp.append("text").attr("x", 736).attr("y", 236).attr("text-anchor", "end").attr("font-size", 9.5).attr("fill", VC.muted).text("frame index →");

    /* ---- numbers ----------------------------------------------------------- */
    VZ.panelBox(g, 388, 262, 358, 130, "the gap as an instrument");
    const kv = VZ.kv(g, 400, 292, { keyW: 210, size: 11, lead: 16 });
    kv("assumed focal length", fA + " px", VC.a2);
    kv("total turned, chaining the pairs", VZ.fmt(VZ.deg(total), 3) + "°");
    kv("gap  θ_g", VZ.fmt(gapDeg, 3) + "°", Math.abs(gapDeg) < 0.5 ? VC.good : VC.bad, true);
    kv("f · (1 − θ_g/360°)", VZ.fmt(fRec, 2) + " px", VC.good, true);
    kv("true focal length", FTRUE + " px", VC.ink);
    kv("recovery error", VZ.fmt(100 * (fRec - FTRUE) / FTRUE, 2) + " %", Math.abs(fRec - FTRUE) < 8 ? VC.good : VC.a2);

    ui.out.innerHTML =
      `assumed f = ${fA} px against a true ${FTRUE} px (${VZ.fmt(100 * (fA - FTRUE) / FTRUE, 1)}% error), ${n} frames · ` +
      `chaining turns <b>${VZ.fmt(VZ.deg(total), 2)}°</b> and leaves a gap of <b>${VZ.fmt(gapDeg, 2)}°</b>.` +
      `<br />` + (mode === "chain"
        ? `Read the focal length back off the gap: f·(1 − θ_g/360°) = <b>${VZ.fmt(fRec, 1)} px</b> against the true ${FTRUE} — an error of ${VZ.fmt(Math.abs(100 * (fRec - FTRUE) / FTRUE), 2)}%, all of it the small-angle approximation, which shrinks as the number of frames grows. Try 36 frames, then 8.`
        : mode === "spread"
          ? `Spreading the gap makes the ring close, and the heading-error plot now oscillates about zero instead of climbing — but every individual rotation is still wrong by θ_g/n = ${VZ.fmt(gapDeg / n, 3)}°, and the focal length is still ${VZ.fmt(100 * (fA - FTRUE) / FTRUE, 1)}% out. Cosmetic, not corrective.`
          : `Re-registering with the recovered f = ${VZ.fmt(fRec, 1)} px leaves a residual gap of <b>${VZ.fmt(VZ.deg(2 * Math.PI - total2), 3)}°</b> — the heading error has collapsed to the per-pair noise. This is the one-parameter caricature of what bundle adjustment does with 4n parameters at once.`);
  }

  [ui.n, ui.f, ui.s, ui.m].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ═══ FIG 13 · §18 — gain compensation ═══════════════════════════════════════
   The gauge freedom is the lesson. With the prior on, g·e is constant across the
   chain; with it off, the system is singular and the solution walks toward zero. */
(function () {
  const svg = d3.select("#expo-svg");
  if (svg.empty()) return;
  const W = 760, H = 420;

  const rn = VZ.rng(3355);
  const EXP = d3.range(10).map(() => VZ.randn(rn));
  const LBAR = d3.range(10).map(() => 0.30 + 0.30 * rn());
  const ENZ = d3.range(10).map(() => VZ.randn(rn));

  const ui = {
    n: document.getElementById("ex-n"), nv: document.getElementById("ex-nv"),
    s: document.getElementById("ex-s"), sv: document.getElementById("ex-sv"),
    p: document.getElementById("ex-p"), pv: document.getElementById("ex-pv"),
    e: document.getElementById("ex-e"), ev: document.getElementById("ex-ev"),
    ring: document.getElementById("ex-ring"), out: document.getElementById("expo-readout")
  };

  function draw() {
    const n = +ui.n.value, spread = +ui.s.value / 100, invSg = +ui.p.value, enoise = +ui.e.value / 100;
    const ring = ui.ring.checked;
    ui.nv.textContent = n; ui.sv.textContent = Math.round(spread * 100) + "%";
    ui.pv.textContent = VZ.fmt(invSg, 1); ui.ev.textContent = VZ.fmt(enoise * 100, 1) + "%";

    const e = d3.range(n).map(i => Math.exp(spread * EXP[i]));           // true exposures
    const edges = [];
    for (let i = 0; i + 1 < n; i++) edges.push([i, i + 1]);
    if (ring && n > 2) edges.push([n - 1, 0]);

    /* the overlap observations */
    const Ibar = d3.range(n).map(() => new Array(n).fill(0));
    const Npx = d3.range(n).map(() => new Array(n).fill(0));
    edges.forEach(([i, j], k) => {
      const L = LBAR[k % 10];
      const ni = 1 + enoise * ENZ[(k * 2) % 10], nj = 1 + enoise * ENZ[(k * 2 + 1) % 10];
      Ibar[i][j] = e[i] * L * ni; Ibar[j][i] = e[j] * L * nj;
      Npx[i][j] = Npx[j][i] = 12000;
    });

    /* the linear system */
    const sN = 10 / 255, sg = invSg > 1e-9 ? 1 / invSg : 1e9;
    const A = d3.range(n).map(() => new Array(n).fill(0)), b = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      A[i][i] += 1 / (sg * sg); b[i] += 1 / (sg * sg);
      for (let j = 0; j < n; j++) {
        if (!Npx[i][j]) continue;
        A[i][i] += Npx[i][j] * Ibar[i][j] * Ibar[i][j] / (sN * sN);
        A[i][j] -= Npx[i][j] * Ibar[i][j] * Ibar[j][i] / (sN * sN);
      }
    }
    const Ai = VZ.invN(A);
    const g = Ai ? VZ.mv(Ai, b) : new Array(n).fill(1);
    const prod = g.map((v, i) => v * e[i]);
    const spreadProd = (d3.max(prod) - d3.min(prod));

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const gg = fr.g;

    /* ---- the strips -------------------------------------------------------- */
    const tw = 700 / n, x0 = 28;
    [["as shot", i => e[i], 46], ["after compensation", i => e[i] * g[i], 116]].forEach(([lab, val, yy]) => {
      gg.append("text").attr("x", 28).attr("y", yy - 6).attr("font-size", 11).attr("fill", VC.ink).attr("font-weight", 600).text(lab);
      for (let i = 0; i < n; i++) {
        const v = VZ.clamp(val(i) * 0.55, 0, 1);
        gg.append("rect").attr("x", x0 + i * tw).attr("y", yy).attr("width", tw - 1).attr("height", 52)
          .attr("fill", d3.rgb(255 * v, 255 * v, 255 * v).toString()).attr("stroke", VC.line);
      }
    });

    /* ---- exposures vs 1/gain ---------------------------------------------- */
    VZ.panelBox(gg, 28, 200, 340, 176, "true exposure  vs  1 / solved gain");
    const bx = d3.scaleBand().domain(d3.range(n)).range([40, 356]).padding(0.25);
    const vmax = Math.max(d3.max(e), d3.max(g.map(v => Math.abs(v) > 1e-9 ? 1 / v : 0)), 1.2) * 1.15;
    const by = d3.scaleLinear().domain([0, vmax]).range([358, 222]);
    [0, 1, 2].filter(v => v <= vmax).forEach(v => {
      gg.append("line").attr("x1", 40).attr("x2", 356).attr("y1", by(v)).attr("y2", by(v)).attr("stroke", VC.grid);
      gg.append("text").attr("x", 36).attr("y", by(v) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted).text(v);
    });
    for (let i = 0; i < n; i++) {
      const w2 = bx.bandwidth() / 2;
      gg.append("rect").attr("x", bx(i)).attr("y", by(e[i])).attr("width", w2).attr("height", 358 - by(e[i]))
        .attr("fill", VC.a2).attr("fill-opacity", 0.85);
      const inv = Math.abs(g[i]) > 1e-9 ? 1 / g[i] : 0;
      gg.append("rect").attr("x", bx(i) + w2).attr("y", by(VZ.clamp(inv, 0, vmax))).attr("width", w2)
        .attr("height", Math.max(0, 358 - by(VZ.clamp(inv, 0, vmax)))).attr("fill", VC.accent).attr("fill-opacity", 0.85);
    }
    VZ.legend(gg, [{ color: VC.a2, label: "true exposure e_k" }, { color: VC.accent, label: "1 / g_k" }], 44, 236, { gap: 13, font: 9.5 });

    /* ---- seam mismatch before / after -------------------------------------- */
    VZ.panelBox(gg, 396, 200, 350, 176, "seam mismatch, log scale");
    const sx = d3.scaleBand().domain(d3.range(edges.length)).range([414, 736]).padding(0.3);
    const sy = d3.scaleLog().domain([1e-8, 1]).range([358, 222]).clamp(true);
    [1e-6, 1e-3, 1].forEach(v => {
      gg.append("line").attr("x1", 414).attr("x2", 736).attr("y1", sy(v)).attr("y2", sy(v)).attr("stroke", VC.grid);
      gg.append("text").attr("x", 410).attr("y", sy(v) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted).text(d3.format(".0e")(v));
    });
    let worstAfter = 0;
    edges.forEach(([i, j], q) => {
      const before = Math.abs(Ibar[i][j] - Ibar[j][i]);
      const after = Math.abs(g[i] * Ibar[i][j] - g[j] * Ibar[j][i]);
      worstAfter = Math.max(worstAfter, after);
      const w2 = sx.bandwidth() / 2;
      gg.append("rect").attr("x", sx(q)).attr("y", sy(Math.max(before, 1e-8))).attr("width", w2)
        .attr("height", Math.max(0, 358 - sy(Math.max(before, 1e-8)))).attr("fill", VC.bad).attr("fill-opacity", 0.85);
      gg.append("rect").attr("x", sx(q) + w2).attr("y", sy(Math.max(after, 1e-8))).attr("width", w2)
        .attr("height", Math.max(0, 358 - sy(Math.max(after, 1e-8)))).attr("fill", VC.good).attr("fill-opacity", 0.85);
    });
    VZ.legend(gg, [{ color: VC.bad, label: "before" }, { color: VC.good, label: "after" }], 418, 236, { gap: 13, font: 9.5 });

    ui.out.innerHTML =
      `g_k · e_k = ${prod.map(v => VZ.fmt(v, 4)).join(", ")} — spread <b>${spreadProd.toExponential(2)}</b>.` +
      ` Worst seam mismatch after compensation ${worstAfter.toExponential(2)} (largest before ${d3.max(edges, ([i, j]) => Math.abs(Ibar[i][j] - Ibar[j][i])).toExponential(2)}).` +
      `<br />` + (invSg < 0.5
        ? `<b style="color:${VC.bad}">Prior switched off.</b> The objective is homogeneous in g, so g = 0 is a global minimum and the solve is singular in exactly one direction; the gains have collapsed toward ${VZ.fmt(d3.mean(g), 3)}. The constant products above show the <i>ratios</i> are still perfectly right — only the scale is unfixed.`
        : (enoise > 0.005
          ? `With ${VZ.fmt(enoise * 100, 1)}% noise on the overlap means the gains no longer fit exactly, and the residual mismatch is the noise, not the model. The products g·e are still nearly constant, which is the diagnostic that says the <i>gain</i> model is right even when the data is not clean.`
          : `Constant to ${spreadProd.toExponential(1)} — the gains recover the exposures exactly, up to one global factor of ${VZ.fmt(d3.mean(prod), 4)} chosen by the prior. That factor is a gauge, not an error.`));
  }

  [ui.n, ui.s, ui.p, ui.e, ui.ring].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ═══ FIG 14 · §19 — four ways across one seam ═══════════════════════════════
   Both metrics are computed live and both should be 1. Feathering can drive
   either to 1 but not both; multi-band drives both. Filtering uses the shared
   VZ.gauss1 / VZ.sep2 with the "clamp" border, matching the python check.      */
(function () {
  const svg = d3.select("#blend-svg");
  if (svg.empty()) return;
  const W = 760, H = 450, N = 128, SEAM = 64;

  /* the underlying texture, built once */
  const rn = VZ.rng(17171);
  const raw = VZ.zeros2(N, N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) raw[i][j] = VZ.randn(rn);
  let tex = VZ.sep2(raw, VZ.gauss1(1.2), "clamp");
  let mu = 0, s2 = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) mu += tex[i][j];
  mu /= N * N;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) s2 += (tex[i][j] - mu) ** 2;
  const sd = Math.sqrt(s2 / (N * N));
  const BASE = VZ.zeros2(N, N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) BASE[i][j] = 0.5 + 0.10 * (tex[i][j] - mu) / sd;

  const ui = {
    m: document.getElementById("bl-m"), w: document.getElementById("bl-w"), wv: document.getElementById("bl-wv"),
    e: document.getElementById("bl-e"), ev: document.getElementById("bl-ev"),
    r: document.getElementById("bl-r"), rv: document.getElementById("bl-rv"),
    out: document.getElementById("blend-readout")
  };

  const bil = (A, x, y) => {
    const x0 = VZ.clamp(Math.floor(x), 0, N - 2), y0 = VZ.clamp(Math.floor(y), 0, N - 2);
    const fx = x - x0, fy = y - y0;
    return A[y0][x0] * (1 - fx) * (1 - fy) + A[y0][x0 + 1] * fx * (1 - fy) +
           A[y0 + 1][x0] * (1 - fx) * fy + A[y0 + 1][x0 + 1] * fx * fy;
  };
  const blur = (A, s) => VZ.sep2(A, VZ.gauss1(s), "clamp");
  function maskCol(width) {
    const w = new Float64Array(N);
    for (let j = 0; j < N; j++) w[j] = VZ.clamp(0.5 - (j - SEAM) / Math.max(width, 1e-6), 0, 1);
    return w;
  }
  const mix = (A, B, w) => {
    const C = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) C[i][j] = w[j] * A[i][j] + (1 - w[j]) * B[i][j];
    return C;
  };

  function draw() {
    const method = ui.m.value, wd = +ui.w.value, ex = +ui.e.value / 100, mis = +ui.r.value;
    ui.wv.textContent = wd + " px"; ui.ev.textContent = Math.round(ex * 100) + "%"; ui.rv.textContent = VZ.fmt(mis, 1) + " px";

    const A = BASE;
    /* What §18's single gain cannot remove: a smooth, spatially varying photometric
       difference — lens fall-off, a response curve, a cloud. It is brightest toward the
       right, so the two images disagree most where they are joined. */
    const vg = new Float64Array(N);
    for (let j = 0; j < N; j++) vg[j] = 1 + ex * (0.35 + 0.65 * Math.exp(-((j - 96) ** 2) / (2 * 55 * 55)));
    const Bfull = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) Bfull[i][j] = BASE[i][j] * vg[j];
    const B = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) B[i][j] = bil(Bfull, j - mis, i);

    let C, wshow;
    if (method === "hard") { wshow = maskCol(1e-6); C = mix(A, B, wshow); }
    else if (method === "avg") { wshow = new Float64Array(N).fill(0.5); C = mix(A, B, wshow); }
    else if (method === "feather") { wshow = maskCol(wd); C = mix(A, B, wshow); }
    else {
      /* multi-band: each band blended with a transition scaled to that band */
      const levels = [1, 2, 4, 8, 16];
      C = VZ.zeros2(N, N);
      let pA = A, pB = B;
      levels.forEach(s => {
        const lA = blur(pA, s), lB = blur(pB, s), w = maskCol(2 * s);
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
          C[i][j] += w[j] * (pA[i][j] - lA[i][j]) + (1 - w[j]) * (pB[i][j] - lB[i][j]);
        pA = lA; pB = lB;
      });
      const w = maskCol(64);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) C[i][j] += w[j] * pA[i][j] + (1 - w[j]) * pB[i][j];
      wshow = maskCol(2);
    }

    /* ---- the two metrics, exactly as verified in python -------------------- */
    const lo = blur(C, 6.0);
    const gl = [];
    for (let i = 0; i < N; i++) { gl.push([]); for (let j = 0; j < N - 1; j++) gl[i].push(Math.abs(lo[i][j + 1] - lo[i][j])); }
    let at = 0, atn = 0; const all = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N - 1; j++) {
      all.push(gl[i][j]);
      if (j >= SEAM - 2 && j < SEAM + 2) { at += gl[i][j]; atn++; }
    }
    all.sort((a, b) => a - b);
    const bg = all[Math.floor(all.length / 2)] || 1e-12;
    const stepRatio = (at / Math.max(1, atn)) / bg;

    const highpass = X => { const l2 = blur(X, 2.0), Y = VZ.zeros2(N, N);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) Y[i][j] = X[i][j] - l2[i][j]; return Y; };
    /* the sharpness reference is the HARD-SEAM composite: same two sources, same
       amplitudes, no averaging anywhere. The ±3 columns at the join are excluded from
       both, so a method is not penalised for having removed the step. */
    const Chard = mix(A, B, maskCol(1e-6));
    const hp = highpass(C), hpR = highpass(Chard);
    const bandSd = Y => {
      let m = 0, k = 0;
      const cols = [];
      for (let j = SEAM - 16; j < SEAM - 3; j++) cols.push(j);
      for (let j = SEAM + 4; j < SEAM + 16; j++) cols.push(j);
      for (let i = 0; i < N; i++) for (const j of cols) { m += Y[i][j]; k++; }
      m /= k;
      let v = 0;
      for (let i = 0; i < N; i++) for (const j of cols) v += (Y[i][j] - m) ** 2;
      return Math.sqrt(v / k);
    };
    const detail = bandSd(hp) / (bandSd(hpR) || 1e-12);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;

    /* ---- images ------------------------------------------------------------ */
    const lohi = [0.25, 0.95];
    VZ.panelBox(g, 14, 60, 128, 128, "image A");
    VZ.raster(g, A, 14, 60, 128, 128, { lo: lohi[0], hi: lohi[1] });
    VZ.panelBox(g, 156, 60, 128, 128, "image B · brighter, displaced");
    VZ.raster(g, B, 156, 60, 128, 128, { lo: lohi[0], hi: lohi[1] });
    VZ.panelBox(g, 306, 44, 160, 160, "composite");
    VZ.raster(g, C, 306, 44, 160, 160, { lo: lohi[0], hi: lohi[1] });
    g.append("line").attr("x1", 306 + 160 * SEAM / N).attr("x2", 306 + 160 * SEAM / N).attr("y1", 44).attr("y2", 204)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.8);

    /* ---- the blending weight ----------------------------------------------- */
    VZ.panelBox(g, 490, 44, 256, 160, "blending weight of image A");
    const wx = d3.scaleLinear().domain([0, N - 1]).range([500, 738]);
    const wy = d3.scaleLinear().domain([-0.05, 1.05]).range([194, 58]);
    [0, 0.5, 1].forEach(v => {
      g.append("line").attr("x1", 500).attr("x2", 738).attr("y1", wy(v)).attr("y2", wy(v)).attr("stroke", VC.grid);
      g.append("text").attr("x", 496).attr("y", wy(v) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted).text(v);
    });
    if (method === "mb") {
      [2, 4, 8, 16, 32, 64].forEach((ww, q) => VZ.poly(g, d3.range(N).map(j =>
        [wx(j), wy(VZ.clamp(0.5 - (j - SEAM) / ww, 0, 1))]),
        { stroke: VC.accent, w: 1.4, close: false, op: 0.4 + 0.1 * q }));
      g.append("text").attr("x", 504).attr("y", 74).attr("font-size", 10).attr("fill", VC.accent)
        .text("one width per frequency band");
    } else {
      VZ.poly(g, d3.range(N).map(j => [wx(j), wy(wshow[j])]), { stroke: VC.accent, w: 2, close: false });
    }

    /* ---- the low-pass profile ---------------------------------------------- */
    VZ.panelBox(g, 14, 240, 452, 178, "a horizontal profile through the composite  (row 64)");
    const px = d3.scaleLinear().domain([0, N - 1]).range([28, 458]);
    const prof = d3.range(N).map(j => C[64][j]), profLo = d3.range(N).map(j => lo[64][j]);
    const py = d3.scaleLinear().domain([d3.min(prof) - 0.02, d3.max(prof) + 0.02]).range([404, 258]);
    py.ticks(4).forEach(v => g.append("line").attr("x1", 28).attr("x2", 458).attr("y1", py(v)).attr("y2", py(v)).attr("stroke", VC.grid));
    VZ.poly(g, prof.map((v, j) => [px(j), py(v)]), { stroke: VC.muted, w: 0.9, close: false, op: 0.8 });
    VZ.poly(g, profLo.map((v, j) => [px(j), py(v)]), { stroke: VC.accent, w: 2.2, close: false });
    g.append("line").attr("x1", px(SEAM)).attr("x2", px(SEAM)).attr("y1", 258).attr("y2", 404)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "3 3");
    VZ.legend(g, [{ color: VC.muted, label: "composite" }, { color: VC.accent, label: "its low-pass" }], 32, 268, { gap: 13, font: 9.5 });

    /* ---- the two metrics --------------------------------------------------- */
    VZ.panelBox(g, 490, 240, 256, 178, "both should be 1");
    const my = d3.scaleLinear().domain([0, Math.max(2, stepRatio * 1.15)]).range([404, 264]);
    [["seam step ÷ elsewhere", stepRatio, 540], ["detail kept in the band", detail, 660]].forEach(([lab, val, xc]) => {
      const good = Math.abs(val - 1) < 0.25;
      g.append("rect").attr("x", xc - 26).attr("y", my(VZ.clamp(val, 0, my.domain()[1]))).attr("width", 52)
        .attr("height", Math.max(1, 404 - my(VZ.clamp(val, 0, my.domain()[1]))))
        .attr("fill", good ? VC.good : VC.bad).attr("fill-opacity", 0.85);
      g.append("text").attr("x", xc).attr("y", my(VZ.clamp(val, 0, my.domain()[1])) - 5).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", good ? VC.good : VC.bad).attr("font-weight", 600).text(VZ.fmt(val, 3));
      g.append("text").attr("x", xc).attr("y", 416).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", VC.muted).text(lab);
    });
    g.append("line").attr("x1", 500).attr("x2", 738).attr("y1", my(1)).attr("y2", my(1))
      .attr("stroke", VC.a2).attr("stroke-dasharray", "4 3");
    g.append("text").attr("x", 738).attr("y", my(1) - 4).attr("text-anchor", "end").attr("font-size", 9).attr("fill", VC.a2).text("ideal");

    ui.out.innerHTML =
      `<b>${{ hard: "hard seam", feather: "feathering, width " + wd + " px", mb: "multi-band, 5 levels", avg: "plain averaging" }[method]}</b>` +
      ` · low-frequency step at the seam <b style="color:${Math.abs(stepRatio - 1) < 0.25 ? VC.good : VC.bad}">${VZ.fmt(stepRatio, 3)}×</b> its value elsewhere` +
      ` · detail retained in the seam band <b style="color:${Math.abs(detail - 1) < 0.25 ? VC.good : VC.bad}">${VZ.fmt(detail, 3)}</b>` +
      `<br />` + (method === "feather"
        ? `Sweep the width. Narrow leaves the step; wide removes it and takes the texture with it. There is no setting that makes both numbers 1 — which is the argument for the next option, not an argument about tuning.`
        : method === "mb"
          ? `Both close to 1 at once, because each frequency band was given the transition width that suits it: two pixels at the finest, sixty-four at the coarsest.`
          : method === "hard"
            ? `The step is fully visible — that is what the first number being far above 1 means — while the texture is untouched.`
            : `Averaging over the whole overlap is the widest possible feather: the step is gone and so is a good deal of the detail, plus any misregistration now appears as a double image everywhere rather than only at a seam.`);
  }

  [ui.m, ui.w, ui.e, ui.r].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ── extra AM helpers: pose refinement, and the analytic scenes §21–§23 use ── */
Object.assign(AM, {
  /* 6-parameter Levenberg–Marquardt on the reprojection error of a known 3-D model.
     The rotation update is an axis-angle 3-vector composed onto the current estimate,
     so the parameterisation is unconstrained and SO(3) is never left. */
  refinePose: function (R0, t0, P3, xs, K, iters) {
    let R = R0.map(r => r.slice()), t = t0.slice(), lam = 1e-3;
    const f = K[0][0], cx = K[0][2], cy = K[1][2];
    const proj = (Rr, tt) => P3.map(p => {
      const X = VZ.add(VZ.mv(Rr, p), tt);
      return (X[2] > 1e-6) ? [f * X[0] / X[2] + cx, f * X[1] / X[2] + cy] : [1e6, 1e6];
    });
    const resOf = (Rr, tt) => { const q = proj(Rr, tt), r = []; for (let i = 0; i < xs.length; i++) { r.push(q[i][0] - xs[i][0], q[i][1] - xs[i][1]); } return r; };
    const costOf = r => r.reduce((s, v) => s + v * v, 0);
    const apply = p => ({ R: VZ.mul(VZ.rodrigues([p[0], p[1], p[2]], VZ.norm([p[0], p[1], p[2]])), R),
                          t: [t[0] + p[3], t[1] + p[4], t[2] + p[5]] });
    let r = resOf(R, t), E = costOf(r);
    for (let it = 0; it < (iters || 12); it++) {
      const J = [];
      for (let i = 0; i < r.length; i++) J.push(new Array(6).fill(0));
      for (let j = 0; j < 6; j++) {
        const e = 1e-6, pp = new Array(6).fill(0), pm = new Array(6).fill(0);
        pp[j] = e; pm[j] = -e;
        const a = apply(pp), b = apply(pm);
        const ra = resOf(a.R, a.t), rb = resOf(b.R, b.t);
        for (let i = 0; i < r.length; i++) J[i][j] = (ra[i] - rb[i]) / (2 * e);
      }
      const JT = VZ.T(J), Nn = VZ.mul(JT, J), g = VZ.mv(JT, r);
      let ok = false;
      for (let tr = 0; tr < 8 && !ok; tr++) {
        const M = Nn.map((row, i) => row.map((v, j) => v + (i === j ? lam * (Nn[i][i] || 1) : 0)));
        const Mi = VZ.invN(M);
        if (!Mi) { lam *= 10; continue; }
        const d = VZ.mv(Mi, g).map(v => -v);
        const c = apply(d), rn2 = resOf(c.R, c.t), En = costOf(rn2);
        if (isFinite(En) && En < E) { R = VZ.orthonormalise(c.R); t = c.t; r = rn2; E = En; lam = Math.max(1e-9, lam * 0.4); ok = true; }
        else lam *= 10;
      }
      if (!ok) break;
    }
    return { R: R, t: t };
  },
  rotAngle: function (A, B) {
    const M = VZ.mul(VZ.T(A), B);
    return VZ.deg(Math.acos(VZ.clamp((M[0][0] + M[1][1] + M[2][2] - 1) / 2, -1, 1)));
  },

  /* ── the analytic scenes of §21–§23 ────────────────────────────────────────
     Closed-form intensity AND gradient, so a "warp" is exact at any sub-pixel
     offset and no interpolation error contaminates the demonstration. */
  sceneI: function (kind, x, y, k, th) {
    const S = t => 1 / (1 + Math.exp(-t)), w = 2.0;
    const c = Math.cos(th), s = Math.sin(th);
    if (kind === "edge") return 0.15 + k * S((x * c + y * s) / w);
    if (kind === "corner") {
      const t2 = th + VZ.rad(85), c2 = Math.cos(t2), s2 = Math.sin(t2);
      return 0.12 + 0.5 * k * S((x * c + y * s) / w) + 0.5 * k * S((x * c2 + y * s2) / w);
    }
    if (kind === "texture") return 0.5 + k * (0.35 * Math.sin(0.55 * x + 0.21 * y) +
      0.30 * Math.sin(-0.28 * x + 0.62 * y) + 0.22 * Math.sin(0.90 * x - 0.44 * y + 1.1));
    return 0.5;
  },
  sceneG: function (kind, x, y, k, th) {
    const S = t => 1 / (1 + Math.exp(-t)), w = 2.0;
    const c = Math.cos(th), s = Math.sin(th);
    if (kind === "edge") { const u = (x * c + y * s) / w, d = k * S(u) * (1 - S(u)) / w; return [d * c, d * s]; }
    if (kind === "corner") {
      const t2 = th + VZ.rad(85), c2 = Math.cos(t2), s2 = Math.sin(t2);
      const u = (x * c + y * s) / w, v = (x * c2 + y * s2) / w;
      const d1 = 0.5 * k * S(u) * (1 - S(u)) / w, d2 = 0.5 * k * S(v) * (1 - S(v)) / w;
      return [d1 * c + d2 * c2, d1 * s + d2 * s2];
    }
    if (kind === "texture") return [
      k * (0.35 * 0.55 * Math.cos(0.55 * x + 0.21 * y) - 0.30 * 0.28 * Math.cos(-0.28 * x + 0.62 * y) + 0.22 * 0.90 * Math.cos(0.90 * x - 0.44 * y + 1.1)),
      k * (0.35 * 0.21 * Math.cos(0.55 * x + 0.21 * y) + 0.30 * 0.62 * Math.cos(-0.28 * x + 0.62 * y) - 0.22 * 0.44 * Math.cos(0.90 * x - 0.44 * y + 1.1))];
    return [0, 0];
  }
});

/* ═══ FIG 15 · §20 — pose from a planar target ═══════════════════════════════
   The reprojection error is NOT the pose error. Tilt the target toward
   fronto-parallel and the first stays at the noise floor while the second grows. */
(function () {
  const svg = d3.select("#pnp-svg");
  if (svg.empty()) return;
  const W = 760, H = 420, f = 800, cx = 320, cy = 240, k = 0.55;
  const K = [[f, 0, cx], [0, f, cy], [0, 0, 1]];

  const rn = VZ.rng(9182736);
  const NZ = d3.range(200).map(() => [VZ.randn(rn), VZ.randn(rn)]);

  const ui = {
    t: document.getElementById("pn-t"), tv: document.getElementById("pn-tv"),
    z: document.getElementById("pn-z"), zv: document.getElementById("pn-zv"),
    s: document.getElementById("pn-s"), sv: document.getElementById("pn-sv"),
    n: document.getElementById("pn-n"), nv: document.getElementById("pn-nv"),
    out: document.getElementById("pnp-readout")
  };

  function draw() {
    const tilt = VZ.rad(+ui.t.value), dist = +ui.z.value, sig = +ui.s.value, ny = +ui.n.value, nx = 9;
    ui.tv.textContent = ui.t.value + "°"; ui.zv.textContent = VZ.fmt(dist, 2) + " m";
    ui.sv.textContent = VZ.fmt(sig, 2) + " px"; ui.nv.textContent = nx + " × " + ny;

    const Rtrue = VZ.mul(VZ.mul(VZ.Rz(VZ.rad(12)), VZ.Ry(-tilt)), VZ.Rx(0.6 * tilt));
    const ttrue = [0.06 - 0.025 * nx / 2, -0.03 - 0.025 * ny / 2, dist];
    const P3 = [], P2 = [];
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) { P3.push([i * 0.025, j * 0.025, 0]); P2.push([i * 0.025, j * 0.025]); }
    const proj = (R, t) => P3.map(p => { const X = VZ.add(VZ.mv(R, p), t); return (X[2] > 1e-6) ? [f * X[0] / X[2] + cx, f * X[1] / X[2] + cy] : null; });
    const clean = proj(Rtrue, ttrue);
    if (clean.some(q => !q)) { ui.out.textContent = "the target is behind the camera at this setting"; return; }
    const xs = clean.map((q, i) => [q[0] + sig * NZ[i][0], q[1] + sig * NZ[i][1]]);

    /* homography → pose */
    const d0 = AM.dlt(P2, xs, true);
    let Rc = VZ.eye(3), tc = [0, 0, 1], orth = { n1: NaN, n2: NaN, dp: NaN };
    if (d0) {
      const M = VZ.mul(VZ.inv3(K), d0.H);
      const c1 = [M[0][0], M[1][0], M[2][0]], c2 = [M[0][1], M[1][1], M[2][1]], c3 = [M[0][2], M[1][2], M[2][2]];
      const lam = 2 / (VZ.norm(c1) + VZ.norm(c2));
      const r1 = VZ.scale(c1, lam), r2 = VZ.scale(c2, lam);
      orth = { n1: VZ.norm(r1), n2: VZ.norm(r2), dp: VZ.dot(VZ.unit(r1), VZ.unit(r2)) };
      const r3 = VZ.cross(r1, r2);
      Rc = VZ.orthonormalise([[r1[0], r2[0], r3[0]], [r1[1], r2[1], r3[1]], [r1[2], r2[2], r3[2]]]);
      tc = VZ.scale(c3, lam);
      if (tc[2] < 0) { tc = VZ.neg(tc); Rc = Rc.map(r => [-r[0], -r[1], r[2]]); }
    }
    const ref = AM.refinePose(Rc, tc, P3, xs, K, 14);

    const rmsOf = (R, t) => {
      const q = proj(R, t);
      let s = 0, n = 0;
      q.forEach((p, i) => { if (p) { s += (p[0] - xs[i][0]) ** 2 + (p[1] - xs[i][1]) ** 2; n++; } });
      return n ? Math.sqrt(s / n) : NaN;
    };
    const eR0 = AM.rotAngle(Rc, Rtrue), eR1 = AM.rotAngle(ref.R, Rtrue);
    const eT0 = 1000 * VZ.norm(VZ.sub(tc, ttrue)), eT1 = 1000 * VZ.norm(VZ.sub(ref.t, ttrue));
    const rms0 = rmsOf(Rc, tc), rms1 = rmsOf(ref.R, ref.t);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    VZ.panelBox(g, 14, 44, 640 * k, 480 * k, "the image · ○ detected  × closed form  ● refined");
    const S = p => [14 + p[0] * k, 44 + p[1] * k];
    const gI = g.append("g").attr("clip-path", VZ.clip(svg, "am-pn-clip", 14, 44, 640 * k, 480 * k));
    const q0 = proj(Rc, tc), q1 = proj(ref.R, ref.t);
    xs.forEach((p, i) => {
      const a = S(p);
      gI.append("circle").attr("cx", a[0]).attr("cy", a[1]).attr("r", 3.2).attr("fill", "none").attr("stroke", VC.muted).attr("stroke-width", 1);
      if (q0[i]) { const b = S(q0[i]); gI.append("path").attr("d", `M${b[0] - 3},${b[1] - 3} l6,6 M${b[0] + 3},${b[1] - 3} l-6,6`).attr("stroke", VC.bad).attr("stroke-width", 1.2); }
      if (q1[i]) { const c = S(q1[i]); gI.append("circle").attr("cx", c[0]).attr("cy", c[1]).attr("r", 1.8).attr("fill", VC.good); }
    });
    /* the target's axes, from the refined pose */
    const O = VZ.add(VZ.mv(ref.R, [0, 0, 0]), ref.t);
    [[[0.06, 0, 0], VC.bad, "X"], [[0, 0.06, 0], VC.good, "Y"], [[0, 0, 0.06], VC.accent, "Z"]].forEach(([d, col, lab]) => {
      const A = VZ.add(VZ.mv(ref.R, [0, 0, 0]), ref.t), B = VZ.add(VZ.mv(ref.R, d), ref.t);
      if (A[2] < 1e-6 || B[2] < 1e-6) return;
      const a = S([f * A[0] / A[2] + cx, f * A[1] / A[2] + cy]), b = S([f * B[0] / B[2] + cx, f * B[1] / B[2] + cy]);
      VZ.arrow(gI, a[0], a[1], b[0], b[1], { color: col, w: 2, head: 6 });
      gI.append("text").attr("x", b[0] + 4).attr("y", b[1] - 3).attr("font-size", 10).attr("fill", col).text(lab);
    });

    /* ---- numbers ----------------------------------------------------------- */
    VZ.panelBox(g, 390, 44, 356, 340, "closed form vs refined");
    const kv = VZ.kv(g, 404, 78, { keyW: 214, size: 11, lead: 17 });
    kv("‖r₁‖ of the raw λK⁻¹H", VZ.fmt(orth.n1, 5), Math.abs(orth.n1 - 1) < 0.01 ? VC.ink : VC.a2);
    kv("‖r₂‖", VZ.fmt(orth.n2, 5), Math.abs(orth.n2 - 1) < 0.01 ? VC.ink : VC.a2);
    kv("r̂₁ · r̂₂  (0 if orthogonal)", VZ.fmt(orth.dp, 5), Math.abs(orth.dp) < 0.01 ? VC.ink : VC.a2);
    kv("", "");
    kv("rotation error, closed form", VZ.fmt(eR0, 4) + "°", VC.bad);
    kv("rotation error, refined", VZ.fmt(eR1, 4) + "°", VC.good, true);
    kv("translation error, closed form", VZ.fmt(eT0, 3) + " mm", VC.bad);
    kv("translation error, refined", VZ.fmt(eT1, 3) + " mm", eT1 < eT0 ? VC.good : VC.a2);
    kv("reprojection RMS, closed form", VZ.fmt(rms0, 4) + " px", VC.bad);
    kv("reprojection RMS, refined", VZ.fmt(rms1, 4) + " px", VC.good, true);
    kv("noise floor  σ√2·√(1 − 6/2n)", VZ.fmt(sig * Math.SQRT2 * Math.sqrt(Math.max(0, 1 - 6 / (2 * P3.length))), 4) + " px");

    const by = d3.scaleLog().domain([0.005, 30]).range([364, 274]).clamp(true);
    [["closed form", eR0, 460, VC.bad], ["refined", eR1, 600, VC.good]].forEach(([lab, v, xc, col]) => {
      g.append("rect").attr("x", xc - 30).attr("y", by(Math.max(v, 0.005))).attr("width", 60)
        .attr("height", Math.max(1, 364 - by(Math.max(v, 0.005)))).attr("fill", col).attr("fill-opacity", 0.85);
      g.append("text").attr("x", xc).attr("y", by(Math.max(v, 0.005)) - 4).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", col).text(VZ.fmt(v, 3) + "°");
      g.append("text").attr("x", xc).attr("y", 376).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", VC.muted).text(lab);
    });

    ui.out.innerHTML =
      `rotation error <b style="color:${VC.bad}">${VZ.fmt(eR0, 4)}°</b> → <b style="color:${VC.good}">${VZ.fmt(eR1, 4)}°</b>` +
      ` (${VZ.fmt(eR0 / Math.max(eR1, 1e-9), 1)}× better) · reprojection RMS ${VZ.fmt(rms0, 4)} → ${VZ.fmt(rms1, 4)} px` +
      ` · translation ${VZ.fmt(eT0, 2)} → ${VZ.fmt(eT1, 2)} mm` +
      `<br />` + (+ui.t.value < 12
        ? `<b>Nearly fronto-parallel.</b> The reprojection error is still at the noise floor and the rotation error is ${VZ.fmt(eR1, 3)}° — a small residual does not mean a good pose. A plane seen head-on constrains the out-of-plane rotations only through second-order effects, so those two angles are poorly observed however many corners you detect. Tilt the target and watch the error fall.`
        : `The raw columns of λK⁻¹H are not orthonormal (‖r₁‖ = ${VZ.fmt(orth.n1, 4)}, r̂₁·r̂₂ = ${VZ.fmt(orth.dp, 4)}); projecting them back onto SO(3) by §05's SVD is mandatory, and refining afterwards is what actually buys the accuracy.`);
  }

  [ui.t, ui.z, ui.s, ui.n].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ═══ FIG 16 · §21 — one pixel, one equation, one line ═══════════════════════
   The scenes are analytic (AM.sceneI / AM.sceneG), so the "true" temporal
   derivative is an exact sub-pixel evaluation and the gap between it and the
   linearised one is the linearisation error, not an interpolation artefact.   */
(function () {
  const svg = d3.select("#bcc-svg");
  if (svg.empty()) return;
  const W = 760, H = 420, R = 11, NG = 45;

  let U = [1.6, -1.1];                       // the true displacement, draggable
  const ui = {
    c: document.getElementById("bc-c"), th: document.getElementById("bc-th"), thv: document.getElementById("bc-thv"),
    k: document.getElementById("bc-k"), kv: document.getElementById("bc-kv"),
    lin: document.getElementById("bc-lin"), out: document.getElementById("bcc-readout")
  };

  function draw() {
    const kind = ui.c.value, th = VZ.rad(+ui.th.value), kc = +ui.k.value, lin = ui.lin.checked;
    ui.thv.textContent = ui.th.value + "°"; ui.kv.textContent = VZ.fmt(kc, 2);

    const G = AM.sceneG(kind, 0, 0, kc, th);
    const gn = Math.hypot(G[0], G[1]);
    const It = lin ? -(G[0] * U[0] + G[1] * U[1])
                   : AM.sceneI(kind, -U[0], -U[1], kc, th) - AM.sceneI(kind, 0, 0, kc, th);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;

    /* ---- the patch --------------------------------------------------------- */
    VZ.panelBox(g, 14, 44, 210, 210, "the patch, and the pixel we are looking at");
    const A = VZ.zeros2(NG, NG);
    for (let i = 0; i < NG; i++) for (let j = 0; j < NG; j++)
      A[i][j] = AM.sceneI(kind, -R + 2 * R * j / (NG - 1), -R + 2 * R * i / (NG - 1), kc, th);
    VZ.raster(g, A, 14, 44, 210, 210, { lo: 0.05, hi: 0.95 });
    const pc = [119, 149], px = 210 / (2 * R);
    g.append("circle").attr("cx", pc[0]).attr("cy", pc[1]).attr("r", 4).attr("fill", "none")
      .attr("stroke", VC.a2).attr("stroke-width", 2);
    if (gn > 1e-9) {
      const s = 34 / gn;
      VZ.arrow(g, pc[0], pc[1], pc[0] + G[0] * s, pc[1] + G[1] * s, { color: VC.accent, w: 2, head: 6 });
      g.append("text").attr("x", pc[0] + G[0] * s + 5).attr("y", pc[1] + G[1] * s).attr("font-size", 10).attr("fill", VC.accent).text("∇I");
    }
    VZ.arrow(g, pc[0], pc[1], pc[0] + U[0] * px, pc[1] + U[1] * px, { color: VC.good, w: 2, head: 6 });
    g.append("text").attr("x", 14).attr("y", 270).attr("font-size", 10).attr("fill", VC.muted)
      .text("green = the true displacement, drag it on the right");

    /* ---- velocity space ---------------------------------------------------- */
    VZ.panelBox(g, 252, 44, 248, 248, "velocity space (u, v), pixels per frame");
    const M = 4.5;
    const x = d3.scaleLinear().domain([-M, M]).range([260, 492]);
    const y = d3.scaleLinear().domain([-M, M]).range([52, 284]);
    const gv = g.append("g");
    [-4, -2, 0, 2, 4].forEach(v => {
      gv.append("line").attr("x1", 260).attr("x2", 492).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", v === 0 ? VC.line : VC.grid);
      gv.append("line").attr("y1", 52).attr("y2", 284).attr("x1", x(v)).attr("x2", x(v)).attr("stroke", v === 0 ? VC.line : VC.grid);
    });
    if (gn > 1e-9) {
      /* Iₓu + I_yv + I_t = 0 */
      const dir = [-G[1] / gn, G[0] / gn];
      const p0 = [-It * G[0] / (gn * gn), -It * G[1] / (gn * gn)];
      const seg = [[p0[0] - 20 * dir[0], p0[1] - 20 * dir[1]], [p0[0] + 20 * dir[0], p0[1] + 20 * dir[1]]];
      const gclip = gv.append("g").attr("clip-path", VZ.clip(svg, "am-bc-clip", 252, 44, 248, 248));
      VZ.poly(gclip, seg.map(p => [x(p[0]), y(p[1])]), { stroke: VC.a2, w: 2, close: false, dash: "6 4" });
      /* the normal flow: the closest point of the line to the origin */
      VZ.arrow(gv, x(0), y(0), x(p0[0]), y(p0[1]), { color: VC.violet, w: 1.8, head: 6 });
      gv.append("circle").attr("cx", x(p0[0])).attr("cy", y(p0[1])).attr("r", 3.6).attr("fill", VC.violet);
      gv.append("text").attr("x", x(p0[0]) + 6).attr("y", y(p0[1]) - 5).attr("font-size", 10).attr("fill", VC.violet).text("normal flow");
      /* ghosts: other velocities on the line are indistinguishable */
      [-2.2, 2.2].forEach(s2 => {
        const q = [p0[0] + s2 * dir[0], p0[1] + s2 * dir[1]];
        if (Math.abs(q[0]) < M && Math.abs(q[1]) < M)
          gv.append("circle").attr("cx", x(q[0])).attr("cy", y(q[1])).attr("r", 3)
            .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-dasharray", "2 2");
      });
    } else {
      gv.append("text").attr("x", 376).attr("y", 168).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", VC.bad)
        .text("∇I = 0 — no constraint at all");
    }
    const gd = gv.append("g");
    AM.drag(gd, [U], p => [x(p[0]), y(p[1])],
      q => [VZ.clamp(x.invert(q[0]), -M, M), VZ.clamp(y.invert(q[1]), -M, M)], () => draw(),
      { r: 7, fill: VC.good, stroke: VC.bg });
    gv.append("text").attr("x", x(U[0]) + 9).attr("y", y(U[1]) + 4).attr("font-size", 10).attr("fill", VC.good).text("true u");

    /* ---- numbers ----------------------------------------------------------- */
    VZ.panelBox(g, 528, 44, 218, 248, "the one equation");
    const kv = VZ.kv(g, 540, 76, { keyW: 110, size: 11, lead: 17 });
    kv("Iₓ", VZ.fmt(G[0], 5));
    kv("I_y", VZ.fmt(G[1], 5));
    kv("‖∇I‖", VZ.fmt(gn, 5), gn < 1e-4 ? VC.bad : VC.ink);
    kv("I_t", VZ.fmt(It, 5), VC.a2);
    kv("true u", `(${VZ.fmt(U[0], 2)}, ${VZ.fmt(U[1], 2)})`, VC.good);
    const cosang = gn > 1e-9 && VZ.norm(U) > 1e-9 ? Math.abs((G[0] * U[0] + G[1] * U[1]) / (gn * VZ.norm(U))) : 0;
    kv("normal flow ‖u_n‖", gn > 1e-9 ? VZ.fmt(Math.abs(It) / gn, 4) + " px" : "—", VC.violet);
    kv("fraction recovered", gn > 1e-9 ? VZ.fmt(cosang, 4) : "—", cosang > 0.9 ? VC.good : VC.bad, true);
    const ItExact = AM.sceneI(kind, -U[0], -U[1], kc, th) - AM.sceneI(kind, 0, 0, kc, th);
    const ItLin = -(G[0] * U[0] + G[1] * U[1]);
    kv("linearisation error", VZ.fmt(Math.abs(ItExact - ItLin), 5), Math.abs(ItExact - ItLin) < 0.02 ? VC.good : VC.bad);

    ui.out.innerHTML =
      (gn < 1e-4
        ? `<b style="color:${VC.bad}">‖∇I‖ = ${gn.toExponential(1)}</b> — a flat region contributes no equation whatsoever, so no velocity is preferred over any other. Not an ill-conditioned estimate: <i>no</i> estimate.`
        : `∇I = (${VZ.fmt(G[0], 4)}, ${VZ.fmt(G[1], 4)}), I_t = ${VZ.fmt(It, 4)} · the constraint line has normal ∇I and passes ${VZ.fmt(Math.abs(It) / gn, 3)} px from the origin.` +
          ` This pixel recovers <b>${VZ.fmt(100 * cosang, 1)}%</b> of the true motion — the component along ∇I — and says nothing at all about the rest.`) +
      `<br />` + (lin
        ? `Using the <i>linearised</i> I_t = −∇I·u, so the line passes exactly through the true velocity by construction. Untick the box to compute I_t by actually warping and differencing: the line then misses the truth by the linearisation error, currently ${VZ.fmt(Math.abs(ItExact - ItLin), 5)} in intensity units, and that error is what forces the iteration of §23 and the pyramid of §25.`
        : `I_t computed by warping and differencing. The line no longer passes exactly through the true velocity — it misses by the ${VZ.fmt(Math.abs(ItExact - ItLin), 5)} of linearisation error. Drag the velocity far from the origin and watch that error grow with ‖u‖².`);
  }

  [ui.c, ui.th, ui.k, ui.lin].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ── extra AM helpers: the structure tensor and one LK step on an analytic scene ── */
Object.assign(AM, {
  tensor: function (kind, kc, th, R, u) {
    /* A = Σ ∇I∇Iᵀ over a (2R+1)² window, evaluated at the CURRENT estimate u */
    let a = 0, b = 0, c = 0;
    const uu = u || [0, 0];
    for (let i = -R; i <= R; i++) for (let j = -R; j <= R; j++) {
      const G = AM.sceneG(kind, j + uu[0], i + uu[1], kc, th);
      a += G[0] * G[0]; b += G[0] * G[1]; c += G[1] * G[1];
    }
    const A = [[a, b], [b, c]];
    const tr = a + c, det = a * c - b * b;
    const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    return { A: A, lam: [Math.max(0, tr / 2 - disc), tr / 2 + disc], det: det, tr: tr };
  },

  /* one Gauss–Newton step of Lucas–Kanade on an analytic scene.
     I₁(x) = I₀(x − d), so I₁(x + u) = I₀(x + u − d). */
  lkStep: function (kind, kc, th, R, d, u, ridge, noise, rng) {
    let a = 0, b = 0, c = 0, bx = 0, by = 0, ss = 0, n = 0;
    for (let i = -R; i <= R; i++) for (let j = -R; j <= R; j++) {
      const x = j + u[0] - d[0], y = i + u[1] - d[1];
      const G = AM.sceneG(kind, x, y, kc, th);
      let e = AM.sceneI(kind, x, y, kc, th) - AM.sceneI(kind, j, i, kc, th);
      if (noise && rng) e += noise * VZ.randn(rng);
      a += G[0] * G[0]; b += G[0] * G[1]; c += G[1] * G[1];
      bx -= e * G[0]; by -= e * G[1];
      ss += e * e; n++;
    }
    const tr = a + c, det0 = a * c - b * b;
    const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det0));
    const lam = [Math.max(0, tr / 2 - disc), tr / 2 + disc];
    const eps = (ridge || 0) * lam[1];
    const A = [[a + eps, b], [b, c + eps]];
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    let du = [0, 0], ok = false;
    if (Math.abs(det) > 1e-14 * Math.max(1, tr * tr)) {
      du = [(A[1][1] * bx - A[0][1] * by) / det, (-A[1][0] * bx + A[0][0] * by) / det];
      ok = du.every(isFinite);
    }
    return { du: ok ? du : [0, 0], ok: ok, A: [[a, b], [b, c]], lam: lam, ssd: ss / Math.max(1, n) };
  }
});

/* ═══ FIG 17 · §22 — every pixel's constraint line ═══════════════════════════ */
(function () {
  const svg = d3.select("#apert-svg");
  if (svg.empty()) return;
  const W = 760, H = 440, NG = 49;

  const rng0 = VZ.rng(4242);
  const NZ = d3.range(40 * 40).map(() => VZ.randn(rng0));

  const ui = {
    c: document.getElementById("ap-c"), r: document.getElementById("ap-r"), rv: document.getElementById("ap-rv"),
    u: document.getElementById("ap-u"), uv: document.getElementById("ap-uv"),
    v: document.getElementById("ap-v"), vv: document.getElementById("ap-vv"),
    n: document.getElementById("ap-n"), nv: document.getElementById("ap-nv"),
    out: document.getElementById("apert-readout")
  };

  function draw() {
    const kind = ui.c.value, R = +ui.r.value, U = [+ui.u.value, +ui.v.value], nz = +ui.n.value;
    ui.rv.textContent = R + " px"; ui.uv.textContent = VZ.fmt(U[0], 1);
    ui.vv.textContent = VZ.fmt(U[1], 1); ui.nv.textContent = VZ.fmt(nz, 3);
    const th = VZ.rad(32), kc = 0.7;

    /* per-pixel constraints, and their least-squares solution */
    const lines = [];
    let a = 0, b = 0, c = 0, bx = 0, by = 0, q = 0;
    for (let i = -R; i <= R; i++) for (let j = -R; j <= R; j++) {
      const G = AM.sceneG(kind, j, i, kc, th);
      const gn = Math.hypot(G[0], G[1]);
      const It = -(G[0] * U[0] + G[1] * U[1]) + nz * NZ[(q++) % NZ.length];
      a += G[0] * G[0]; b += G[0] * G[1]; c += G[1] * G[1];
      bx -= It * G[0]; by -= It * G[1];
      if (gn > 1e-3) lines.push({ G: G, gn: gn, It: It });
    }
    const det = a * c - b * b, tr = a + c;
    const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const lam = [tr / 2 - disc, tr / 2 + disc];
    let sol = null;
    if (Math.abs(det) > 1e-13 * Math.max(1, tr * tr)) sol = [(c * bx - b * by) / det, (-b * bx + a * by) / det];

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;

    /* ---- the window -------------------------------------------------------- */
    VZ.panelBox(g, 14, 44, 220, 220, "the window");
    const A2 = VZ.zeros2(NG, NG), M0 = 17;
    for (let i = 0; i < NG; i++) for (let j = 0; j < NG; j++)
      A2[i][j] = AM.sceneI(kind, -M0 + 2 * M0 * j / (NG - 1), -M0 + 2 * M0 * i / (NG - 1), kc, th);
    VZ.raster(g, A2, 14, 44, 220, 220, { lo: 0.05, hi: 0.95 });
    const s0 = 220 / (2 * M0);
    g.append("rect").attr("x", 124 - R * s0).attr("y", 154 - R * s0).attr("width", 2 * R * s0).attr("height", 2 * R * s0)
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1.6);

    /* ---- velocity space ---------------------------------------------------- */
    VZ.panelBox(g, 258, 44, 250, 250, `${lines.length} constraint lines`);
    const M = 4.5;
    const x = d3.scaleLinear().domain([-M, M]).range([266, 500]);
    const y = d3.scaleLinear().domain([-M, M]).range([52, 286]);
    const gv = g.append("g").attr("clip-path", VZ.clip(svg, "am-ap-clip", 258, 44, 250, 250));
    [-4, -2, 0, 2, 4].forEach(v => {
      gv.append("line").attr("x1", 266).attr("x2", 500).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", v === 0 ? VC.line : VC.grid);
      gv.append("line").attr("y1", 52).attr("y2", 286).attr("x1", x(v)).attr("x2", x(v)).attr("stroke", v === 0 ? VC.line : VC.grid);
    });
    const stride = Math.max(1, Math.ceil(lines.length / 220));
    lines.forEach((L, i) => {
      if (i % stride) return;
      const dir = [-L.G[1] / L.gn, L.G[0] / L.gn];
      const p0 = [-L.It * L.G[0] / (L.gn * L.gn), -L.It * L.G[1] / (L.gn * L.gn)];
      gv.append("line")
        .attr("x1", x(p0[0] - 30 * dir[0])).attr("y1", y(p0[1] - 30 * dir[1]))
        .attr("x2", x(p0[0] + 30 * dir[0])).attr("y2", y(p0[1] + 30 * dir[1]))
        .attr("stroke", VC.accent).attr("stroke-width", 0.7).attr("stroke-opacity", 0.28);
    });
    /* the uncertainty ellipse: axes ∝ 1/√λ */
    if (sol) {
      const e = VZ.jacobiEig([[a, b], [b, c]], 40);
      const sc = 1.6 * Math.sqrt(tr) * 0.35;
      const ax = [sc / Math.sqrt(Math.max(e.values[0], 1e-9)), sc / Math.sqrt(Math.max(e.values[1], 1e-9))];
      const v0 = e.vectors[0];
      const ang = VZ.deg(Math.atan2(v0[1], v0[0]));
      gv.append("ellipse").attr("cx", x(sol[0])).attr("cy", y(sol[1]))
        .attr("rx", Math.min(600, Math.abs(x(ax[0]) - x(0)))).attr("ry", Math.min(600, Math.abs(x(ax[1]) - x(0))))
        .attr("transform", `rotate(${ang},${x(sol[0])},${y(sol[1])})`)
        .attr("fill", VC.violet).attr("fill-opacity", 0.10).attr("stroke", VC.violet).attr("stroke-dasharray", "4 3");
      gv.append("circle").attr("cx", x(sol[0])).attr("cy", y(sol[1])).attr("r", 4.5).attr("fill", VC.violet);
    }
    gv.append("path").attr("d", `M${x(U[0]) - 7},${y(U[1])} H${x(U[0]) + 7} M${x(U[0])},${y(U[1]) - 7} V${y(U[1]) + 7}`)
      .attr("stroke", VC.good).attr("stroke-width", 2.4);
    VZ.legend(g, [{ color: VC.good, label: "true motion" }, { color: VC.violet, label: "least squares + its ellipse" }], 266, 306, { gap: 14, font: 10 });

    /* ---- the tensor -------------------------------------------------------- */
    VZ.panelBox(g, 530, 44, 216, 250, "the structure tensor A = ∑ ∇I∇Iᵀ");
    VZ.matText(g, [[a, b], [b, c]], 546, 90, { dp: 4, pad: 10, size: 10.5, lead: 15 });
    const kv = VZ.kv(g, 546, 148, { keyW: 108, size: 11, lead: 17 });
    kv("λ₀ (the smaller)", VZ.sig(lam[0], 5), lam[0] < 1e-6 * Math.max(lam[1], 1e-12) ? VC.bad : VC.good, true);
    kv("λ₁", VZ.sig(lam[1], 5));
    kv("λ₁ / λ₀", lam[0] > 1e-12 ? VZ.fmt(lam[1] / lam[0], 2) : "∞", lam[0] > 1e-12 ? VC.ink : VC.bad);
    kv("det A", VZ.sig(det, 4));
    kv("error of the LS fit", sol ? VZ.fmt(Math.hypot(sol[0] - U[0], sol[1] - U[1]), 4) + " px" : "no solution",
      sol && Math.hypot(sol[0] - U[0], sol[1] - U[1]) < 0.1 ? VC.good : VC.bad, true);
    kv("Harris R, κ = 0.05", VZ.sig(det - 0.05 * tr * tr, 4));

    ui.out.innerHTML =
      (kind === "flat"
        ? `<b style="color:${VC.bad}">A = 0 exactly.</b> No pixel of a flat region contributes a constraint line, so velocity space is completely empty and every motion is equally consistent with the data.`
        : (lam[0] < 1e-6 * lam[1]
          ? `<b style="color:${VC.bad}">λ₀ = ${VZ.sig(lam[0], 3)} against λ₁ = ${VZ.sig(lam[1], 3)}</b> — every constraint line is parallel, they intersect in a <i>line</i> rather than a point, and the least-squares solution is free to slide along it. The uncertainty ellipse has degenerated into a stripe. This is the aperture problem, and it is a property of the data, not of the solver.`
          : `λ₀ = ${VZ.sig(lam[0], 4)}, λ₁ = ${VZ.sig(lam[1], 4)}, ratio ${VZ.fmt(lam[1] / lam[0], 2)} — the lines cross at a well-defined point and the least-squares solution is ${sol ? VZ.fmt(Math.hypot(sol[0] - U[0], sol[1] - U[1]), 4) : "—"} px from the truth.`)) +
      `<br />This matrix is <a href="features.html#tensor">Part 4 §05</a>'s structure tensor, unchanged. Its Harris score is printed on the right: the detector's question "is this a corner?" and the tracker's question "can I solve for the motion here?" are the same question about the same 2 × 2 matrix.`;
  }

  [ui.c, ui.r, ui.u, ui.v, ui.n].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ═══ FIG 18 · §23 — Lucas–Kanade, iteration by iteration ════════════════════ */
(function () {
  const svg = d3.select("#lk-svg");
  if (svg.empty()) return;
  const W = 760, H = 450, NG = 41;
  const KC = 0.7, TH = VZ.rad(32);
  let st = { u: [0, 0], it: 0, path: [[0, 0]], ssd: [] };

  const ui = {
    c: document.getElementById("lk-c"), d: document.getElementById("lk-d"), dv: document.getElementById("lk-dv"),
    a: document.getElementById("lk-a"), av: document.getElementById("lk-av"),
    r: document.getElementById("lk-r"), rv: document.getElementById("lk-rv"),
    e: document.getElementById("lk-e"), ev: document.getElementById("lk-ev"),
    step: document.getElementById("lk-step"), run: document.getElementById("lk-run"),
    reset: document.getElementById("lk-reset"), out: document.getElementById("lk-readout")
  };

  const D = () => { const m = +ui.d.value, a = VZ.rad(+ui.a.value); return [m * Math.cos(a), m * Math.sin(a)]; };
  function reset() {
    st = { u: [0, 0], it: 0, path: [[0, 0]], ssd: [] };
    const s = AM.lkStep(ui.c.value, KC, TH, +ui.r.value, D(), st.u, +ui.e.value / 100);
    st.ssd.push(s.ssd);
  }
  function step() {
    const s = AM.lkStep(ui.c.value, KC, TH, +ui.r.value, D(), st.u, +ui.e.value / 100);
    if (!s.ok) return false;
    const nu = [st.u[0] + VZ.clamp(s.du[0], -3, 3), st.u[1] + VZ.clamp(s.du[1], -3, 3)];
    st.u = nu; st.it++; st.path.push(nu.slice());
    st.ssd.push(AM.lkStep(ui.c.value, KC, TH, +ui.r.value, D(), st.u, +ui.e.value / 100).ssd);
    return Math.hypot(s.du[0], s.du[1]) > 1e-4;
  }

  function draw() {
    const kind = ui.c.value, R = +ui.r.value, ridge = +ui.e.value / 100, d = D();
    ui.dv.textContent = VZ.fmt(+ui.d.value, 1) + " px"; ui.av.textContent = ui.a.value + "°";
    ui.rv.textContent = R + " px"; ui.ev.textContent = ui.e.value + "%";

    const cur = AM.lkStep(kind, KC, TH, R, d, st.u, ridge);
    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;

    /* ---- template / warped target / difference ----------------------------- */
    const M0 = R + 3;
    const grid = (ox, oy) => {
      const A = VZ.zeros2(NG, NG);
      for (let i = 0; i < NG; i++) for (let j = 0; j < NG; j++)
        A[i][j] = AM.sceneI(kind, -M0 + 2 * M0 * j / (NG - 1) + ox, -M0 + 2 * M0 * i / (NG - 1) + oy, KC, TH);
      return A;
    };
    const T0 = grid(0, 0), T1 = grid(st.u[0] - d[0], st.u[1] - d[1]);
    const DF = VZ.zeros2(NG, NG);
    for (let i = 0; i < NG; i++) for (let j = 0; j < NG; j++) DF[i][j] = T1[i][j] - T0[i][j];
    [["template I₀", T0, 58, { lo: 0.05, hi: 0.95 }],
     ["I₁ warped by the current u", T1, 178, { lo: 0.05, hi: 0.95 }],
     ["their difference", DF, 298, { signed: true, lo: -0.5, hi: 0.5 }]].forEach(([lab, A, yy, opt]) => {
      VZ.panelBox(g, 14, yy, 106, 106, lab);
      VZ.raster(g, A, 14, yy, 106, 106, opt);
    });

    /* ---- velocity space ---------------------------------------------------- */
    VZ.panelBox(g, 250, 44, 250, 250, "velocity space · the iteration's path");
    const M = Math.max(4, +ui.d.value * 1.35);
    const x = d3.scaleLinear().domain([-M, M]).range([258, 492]);
    const y = d3.scaleLinear().domain([-M, M]).range([52, 286]);
    const gv = g.append("g").attr("clip-path", VZ.clip(svg, "am-lk-clip", 250, 44, 250, 250));
    x.ticks(5).forEach(v => {
      gv.append("line").attr("x1", 258).attr("x2", 492).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", v === 0 ? VC.line : VC.grid);
      gv.append("line").attr("y1", 52).attr("y2", 286).attr("x1", x(v)).attr("x2", x(v)).attr("stroke", v === 0 ? VC.line : VC.grid);
    });
    for (let i = 1; i < st.path.length; i++)
      VZ.arrow(gv, x(st.path[i - 1][0]), y(st.path[i - 1][1]), x(st.path[i][0]), y(st.path[i][1]),
        { color: VC.accent, w: 1.4, head: 5, op: 0.85 });
    if (cur.lam[0] > 1e-12) {
      const e = VZ.jacobiEig(cur.A, 40), sc = 0.9 * Math.sqrt(cur.lam[1]) * 0.35;
      const v0 = e.vectors[0];
      gv.append("ellipse").attr("cx", x(st.u[0])).attr("cy", y(st.u[1]))
        .attr("rx", Math.min(900, Math.abs(x(sc / Math.sqrt(e.values[0])) - x(0))))
        .attr("ry", Math.min(900, Math.abs(x(sc / Math.sqrt(e.values[1])) - x(0))))
        .attr("transform", `rotate(${VZ.deg(Math.atan2(v0[1], v0[0]))},${x(st.u[0])},${y(st.u[1])})`)
        .attr("fill", VC.violet).attr("fill-opacity", 0.09).attr("stroke", VC.violet).attr("stroke-dasharray", "4 3");
    }
    gv.append("path").attr("d", `M${x(d[0]) - 8},${y(d[1])} H${x(d[0]) + 8} M${x(d[0])},${y(d[1]) - 8} V${y(d[1]) + 8}`)
      .attr("stroke", VC.good).attr("stroke-width", 2.4);
    gv.append("circle").attr("cx", x(st.u[0])).attr("cy", y(st.u[1])).attr("r", 4.5).attr("fill", VC.accent);
    VZ.legend(g, [{ color: VC.good, label: "the true displacement" }, { color: VC.accent, label: "the estimate, and its path" }], 258, 306, { gap: 14, font: 10 });

    /* ---- SSD against iteration -------------------------------------------- */
    VZ.panelBox(g, 250, 344, 250, 92, "mean squared difference, log scale");
    const lo = Math.max(d3.min(st.ssd) || 1e-8, 1e-10), hi = Math.max(d3.max(st.ssd) || 1e-6, lo * 10);
    const sx = d3.scaleLinear().domain([0, Math.max(1, st.ssd.length - 1)]).range([262, 492]);
    const sy = d3.scaleLog().domain([lo * 0.7, hi * 1.4]).range([428, 352]).clamp(true);
    if (st.ssd.length > 1) VZ.poly(g, st.ssd.map((v, i) => [sx(i), sy(Math.max(v, 1e-10))]), { stroke: VC.a2, w: 1.8, close: false });
    st.ssd.forEach((v, i) => g.append("circle").attr("cx", sx(i)).attr("cy", sy(Math.max(v, 1e-10))).attr("r", 2).attr("fill", VC.a2));

    /* ---- numbers ----------------------------------------------------------- */
    VZ.panelBox(g, 528, 44, 218, 392, "the 2 × 2 system");
    VZ.matText(g, cur.A, 544, 92, { dp: 4, pad: 10, size: 10.5, lead: 15 });
    const kv = VZ.kv(g, 544, 150, { keyW: 116, size: 11, lead: 17 });
    const sing = cur.lam[0] < 1e-9 * Math.max(cur.lam[1], 1e-12);
    kv("λ₀", VZ.sig(cur.lam[0], 5), sing ? VC.bad : VC.good, true);
    kv("λ₁", VZ.sig(cur.lam[1], 5));
    kv("λ₁ / λ₀", sing ? "∞ (singular)" : VZ.fmt(cur.lam[1] / cur.lam[0], 2), sing ? VC.bad : VC.ink);
    kv("iterations so far", String(st.it));
    kv("current estimate u", `(${VZ.fmt(st.u[0], 4)}, ${VZ.fmt(st.u[1], 4)})`, VC.accent);
    kv("true displacement", `(${VZ.fmt(d[0], 4)}, ${VZ.fmt(d[1], 4)})`, VC.good);
    kv("error", VZ.fmt(Math.hypot(st.u[0] - d[0], st.u[1] - d[1]), 5) + " px",
      Math.hypot(st.u[0] - d[0], st.u[1] - d[1]) < 0.02 ? VC.good : VC.bad, true);
    kv("last update ‖Δu‖", VZ.fmt(Math.hypot(cur.du[0], cur.du[1]), 5) + " px");
    kv("mean squared diff", cur.ssd.toExponential(3));
    kv("ridge ε", ridge ? VZ.fmt(ridge * cur.lam[1], 5) + "  (" + ui.e.value + "% of λ₁)" : "0");

    const err = Math.hypot(st.u[0] - d[0], st.u[1] - d[1]);
    ui.out.innerHTML =
      `${kind} · true displacement ${VZ.fmt(+ui.d.value, 1)} px · after <b>${st.it}</b> iterations the estimate is (${VZ.fmt(st.u[0], 4)}, ${VZ.fmt(st.u[1], 4)}),` +
      ` error <b style="color:${err < 0.02 ? VC.good : VC.bad}">${VZ.fmt(err, 5)} px</b> · λ₀ = ${VZ.sig(cur.lam[0], 4)}, λ₁ = ${VZ.sig(cur.lam[1], 4)}` +
      `<br />` + (kind === "flat"
        ? `A = 0: there is nothing to solve and the estimate never leaves the origin. The ridge control cannot help either, because it adds a multiple of λ₁ and λ₁ is zero too.`
        : sing && ridge === 0
          ? `<b style="color:${VC.bad}">A is singular</b> — an edge gives one equation for two unknowns however many pixels are in the window. The solver refuses. Raise the ridge above zero and it will produce a stable, biased answer: the normal-flow component right and the tangential component pulled to zero.`
          : ridge > 0 && sing
            ? `With a ridge of ${ui.e.value}% of λ₁ the singular system becomes solvable. The component along the gradient is recovered correctly and the component along the edge is biased toward zero — which is the honest thing for a solver to do when the data says nothing about it.`
            : err < 0.02
              ? `Converged to machine precision in ${st.it} iterations. Push the true displacement past about 7 px on the texture and it will converge instead to a <i>different</i> local minimum — confidently, and with a small residual. That is what §25's pyramid exists to prevent.`
              : `Not converged. Either more iterations are needed, or the displacement is outside the basin of attraction and no number of iterations will help.`);
  }

  ui.step.addEventListener("click", () => { step(); draw(); });
  ui.run.addEventListener("click", () => { let n = 0; while (step() && n++ < 40); draw(); });
  ui.reset.addEventListener("click", () => { reset(); draw(); });
  [ui.c, ui.d, ui.a, ui.r, ui.e].forEach(el => {
    el.addEventListener("input", () => { reset(); draw(); });
    el.addEventListener("change", () => { reset(); draw(); });
  });
  reset();
  for (let i = 0; i < 6 && step(); i++);      // arrive part-converged, not blank
  draw();
})();

/* ── extra AM helpers: the raster dense-motion kit (§24–§27) ─────────────────
   Filtering goes through VZ.gauss1 / VZ.sep2 with the "clamp" border, which is
   scipy's 'nearest' and was verified against it to 0.000e+00. */
Object.assign(AM, {
  bilinear: function (A, x, y) {
    const H = A.length, W = A[0].length;
    const x0 = VZ.clamp(Math.floor(x), 0, W - 2), y0 = VZ.clamp(Math.floor(y), 0, H - 2);
    const fx = x - x0, fy = y - y0;
    return A[y0][x0] * (1 - fx) * (1 - fy) + A[y0][x0 + 1] * fx * (1 - fy) +
           A[y0 + 1][x0] * (1 - fx) * fy + A[y0 + 1][x0 + 1] * fx * fy;
  },
  grads: function (A) {
    const H = A.length, W = A[0].length, Ix = VZ.zeros2(H, W), Iy = VZ.zeros2(H, W);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
      Ix[i][j] = (j > 0 && j < W - 1) ? (A[i][j + 1] - A[i][j - 1]) / 2 : (j === 0 ? A[i][1] - A[i][0] : A[i][W - 1] - A[i][W - 2]);
      Iy[i][j] = (i > 0 && i < H - 1) ? (A[i + 1][j] - A[i - 1][j]) / 2 : (i === 0 ? A[1][j] - A[0][j] : A[H - 1][j] - A[H - 2][j]);
    }
    return { Ix: Ix, Iy: Iy };
  },
  /* one pyramid level down: smooth with σ = 1, then take every other sample.
     Skipping the smoothing is the classic bug, so it is exposed as a flag. */
  pyrDown: function (A, smooth) {
    const S = (smooth === false) ? A : VZ.sep2(A, VZ.gauss1(1.0), "clamp");
    const H = S.length >> 1, W = S[0].length >> 1, B = VZ.zeros2(H, W);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) B[i][j] = S[2 * i][2 * j];
    return B;
  },
  pyramid: function (A, L, smooth) {
    const P = [A];
    for (let l = 1; l < L; l++) P.push(AM.pyrDown(P[l - 1], smooth));
    return P;
  },
  /* translation-only Lucas–Kanade on a raster pair, with a per-step clamp so a
     single wild update cannot throw the iteration out of the image */
  lkTrans: function (I0, I1, u0, iters, marg, cap) {
    const H = I0.length, W = I0[0].length, m = marg === undefined ? 4 : marg;
    const G = AM.grads(I1);
    let u = u0.slice();
    const lim = cap === undefined ? 1.0 : cap;
    for (let k = 0; k < iters; k++) {
      let a = 0, b = 0, c = 0, bx = 0, by = 0;
      for (let i = m; i < H - m; i++) for (let j = m; j < W - m; j++) {
        const xs = j + u[0], ys = i + u[1];
        if (xs < 0 || xs > W - 2 || ys < 0 || ys > H - 2) continue;
        const w1 = AM.bilinear(I1, xs, ys);
        const gx = AM.bilinear(G.Ix, xs, ys), gy = AM.bilinear(G.Iy, xs, ys);
        const e = w1 - I0[i][j];
        a += gx * gx; b += gx * gy; c += gy * gy; bx -= e * gx; by -= e * gy;
      }
      const det = a * c - b * b;
      if (!(Math.abs(det) > 1e-12)) break;
      let du = [(c * bx - b * by) / det, (-b * bx + a * by) / det];
      du = [VZ.clamp(du[0], -lim, lim), VZ.clamp(du[1], -lim, lim)];
      u = [u[0] + du[0], u[1] + du[1]];
      if (Math.hypot(du[0], du[1]) < 1e-4) break;
    }
    return u;
  },
  coarseToFine: function (I0, I1, L, iters, smooth) {
    const P0 = AM.pyramid(I0, L, smooth), P1 = AM.pyramid(I1, L, smooth);
    let u = [0, 0];
    for (let l = L - 1; l >= 0; l--) {
      u = AM.lkTrans(P0[l], P1[l], u, iters, 3);
      if (l > 0) u = [2 * u[0], 2 * u[1]];
    }
    return u;
  },
  /* Horn–Schunck, Jacobi form, with the classic 3 × 3 averaging stencil */
  hornSchunck: function (Ix, Iy, It, alpha, iters) {
    const H = Ix.length, W = Ix[0].length;
    let u = VZ.zeros2(H, W), v = VZ.zeros2(H, W);
    const K = [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]];
    const a2 = alpha * alpha;
    for (let k = 0; k < iters; k++) {
      const ub = VZ.corr2(u, K, "clamp"), vb = VZ.corr2(v, K, "clamp");
      const un = VZ.zeros2(H, W), vn = VZ.zeros2(H, W);
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        const d = (Ix[i][j] * ub[i][j] + Iy[i][j] * vb[i][j] + It[i][j]) /
                  (a2 + Ix[i][j] * Ix[i][j] + Iy[i][j] * Iy[i][j]);
        un[i][j] = ub[i][j] - Ix[i][j] * d;
        vn[i][j] = vb[i][j] - Iy[i][j] * d;
      }
      u = un; v = vn;
    }
    return { u: u, v: v };
  },
  /* hue = direction, saturation = magnitude — the standard flow colour wheel */
  flowColor: function (u, v, mx) {
    const m = Math.hypot(u, v) / (mx || 1);
    const h = (VZ.deg(Math.atan2(v, u)) + 360) % 360;
    return d3.hsl(h, 0.95, VZ.clamp(0.92 - 0.55 * VZ.clamp(m, 0, 1), 0.28, 0.95)).toString();
  }
});

/* ═══ FIG 19 · §24 — Horn–Schunck ════════════════════════════════════════════
   40 × 40, so the α sweep and the iteration sweep can both be recomputed live.
   The right half of the default scene is exactly flat: the fill-in front is the
   thing to watch, and it is slow on purpose, because it really is.            */
(function () {
  const svg = d3.select("#hs-svg");
  if (svg.empty()) return;
  const W = 760, H = 450, N = 40, PAD = 12, NB = N + 2 * PAD;
  const DTRUE = [1.0, -0.6];

  const rn = VZ.rng(90210);
  const raw = VZ.zeros2(NB, NB);
  for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) raw[i][j] = VZ.randn(rn);
  let tx = VZ.sep2(raw, VZ.gauss1(1.5), "clamp");
  let mu = 0; for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) mu += tx[i][j];
  mu /= NB * NB;
  let sd = 0; for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) sd += (tx[i][j] - mu) ** 2;
  sd = Math.sqrt(sd / (NB * NB));
  for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) tx[i][j] = (tx[i][j] - mu) / sd;

  const ui = {
    a: document.getElementById("hs-a"), av: document.getElementById("hs-av"),
    i: document.getElementById("hs-i"), iv: document.getElementById("hs-iv"),
    s: document.getElementById("hs-s"), lk: document.getElementById("hs-lk"),
    out: document.getElementById("hs-readout")
  };

  function build(scene) {
    /* the full-canvas image, and the true flow at every pixel of the window */
    const img = VZ.zeros2(NB, NB);
    for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) {
      const jj = j - PAD;
      const on = (scene === "half") ? (jj < N / 2) : true;
      img[i][j] = 0.5 + (on ? 0.12 * tx[i][j] : 0);
    }
    const I0 = VZ.zeros2(N, N), I1 = VZ.zeros2(N, N);
    const GU = VZ.zeros2(N, N), GV = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const d = (scene === "split") ? (j < N / 2 ? DTRUE : [-DTRUE[0], -DTRUE[1]]) : DTRUE;
      GU[i][j] = d[0]; GV[i][j] = d[1];
      I0[i][j] = img[i + PAD][j + PAD];
      I1[i][j] = AM.bilinear(img, j + PAD - d[0], i + PAD - d[1]);
    }
    return { I0: I0, I1: I1, GU: GU, GV: GV };
  }

  function aepe(u, v, GU, GV, j0, j1) {
    let s = 0, n = 0;
    for (let i = 2; i < N - 2; i++) for (let j = Math.max(2, j0); j < Math.min(N - 2, j1); j++) {
      s += Math.hypot(u[i][j] - GU[i][j], v[i][j] - GV[i][j]); n++;
    }
    return n ? s / n : NaN;
  }

  function draw() {
    const alpha = Math.pow(10, +ui.a.value), iters = +ui.i.value, scene = ui.s.value, cmp = ui.lk.checked;
    ui.av.textContent = VZ.sig(alpha, 3); ui.iv.textContent = iters;

    const S = build(scene);
    const G = AM.grads(S.I0);
    const It = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) It[i][j] = S.I1[i][j] - S.I0[i][j];
    const F = AM.hornSchunck(G.Ix, G.Iy, It, alpha, iters);
    const eAll = aepe(F.u, F.v, S.GU, S.GV, 0, N);
    const eL = aepe(F.u, F.v, S.GU, S.GV, 0, N / 2), eR = aepe(F.u, F.v, S.GU, S.GV, N / 2, N);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;

    VZ.panelBox(g, 14, 60, 176, 176, "frame 0" + (scene === "half" ? " · right half is flat" : ""));
    VZ.raster(g, S.I0, 14, 60, 176, 176, { lo: 0.2, hi: 0.8 });

    /* ---- the flow field ---------------------------------------------------- */
    VZ.panelBox(g, 212, 60, 200, 200, "recovered flow · hue = direction");
    const mx = 1.4 * Math.hypot(DTRUE[0], DTRUE[1]);
    VZ.cells(g, 212, 60, 200 / N, N, N, (jx, iy) => AM.flowColor(F.u[iy][jx], F.v[iy][jx], mx));
    const cw = 200 / N;
    for (let i = 2; i < N; i += 5) for (let j = 2; j < N; j += 5) {
      const x0 = 212 + (j + 0.5) * cw, y0 = 60 + (i + 0.5) * cw;
      const s2 = 9 / Math.max(0.3, mx);
      if (Math.hypot(F.u[i][j], F.v[i][j]) > 0.03)
        VZ.arrow(g, x0, y0, x0 + F.u[i][j] * s2, y0 + F.v[i][j] * s2, { color: VC.ink, w: 0.9, head: 3, op: 0.75 });
    }
    if (cmp) {
      /* the windowed Lucas–Kanade answer, drawn only where its 2 × 2 is solvable */
      for (let i = 4; i < N - 4; i += 4) for (let j = 4; j < N - 4; j += 4) {
        let a = 0, b = 0, c = 0, bx = 0, by = 0;
        for (let p = -4; p <= 4; p++) for (let q = -4; q <= 4; q++) {
          const gx = G.Ix[i + p][j + q], gy = G.Iy[i + p][j + q], e = It[i + p][j + q];
          a += gx * gx; b += gx * gy; c += gy * gy; bx -= e * gx; by -= e * gy;
        }
        const det = a * c - b * b, tr = a + c;
        if (!(det > 1e-6 * tr * tr)) continue;
        const uu = (c * bx - b * by) / det, vv = (-b * bx + a * by) / det;
        const x0 = 212 + (j + 0.5) * cw, y0 = 60 + (i + 0.5) * cw;
        VZ.arrow(g, x0, y0, x0 + uu * 9 / Math.max(0.3, mx), y0 + vv * 9 / Math.max(0.3, mx),
          { color: VC.a2, w: 1.3, head: 4 });
      }
      g.append("text").attr("x", 212).attr("y", 276).attr("font-size", 10).attr("fill", VC.a2)
        .text("orange = windowed Lucas–Kanade, drawn only where its 2 × 2 is solvable");
    }

    /* ---- error against α --------------------------------------------------- */
    VZ.panelBox(g, 440, 60, 306, 168, "average endpoint error vs α");
    const ax = d3.scaleLog().domain([1e-3, 2]).range([460, 736]);
    const curves = { all: [], left: [], right: [] };
    [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 2].forEach(av => {
      const f2 = AM.hornSchunck(G.Ix, G.Iy, It, av, Math.min(iters, 220));
      curves.all.push([av, aepe(f2.u, f2.v, S.GU, S.GV, 0, N)]);
      curves.left.push([av, aepe(f2.u, f2.v, S.GU, S.GV, 0, N / 2)]);
      curves.right.push([av, aepe(f2.u, f2.v, S.GU, S.GV, N / 2, N)]);
    });
    const ymax = d3.max(curves.all.concat(curves.left, curves.right), d => d[1]) * 1.1 || 1;
    const ay = d3.scaleLinear().domain([0, ymax]).range([214, 78]);
    ay.ticks(4).forEach(v => {
      g.append("line").attr("x1", 460).attr("x2", 736).attr("y1", ay(v)).attr("y2", ay(v)).attr("stroke", VC.grid);
      g.append("text").attr("x", 456).attr("y", ay(v) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted).text(VZ.fmt(v, 1));
    });
    [[curves.all, VC.accent, "whole field"], [curves.left, VC.good, "textured half"], [curves.right, VC.bad, "flat half"]]
      .forEach(([cv, col]) => VZ.poly(g, cv.map(d => [ax(d[0]), ay(VZ.clamp(d[1], 0, ymax))]), { stroke: col, w: 1.8, close: false }));
    g.append("line").attr("x1", ax(VZ.clamp(alpha, 1e-3, 2))).attr("x2", ax(VZ.clamp(alpha, 1e-3, 2)))
      .attr("y1", 78).attr("y2", 214).attr("stroke", VC.a2).attr("stroke-dasharray", "3 3");
    [1e-3, 1e-2, 1e-1, 1].forEach(v => g.append("text").attr("x", ax(v)).attr("y", 228).attr("text-anchor", "middle")
      .attr("font-size", 8.5).attr("fill", VC.muted).text(d3.format(".0e")(v)));
    VZ.legend(g, [{ color: VC.accent, label: "whole field" }, { color: VC.good, label: "textured" }, { color: VC.bad, label: "flat" }],
      466, 88, { gap: 13, font: 9.5 });

    /* ---- error against iteration count ------------------------------------- */
    VZ.panelBox(g, 440, 264, 306, 154, "error vs iteration count, at the current α");
    const its = [20, 60, 120, 240, 480, 900];
    const ser = { all: [], left: [], right: [] };
    its.forEach(k => {
      const f2 = AM.hornSchunck(G.Ix, G.Iy, It, alpha, k);
      ser.all.push([k, aepe(f2.u, f2.v, S.GU, S.GV, 0, N)]);
      ser.left.push([k, aepe(f2.u, f2.v, S.GU, S.GV, 0, N / 2)]);
      ser.right.push([k, aepe(f2.u, f2.v, S.GU, S.GV, N / 2, N)]);
    });
    const ix = d3.scaleLog().domain([20, 900]).range([460, 736]);
    const imax = d3.max(ser.all.concat(ser.left, ser.right), d => d[1]) * 1.1 || 1;
    const iy = d3.scaleLinear().domain([0, imax]).range([404, 288]);
    iy.ticks(3).forEach(v => {
      g.append("line").attr("x1", 460).attr("x2", 736).attr("y1", iy(v)).attr("y2", iy(v)).attr("stroke", VC.grid);
      g.append("text").attr("x", 456).attr("y", iy(v) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted).text(VZ.fmt(v, 1));
    });
    [[ser.all, VC.accent], [ser.left, VC.good], [ser.right, VC.bad]].forEach(([cv, col]) =>
      VZ.poly(g, cv.map(d => [ix(d[0]), iy(VZ.clamp(d[1], 0, imax))]), { stroke: col, w: 1.8, close: false }));
    [20, 100, 500].forEach(v => g.append("text").attr("x", ix(v)).attr("y", 418).attr("text-anchor", "middle")
      .attr("font-size", 8.5).attr("fill", VC.muted).text(v));

    ui.out.innerHTML =
      `<span class="keep">α</span> = ${VZ.sig(alpha, 3)}, ${iters} iterations · AEPE <b>${VZ.fmt(eAll, 4)}</b> over the whole field` +
      (scene === "half" ? ` — <b style="color:${VC.good}">${VZ.fmt(eL, 4)}</b> where there is texture, <b style="color:${VC.bad}">${VZ.fmt(eR, 4)}</b> over the flat half` : "") +
      `<br />` + (scene === "half"
        ? `The textured half converges within about a hundred iterations and then stops moving; the flat half is still improving at nine hundred, because the answer has to diffuse across twenty pixels one neighbour at a time. That factor of roughly thirty in convergence time is the reason nobody runs plain Jacobi Horn–Schunck — multigrid, conjugate gradients or a pyramid (§25) all attack exactly this.`
        : scene === "split"
          ? `Two objects with opposite motions. The smoothness term charges (2‖d‖)² for the true discontinuity, so the optimiser prefers a ramp; raise <span class="keep">α</span> and watch the boundary smear further. No amount of iterating fixes it — it is what the model asked for. A robust or total-variation penalty is the fix (§10).`
          : `Textured everywhere, so the fill-in problem disappears and the smoothness term is doing pure noise suppression. Notice that the best <span class="keep">α</span> is now larger than it was with a flat region present: the two jobs the smoothness term does have different optima.`);
  }

  [ui.a, ui.i, ui.s, ui.lk].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ═══ FIG 20 · §25 — single scale against a pyramid ══════════════════════════
   Both methods get the same iteration budget per level. The error curve is
   recomputed live across the whole slider range, so the cliff is measured
   rather than asserted, and the aliasing flag reproduces the classic bug.     */
(function () {
  const svg = d3.select("#c2f-svg");
  if (svg.empty()) return;
  const W = 760, H = 470, N = 96, NB = 208, OFF = (NB - N) >> 1;
  const DIR = [0.8, -0.6];                    // unit direction, so |d| is the slider

  const rn = VZ.rng(50501);
  const raw = VZ.zeros2(NB, NB);
  for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) raw[i][j] = VZ.randn(rn);
  const norm = A => {
    let m = 0, s = 0;
    for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) m += A[i][j];
    m /= NB * NB;
    for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) s += (A[i][j] - m) ** 2;
    s = Math.sqrt(s / (NB * NB)) || 1;
    const B = VZ.zeros2(NB, NB);
    for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) B[i][j] = (A[i][j] - m) / s;
    return B;
  };
  const TEX = {
    fine: norm(VZ.sep2(raw, VZ.gauss1(1.0), "clamp")),
    coarse: norm(VZ.sep2(raw, VZ.gauss1(5.0), "clamp")),
    broad: (function () {
      const a = norm(VZ.sep2(raw, VZ.gauss1(1.2), "clamp")), b = norm(VZ.sep2(raw, VZ.gauss1(3.0), "clamp"));
      const c = VZ.zeros2(NB, NB);
      for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) c[i][j] = a[i][j] + 0.8 * b[i][j];
      return norm(c);
    })()
  };

  const ui = {
    d: document.getElementById("cf-d"), dv: document.getElementById("cf-dv"),
    l: document.getElementById("cf-l"), lv: document.getElementById("cf-lv"),
    t: document.getElementById("cf-t"), alias: document.getElementById("cf-alias"),
    out: document.getElementById("c2f-readout")
  };

  function pair(kind, mag) {
    const img = TEX[kind], d = [DIR[0] * mag, DIR[1] * mag];
    const I0 = VZ.zeros2(N, N), I1 = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      I0[i][j] = 0.5 + 0.12 * img[i + OFF][j + OFF];
      I1[i][j] = 0.5 + 0.12 * AM.bilinear(img, j + OFF - d[0], i + OFF - d[1]);
    }
    return { I0: I0, I1: I1, d: d };
  }
  const usableLevels = L => { let l = 1; let s = N; while (l < L && (s >> 1) >= 12) { s >>= 1; l++; } return l; };

  function solve(kind, mag, L, smooth) {
    const P = pair(kind, mag);
    const single = AM.lkTrans(P.I0, P.I1, [0, 0], 30, 4);
    const multi = AM.coarseToFine(P.I0, P.I1, usableLevels(L), 15, smooth);
    return { d: P.d, single: single, multi: multi, P: P,
             eS: Math.hypot(single[0] - P.d[0], single[1] - P.d[1]),
             eM: Math.hypot(multi[0] - P.d[0], multi[1] - P.d[1]) };
  }

  function draw() {
    const mag = +ui.d.value, L = +ui.l.value, kind = ui.t.value, smooth = !ui.alias.checked;
    ui.dv.textContent = mag + " px"; ui.lv.textContent = L;
    const UL = usableLevels(L);
    const R = solve(kind, mag, L, smooth);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;

    /* ---- the pyramid ------------------------------------------------------- */
    VZ.panelBox(g, 14, 60, 150, 386, "the pyramid of I₀");
    const P0 = AM.pyramid(R.P.I0, UL, smooth);
    let yy = 74;
    P0.forEach((A, l) => {
      const sz = Math.max(26, 118 / Math.pow(1.55, l));
      VZ.raster(g, A, 24, yy, sz, sz, {});
      g.append("text").attr("x", 24 + sz + 8).attr("y", yy + 14).attr("font-size", 10).attr("fill", VC.muted)
        .text(`level ${l} · ${A.length}²`);
      g.append("text").attr("x", 24 + sz + 8).attr("y", yy + 27).attr("font-size", 10)
        .attr("fill", mag / Math.pow(2, l) < 1.5 ? VC.good : VC.a2)
        .text(`d = ${VZ.fmt(mag / Math.pow(2, l), 2)} px here`);
      yy += sz + 12;
    });

    /* ---- velocity space ---------------------------------------------------- */
    VZ.panelBox(g, 190, 60, 240, 240, "where each method ended up");
    const M = Math.max(6, mag * 1.3);
    const x = d3.scaleLinear().domain([-M, M]).range([198, 422]);
    const y = d3.scaleLinear().domain([-M, M]).range([68, 292]);
    const gv = g.append("g").attr("clip-path", VZ.clip(svg, "am-cf-clip", 190, 60, 240, 240));
    x.ticks(5).forEach(v => {
      gv.append("line").attr("x1", 198).attr("x2", 422).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", v === 0 ? VC.line : VC.grid);
      gv.append("line").attr("y1", 68).attr("y2", 292).attr("x1", x(v)).attr("x2", x(v)).attr("stroke", v === 0 ? VC.line : VC.grid);
    });
    gv.append("path").attr("d", `M${x(R.d[0]) - 9},${y(R.d[1])} H${x(R.d[0]) + 9} M${x(R.d[0])},${y(R.d[1]) - 9} V${y(R.d[1]) + 9}`)
      .attr("stroke", VC.good).attr("stroke-width", 2.4);
    VZ.arrow(gv, x(0), y(0), x(R.single[0]), y(R.single[1]), { color: VC.bad, w: 1.6, head: 6 });
    VZ.arrow(gv, x(0), y(0), x(R.multi[0]), y(R.multi[1]), { color: VC.accent, w: 1.6, head: 6 });
    gv.append("circle").attr("cx", x(R.single[0])).attr("cy", y(R.single[1])).attr("r", 4).attr("fill", VC.bad);
    gv.append("circle").attr("cx", x(R.multi[0])).attr("cy", y(R.multi[1])).attr("r", 4).attr("fill", VC.accent);
    VZ.legend(g, [{ color: VC.good, label: "the true displacement" }, { color: VC.bad, label: "single scale" },
                  { color: VC.accent, label: `coarse-to-fine, ${UL} levels` }], 198, 312, { gap: 14, font: 10 });

    /* ---- error against displacement ---------------------------------------- */
    VZ.panelBox(g, 452, 60, 294, 300, "final error vs displacement, both methods");
    const mags = d3.range(1, 41, 3);
    const rows = mags.map(m => solve(kind, m, L, smooth));
    const ex = d3.scaleLinear().domain([0, 40]).range([472, 736]);
    const ey = d3.scaleLog().domain([1e-4, 80]).range([336, 84]).clamp(true);
    [1e-3, 1e-1, 1, 10].forEach(v => {
      g.append("line").attr("x1", 472).attr("x2", 736).attr("y1", ey(v)).attr("y2", ey(v)).attr("stroke", VC.grid);
      g.append("text").attr("x", 468).attr("y", ey(v) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted).text(d3.format(".0e")(v));
    });
    [0, 10, 20, 30, 40].forEach(v => g.append("text").attr("x", ex(v)).attr("y", 350).attr("text-anchor", "middle")
      .attr("font-size", 8.5).attr("fill", VC.muted).text(v));
    VZ.poly(g, rows.map((r, i) => [ex(mags[i]), ey(Math.max(r.eS, 1e-4))]), { stroke: VC.bad, w: 1.8, close: false });
    VZ.poly(g, rows.map((r, i) => [ex(mags[i]), ey(Math.max(r.eM, 1e-4))]), { stroke: VC.accent, w: 1.8, close: false });
    g.append("line").attr("x1", ex(mag)).attr("x2", ex(mag)).attr("y1", 84).attr("y2", 336)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "3 3");
    g.append("text").attr("x", 736).attr("y", 362).attr("text-anchor", "end").attr("font-size", 10).attr("fill", VC.muted).text("‖d‖, pixels →");
    /* the measured cliffs */
    const cliff = key => { for (let i = 0; i < rows.length; i++) if (rows[i][key] > 0.5) return mags[i]; return null; };
    const cS = cliff("eS"), cM = cliff("eM");

    const kv = VZ.kv(g, 466, 386, { keyW: 208, size: 11, lead: 16 });
    kv("single scale, error here", VZ.fmt(R.eS, 4) + " px", R.eS < 0.1 ? VC.good : VC.bad, true);
    kv(`coarse-to-fine (${UL} levels), error here`, VZ.fmt(R.eM, 4) + " px", R.eM < 0.1 ? VC.good : VC.bad, true);
    kv("single scale breaks at", cS === null ? "beyond 40 px" : "‖d‖ ≈ " + cS + " px", VC.bad);
    kv("coarse-to-fine breaks at", cM === null ? "beyond 40 px" : "‖d‖ ≈ " + cM + " px", VC.accent);

    ui.out.innerHTML =
      `‖d‖ = ${mag} px · single scale is <b style="color:${R.eS < 0.1 ? VC.good : VC.bad}">${VZ.fmt(R.eS, 4)} px</b> out, ` +
      `coarse-to-fine over ${UL} levels is <b style="color:${R.eM < 0.1 ? VC.good : VC.bad}">${VZ.fmt(R.eM, 4)} px</b> out.` +
      ` Measured breakdown: single scale at ${cS === null ? "beyond 40" : "≈ " + cS} px, coarse-to-fine at ${cM === null ? "beyond 40" : "≈ " + cM} px.` +
      `<br />` + (!smooth
        ? `<b style="color:${VC.bad}">Decimating without pre-smoothing.</b> The coarse levels now contain aliased structure that is not in the scene, so the coarsest estimate can be confidently wrong and every finer level inherits it. Compare the coarse-to-fine curve with the box unticked — this is one of the few bugs that makes a method worse than not using it.`
        : R.eS > 0.5 && R.eM < 0.1
          ? `The single-scale estimate has not merely lost accuracy — it has converged to a <i>different local minimum</i>, with a small residual and no internal sign of trouble. The pyramid never let the linearisation see more than about a pixel of motion at any level, so there was no wrong basin to fall into.`
          : R.eS < 0.1
            ? `Both work: this displacement is inside the single-scale basin, so the pyramid is buying nothing here. Raise the displacement past the cliff at about ${cS === null ? "40+" : cS} px.`
            : `Both have failed. The coarsest level's displacement is ${VZ.fmt(mag / Math.pow(2, UL - 1), 2)} px, which is itself past the basin — add levels, or accept that this motion needs a feature-based initialisation (§09) rather than a direct one.`);
  }

  [ui.d, ui.l, ui.t, ui.alias].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ═══ FIG 21 · §26 — forward against inverse compositional ═══════════════════
   Both formulations are run on the same pair with the same budget. The residual
   curves lie on top of each other; the work per iteration does not.           */
(function () {
  const svg = d3.select("#ic-svg");
  if (svg.empty()) return;
  const W = 760, H = 430, NB = 168;

  const rn = VZ.rng(31415);
  const raw = VZ.zeros2(NB, NB);
  for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) raw[i][j] = VZ.randn(rn);
  let T = VZ.sep2(raw, VZ.gauss1(1.6), "clamp");
  let m = 0, s = 0;
  for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) m += T[i][j];
  m /= NB * NB;
  for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) s += (T[i][j] - m) ** 2;
  s = Math.sqrt(s / (NB * NB));
  const IMG = VZ.zeros2(NB, NB);
  for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) IMG[i][j] = 0.5 + 0.13 * (T[i][j] - m) / s;

  const ui = {
    m: document.getElementById("ic-m"), s: document.getElementById("ic-s"), sv: document.getElementById("ic-sv"),
    n: document.getElementById("ic-n"), nv: document.getElementById("ic-nv"),
    a: document.getElementById("ic-a"), out: document.getElementById("ic-readout")
  };

  /* p = (a₀₀, a₀₁, t₀, a₁₀, a₁₁, t₁) as a DEVIATION from the identity */
  const asMat = (p, model) => (model === "trans")
    ? [[1, 0, p[0]], [0, 1, p[1]], [0, 0, 1]]
    : [[1 + p[0], p[1], p[2]], [p[3], 1 + p[4], p[5]], [0, 0, 1]];
  const NP = model => model === "trans" ? 2 : 6;

  function run(model, N, Wt, inverse, iters) {
    const off = (NB - N) >> 1;
    const I0 = VZ.zeros2(N, N), I1 = VZ.zeros2(N, N);
    /* target = template warped by Wt, sampled about the patch centre */
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      I0[i][j] = IMG[i + off][j + off];
      const cx = j - (N - 1) / 2, cy = i - (N - 1) / 2;
      const q = VZ.applyH(Wt, [cx, cy]);
      I1[i][j] = AM.bilinear(IMG, q[0] + (N - 1) / 2 + off, q[1] + (N - 1) / 2 + off);
    }
    const n = NP(model), G0 = AM.grads(I0), G1 = AM.grads(I1);
    /* the warp Jacobian at the identity: constant, and this is the whole trick */
    const jac = (x, y) => model === "trans" ? [[1, 0], [0, 1]] : [[x, y, 1, 0, 0, 0], [0, 0, 0, x, y, 1]];
    let p = new Array(n).fill(0), Wcur = VZ.eye(3);
    const hist = [];
    /* precomputed for the inverse form: steepest-descent images and A⁻¹ */
    let SD = null, Ainv = null;
    if (inverse) {
      SD = [];
      const A = [];
      for (let k = 0; k < n; k++) A.push(new Array(n).fill(0));
      for (let k = 0; k < n; k++) SD.push(VZ.zeros2(N, N));
      for (let i = 2; i < N - 2; i++) for (let j = 2; j < N - 2; j++) {
        const cx = j - (N - 1) / 2, cy = i - (N - 1) / 2;
        const J = jac(cx, cy), row = new Array(n).fill(0);
        for (let k = 0; k < n; k++) row[k] = G0.Ix[i][j] * J[0][k] + G0.Iy[i][j] * J[1][k];
        for (let k = 0; k < n; k++) { SD[k][i][j] = row[k]; for (let l = 0; l < n; l++) A[k][l] += row[k] * row[l]; }
      }
      Ainv = VZ.invN(A);
    }
    for (let it = 0; it < iters; it++) {
      /* warp I1 by the current estimate */
      const Wi = VZ.inv3(Wcur);
      if (!Wi) break;
      let ss = 0, cnt = 0;
      const A = [], b = new Array(n).fill(0);
      for (let k = 0; k < n; k++) A.push(new Array(n).fill(0));
      for (let i = 2; i < N - 2; i++) for (let j = 2; j < N - 2; j++) {
        const cx = j - (N - 1) / 2, cy = i - (N - 1) / 2;
        const q = VZ.applyH(Wcur, [cx, cy]);
        const xs = q[0] + (N - 1) / 2, ys = q[1] + (N - 1) / 2;
        if (xs < 1 || xs > N - 2 || ys < 1 || ys > N - 2) continue;
        const w1 = AM.bilinear(I1, xs, ys);
        const e = w1 - I0[i][j];
        ss += e * e; cnt++;
        if (inverse) { for (let k = 0; k < n; k++) b[k] += SD[k][i][j] * e; }
        else {
          const gx = AM.bilinear(G1.Ix, xs, ys), gy = AM.bilinear(G1.Iy, xs, ys);
          const J = jac(cx, cy), row = new Array(n).fill(0);
          for (let k = 0; k < n; k++) row[k] = gx * J[0][k] + gy * J[1][k];
          for (let k = 0; k < n; k++) { b[k] -= row[k] * e; for (let l = 0; l < n; l++) A[k][l] += row[k] * row[l]; }
        }
      }
      hist.push(Math.sqrt(ss / Math.max(1, cnt)));
      let dp;
      if (inverse) { if (!Ainv) break; dp = VZ.mv(Ainv, b); }
      else { const Ai = VZ.invN(A); if (!Ai) break; dp = VZ.mv(Ai, b); }
      if (!dp.every(isFinite)) break;
      const D = asMat(dp, model);
      /* the inverse form's Δp warps the TEMPLATE, so compose its inverse */
      const Dc = inverse ? VZ.inv3(D) : D;
      if (!Dc) break;
      Wcur = VZ.mul(Wcur, Dc);
      const nrm = Math.sqrt(dp.reduce((a2, v) => a2 + v * v, 0));
      if (nrm < 1e-7) break;
    }
    return { hist: hist, Wc: Wcur, I0: I0, I1: I1, SD: SD, n: n, N: N };
  }

  function draw() {
    const model = ui.m.value, sz = +ui.s.value, N = +ui.n.value, which = ui.a.value;
    ui.sv.textContent = sz; ui.nv.textContent = N + "²";
    const k = sz / 100;
    const Wt = model === "trans"
      ? VZ.T2(sz * 0.55, -sz * 0.4)
      : [[1 + 1.4 * k, 0.9 * k, sz * 0.45], [-1.1 * k, 1 - 0.8 * k, -sz * 0.35], [0, 0, 1]];

    const F = run(model, N, Wt, false, 24), I = run(model, N, Wt, true, 24);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    const show = which === "inv" ? I : F;
    const DF = VZ.zeros2(N, N);
    const Wi = VZ.inv3(show.Wc) ? show.Wc : VZ.eye(3);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const cx = j - (N - 1) / 2, cy = i - (N - 1) / 2;
      const q = VZ.applyH(Wi, [cx, cy]);
      const xs = VZ.clamp(q[0] + (N - 1) / 2, 0, N - 1.001), ys = VZ.clamp(q[1] + (N - 1) / 2, 0, N - 1.001);
      DF[i][j] = AM.bilinear(show.I1, xs, ys) - show.I0[i][j];
    }
    [["template I₀", show.I0, 56, {}], ["target I₁", show.I1, 182, {}], ["difference after alignment", DF, 308, { signed: true, lo: -0.35, hi: 0.35 }]]
      .forEach(([lab, A, yy, opt]) => { VZ.panelBox(g, 14, yy, 104, 104, lab); VZ.raster(g, A, 14, yy, 104, 104, opt); });

    /* ---- residual curves --------------------------------------------------- */
    VZ.panelBox(g, 150, 56, 292, 230, "RMS residual vs iteration");
    const mx = Math.max(F.hist.length, I.hist.length);
    const x = d3.scaleLinear().domain([0, Math.max(1, mx - 1)]).range([172, 434]);
    const lo = Math.max(1e-6, d3.min(F.hist.concat(I.hist)) * 0.7), hi = d3.max(F.hist.concat(I.hist)) * 1.4;
    const y = d3.scaleLog().domain([lo, hi]).range([272, 74]).clamp(true);
    y.ticks(4).forEach(v => {
      g.append("line").attr("x1", 172).attr("x2", 434).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", VC.grid);
      g.append("text").attr("x", 168).attr("y", y(v) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted).text(d3.format(".1e")(v));
    });
    if (which !== "inv") VZ.poly(g, F.hist.map((v, i) => [x(i), y(v)]), { stroke: VC.a2, w: 2.4, close: false });
    if (which !== "fwd") VZ.poly(g, I.hist.map((v, i) => [x(i), y(v)]), { stroke: VC.accent, w: 1.6, close: false, dash: "5 3" });
    VZ.legend(g, [{ color: VC.a2, label: "forward compositional" }, { color: VC.accent, label: "inverse compositional", dash: "5 3" }], 178, 88, { gap: 14, font: 10 });
    g.append("text").attr("x", 434).attr("y", 292).attr("text-anchor", "end").attr("font-size", 10).attr("fill", VC.muted).text("iteration →");

    /* ---- the steepest-descent images --------------------------------------- */
    VZ.panelBox(g, 150, 316, 292, 100, "the precomputed steepest-descent images ∇I₀ · ∂x̃/∂p");
    if (I.SD) {
      const cw = Math.min(46, 284 / I.n - 4);
      I.SD.forEach((A, k2) => VZ.raster(g, A, 156 + k2 * (cw + 4), 326, cw, cw, { signed: true }));
    }

    /* ---- numbers ----------------------------------------------------------- */
    VZ.panelBox(g, 458, 56, 288, 360, "the same answer, different work");
    const kv = VZ.kv(g, 470, 88, { keyW: 176, size: 11, lead: 17 });
    const nn = NP(model), Npx = (N - 4) * (N - 4);
    /* the target is I₁ = I₀ ∘ W, so the warp that ALIGNS them is W⁻¹ — that is what
       the iteration converges to, and it is what the error must be measured against */
    const Wtinv = VZ.inv3(Wt) || VZ.eye(3);
    const err = Wc => { let e = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) e = Math.max(e, Math.abs(Wc[i][j] - Wtinv[i][j])); return e; };
    kv("parameters n", String(nn));
    kv("pixels N", String(Npx));
    kv("forward: iterations", String(F.hist.length), VC.a2);
    kv("forward: final RMS", F.hist.length ? F.hist[F.hist.length - 1].toExponential(3) : "—", VC.a2);
    kv("forward: max |Ŵ − W⁻¹|", err(F.Wc).toExponential(2), err(F.Wc) < 0.02 ? VC.good : VC.a2);
    kv("inverse: iterations", String(I.hist.length), VC.accent);
    kv("inverse: final RMS", I.hist.length ? I.hist[I.hist.length - 1].toExponential(3) : "—", VC.accent);
    kv("inverse: max |Ŵ − W⁻¹|", err(I.Wc).toExponential(2), err(I.Wc) < 0.02 ? VC.good : VC.accent);
    kv("", "");
    kv("forward work / iteration", d3.format(",")(Math.round(nn * nn * Npx)) + "  ~ n²N", VC.a2);
    kv("inverse work / iteration", d3.format(",")(Math.round(nn * Npx)) + "  ~ nN", VC.accent, true);
    kv("ratio", VZ.fmt(nn, 0) + "×", VC.good, true);

    ui.out.innerHTML =
      `${model === "trans" ? "translation, 2 parameters" : "affine, 6 parameters"} on a ${N} × ${N} patch · ` +
      `forward compositional converged in <b>${F.hist.length}</b> iterations to ${F.hist.length ? F.hist[F.hist.length - 1].toExponential(2) : "—"}, ` +
      `inverse compositional in <b>${I.hist.length}</b> to ${I.hist.length ? I.hist[I.hist.length - 1].toExponential(2) : "—"}` +
      ` · recovered warps within ${err(F.Wc).toExponential(1)} and ${err(I.Wc).toExponential(1)} of the true W⁻¹.` +
      `<br />The two curves lie on top of each other because the formulations are equivalent to first order — the whole difference is that the inverse form's Hessian, its inverse, and the ${nn} steepest-descent images below were computed <i>once</i>, before the loop, from the template alone. That turns O(n²N) = ${d3.format(",")(Math.round(nn * nn * Npx))} multiply-adds per iteration into O(nN) = ${d3.format(",")(Math.round(nn * Npx))}, a factor of ${nn}, and the factor is n.`;
  }

  [ui.m, ui.s, ui.n, ui.a].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ── AM.blockMatch1D: dense horizontal correspondence by a box-filtered search ──
   The scene of §27 moves horizontally, so an exhaustive search over integer shifts
   with a parabolic sub-pixel refinement is both exact within its range and free of
   any basin-of-attraction question — which is what this figure needs, because the
   subject here is the CONSISTENCY CHECK, not the estimator. The window sums are done
   with a separable box (VZ.box + VZ.sep2), i.e. the moving-average trick of §22.   */
Object.assign(AM, {
  blockMatch1D: function (I0, I1, R, maxShift) {
    const H = I0.length, W = I0[0].length, box = VZ.box(2 * R + 1);
    const best = VZ.zeros2(H, W), bestD = VZ.zeros2(H, W);
    const cost = [];
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) best[i][j] = Infinity;
    for (let d = -maxShift; d <= maxShift; d++) {
      const D = VZ.zeros2(H, W);
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
        const e = AM.bilinear(I1, VZ.clamp(j + d, 0, W - 1.001), i) - I0[i][j];
        D[i][j] = e * e;
      }
      const S = VZ.sep2(D, box, "clamp");
      cost.push(S);
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++)
        if (S[i][j] < best[i][j]) { best[i][j] = S[i][j]; bestD[i][j] = d; }
    }
    /* parabolic refinement through the three costs around the winner */
    const u = VZ.zeros2(H, W);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
      const k = bestD[i][j] + maxShift;
      if (k <= 0 || k >= cost.length - 1) { u[i][j] = bestD[i][j]; continue; }
      const cm = cost[k - 1][i][j], c0 = cost[k][i][j], cp = cost[k + 1][i][j];
      const den = cm - 2 * c0 + cp;
      u[i][j] = bestD[i][j] + (Math.abs(den) > 1e-14 ? VZ.clamp(0.5 * (cm - cp) / den, -0.5, 0.5) : 0);
    }
    return { u: u, cost: best };
  }
});

/* ═══ FIG 22 · §27 — occlusion and the consistency check ═════════════════════ */
(function () {
  const svg = d3.select("#occl-svg");
  if (svg.empty()) return;
  const W = 760, H = 440, N = 72, PAD = 20, NB = N + 2 * PAD, SQ = 15;

  const rn = VZ.rng(606060);
  function tex(sig, seed) {
    const r2 = VZ.rng(seed), A = VZ.zeros2(NB, NB);
    for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) A[i][j] = VZ.randn(r2);
    const B = VZ.sep2(A, VZ.gauss1(sig), "clamp");
    let m = 0, s = 0;
    for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) m += B[i][j];
    m /= NB * NB;
    for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) s += (B[i][j] - m) ** 2;
    s = Math.sqrt(s / (NB * NB)) || 1;
    const C = VZ.zeros2(NB, NB);
    for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) C[i][j] = (B[i][j] - m) / s;
    return C;
  }
  const BG = tex(1.4, 11), FG = tex(1.1, 77);

  const ui = {
    d: document.getElementById("oc-d"), dv: document.getElementById("oc-dv"),
    b: document.getElementById("oc-b"), bv: document.getElementById("oc-bv"),
    w: document.getElementById("oc-w"), wv: document.getElementById("oc-wv"),
    t: document.getElementById("oc-t"), tv: document.getElementById("oc-tv"),
    out: document.getElementById("occl-readout")
  };

  const inSq = (x, y, sx) => Math.abs(x - (N / 2 + sx)) <= SQ && Math.abs(y - N / 2) <= SQ;

  function frameAt(sx, bx) {
    const A = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      A[i][j] = inSq(j, i, sx)
        ? 0.62 + 0.13 * AM.bilinear(FG, j - sx + PAD, i + PAD)
        : 0.38 + 0.13 * AM.bilinear(BG, j - bx + PAD, i + PAD);
    }
    return A;
  }

  function draw() {
    const d = +ui.d.value, bmv = +ui.b.value, R = +ui.w.value, thr = +ui.t.value;
    ui.dv.textContent = d + " px"; ui.bv.textContent = bmv + " px";
    ui.wv.textContent = R + " px"; ui.tv.textContent = VZ.fmt(thr, 2) + " px";

    const I0 = frameAt(0, 0), I1 = frameAt(d, bmv);
    const GU = VZ.zeros2(N, N), OCC = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const onSq = inSq(j, i, 0);
      GU[i][j] = onSq ? d : bmv;
      /* a background pixel is occluded if its destination is covered by the square */
      OCC[i][j] = (!onSq && inSq(j + bmv, i, d)) ? 1 : 0;
    }
    const MS = 15;
    const F = AM.blockMatch1D(I0, I1, R, MS), Bk = AM.blockMatch1D(I1, I0, R, MS);
    const CON = VZ.zeros2(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const xs = VZ.clamp(j + F.u[i][j], 0, N - 1.001);
      CON[i][j] = Math.abs(F.u[i][j] + AM.bilinear(Bk.u, xs, i));
    }
    let nk = 0, sk = 0, sa = 0, na = 0, nocc = 0, noccRej = 0;
    for (let i = 4; i < N - 4; i++) for (let j = 4; j < N - 4; j++) {
      const e = Math.abs(F.u[i][j] - GU[i][j]);
      sa += e; na++;
      if (OCC[i][j]) { nocc++; if (CON[i][j] >= thr) noccRej++; }
      if (CON[i][j] < thr) { sk += e; nk++; }
    }
    const aepeAll = sa / Math.max(1, na), aepeKept = nk ? sk / nk : NaN, rej = 1 - nk / Math.max(1, na);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;
    VZ.panelBox(g, 14, 56, 150, 150, "frame 0");
    VZ.raster(g, I0, 14, 56, 150, 150, { lo: 0.2, hi: 0.85 });
    VZ.panelBox(g, 14, 240, 150, 150, "frame 1");
    VZ.raster(g, I1, 14, 240, 150, 150, { lo: 0.2, hi: 0.85 });

    VZ.panelBox(g, 194, 56, 190, 190, "recovered flow · hue = direction");
    const mx = Math.max(2, Math.max(Math.abs(d), Math.abs(bmv)) * 1.3);
    VZ.cells(g, 194, 56, 190 / N, N, N, (jx, iy) => AM.flowColor(F.u[iy][jx], 0, mx));
    VZ.panelBox(g, 194, 276, 190, 130, "ground truth, same scale");
    VZ.cells(g, 194, 276, 190 / N, N, 44, (jx, iy) => AM.flowColor(GU[iy + 14][jx], 0, mx));

    VZ.panelBox(g, 414, 56, 190, 190, "forward–backward residual");
    VZ.raster(g, CON, 414, 56, 190, 190, { lo: 0, hi: Math.max(1, thr * 3) });
    VZ.panelBox(g, 414, 276, 190, 130, "kept (bright) vs rejected");
    VZ.cells(g, 414, 276, 190 / N, N, 44, (jx, iy) => CON[iy + 14][jx] < thr ? "#e6e9ef" : "#7f1d1d");

    VZ.panelBox(g, 626, 56, 120, 350, "measured");
    const kv = VZ.kv(g, 636, 88, { keyW: 0, size: 10.5, lead: 15 });
    const line = (t, col) => { kv(t, "", col); };
    [["rejected", VZ.fmt(100 * rej, 1) + "%", VC.a2],
     ["true occluded", VZ.fmt(100 * nocc / Math.max(1, na), 1) + "%", VC.muted],
     ["of those, caught", nocc ? VZ.fmt(100 * noccRej / nocc, 1) + "%" : "—", VC.good],
     ["AEPE, all pixels", VZ.fmt(aepeAll, 4) + " px", VC.bad],
     ["AEPE, kept only", VZ.fmt(aepeKept, 4) + " px", VC.good],
     ["improvement", VZ.fmt(aepeAll / Math.max(aepeKept, 1e-9), 2) + "×", VC.good]].forEach(([a, b, col], q) => {
      g.append("text").attr("x", 636).attr("y", 88 + q * 34).attr("font-size", 10).attr("fill", VC.muted).text(a);
      g.append("text").attr("x", 636).attr("y", 102 + q * 34).attr("font-size", 12.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col).attr("font-weight", 600).text(b);
    });

    ui.out.innerHTML =
      `square moves ${d} px, background ${bmv} px — a relative displacement of ${Math.abs(d - bmv)} px, which is the width of the occluded band.` +
      ` The check rejects <b>${VZ.fmt(100 * rej, 1)}%</b> of pixels and catches <b>${nocc ? VZ.fmt(100 * noccRej / nocc, 1) : "—"}%</b> of the truly occluded ones.` +
      `<br />AEPE over all pixels <b style="color:${VC.bad}">${VZ.fmt(aepeAll, 4)} px</b>, over the kept pixels <b style="color:${VC.good}">${VZ.fmt(aepeKept, 4)} px</b> — ` +
      (aepeAll / Math.max(aepeKept, 1e-9) > 1.3
        ? `a factor of ${VZ.fmt(aepeAll / aepeKept, 2)}. Almost all the error lives in a band whose width is the relative displacement, and it lives there because those pixels have <i>no correct answer</i>, not because the estimator was inaccurate.`
        : `barely different, because at this setting there is little or no occlusion to find. Increase the relative motion between the square and the background.`);
  }

  [ui.d, ui.b, ui.w, ui.t].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();

/* ═══ FIG 23 · §28 — two metrics, one error ══════════════════════════════════
   The error magnitude is held fixed while the flow magnitude sweeps. EPE is a
   horizontal line; the angular error falls by two decades. That is the whole
   argument for reporting the first and not the second.                        */
(function () {
  const svg = d3.select("#epe-svg");
  if (svg.empty()) return;
  const W = 760, H = 430, NF = 40;

  const rn = VZ.rng(818181);
  const PERT = d3.range(NF * NF).map(() => [VZ.randn(rn), VZ.randn(rn), rn()]);

  const ui = {
    e: document.getElementById("ep-e"), ev: document.getElementById("ep-ev"),
    m: document.getElementById("ep-m"), mv: document.getElementById("ep-mv"),
    d: document.getElementById("ep-d"), x: document.getElementById("ep-x"), xv: document.getElementById("ep-xv"),
    out: document.getElementById("epe-readout")
  };

  const ae = (u, v, ug, vg) => {
    const a = [u, v, 1], b = [ug, vg, 1];
    const c = VZ.dot(a, b) / (VZ.norm(a) * VZ.norm(b));
    return VZ.deg(Math.acos(VZ.clamp(c, -1, 1)));
  };

  function draw() {
    const err = +ui.e.value, mag = Math.pow(10, +ui.m.value), dir = ui.d.value, xth = +ui.x.value;
    ui.ev.textContent = VZ.fmt(err, 2) + " px"; ui.mv.textContent = VZ.fmt(mag, 2) + " px";
    ui.xv.textContent = VZ.fmt(xth, 2) + " px";

    const gt = [mag, 0];
    const est = dir === "along" ? [mag + err, 0] : [mag, err];
    const epe = Math.hypot(est[0] - gt[0], est[1] - gt[1]);
    const ang = ae(est[0], est[1], gt[0], gt[1]);

    const fr = VZ.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const g = fr.g;

    /* ---- velocity space ---------------------------------------------------- */
    VZ.panelBox(g, 14, 56, 210, 210, "the two vectors");
    const M = Math.max(mag * 1.4, err * 3, 0.4);
    const x = d3.scaleLinear().domain([-0.15 * M, M]).range([26, 214]);
    const y = d3.scaleLinear().domain([-0.5 * M, 0.5 * M]).range([250, 74]);
    g.append("line").attr("x1", 26).attr("x2", 214).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", VC.grid);
    g.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 74).attr("y2", 250).attr("stroke", VC.grid);
    VZ.arrow(g, x(0), y(0), x(gt[0]), y(gt[1]), { color: VC.good, w: 2.2, head: 7 });
    VZ.arrow(g, x(0), y(0), x(est[0]), y(est[1]), { color: VC.a2, w: 2.2, head: 7 });
    g.append("line").attr("x1", x(gt[0])).attr("y1", y(gt[1])).attr("x2", x(est[0])).attr("y2", y(est[1]))
      .attr("stroke", VC.bad).attr("stroke-width", 2.4);
    VZ.legend(g, [{ color: VC.good, label: "ground truth" }, { color: VC.a2, label: "estimate" },
                  { color: VC.bad, label: "endpoint error" }], 30, 88, { gap: 13, font: 9.5 });

    /* ---- the sweep --------------------------------------------------------- */
    VZ.panelBox(g, 246, 56, 250, 210, "both metrics vs flow magnitude, error fixed");
    const sx = d3.scaleLog().domain([0.04, 32]).range([266, 486]);
    const sy = d3.scaleLog().domain([1e-3, 40]).range([250, 74]).clamp(true);
    [1e-2, 1e-1, 1, 10].forEach(v => {
      g.append("line").attr("x1", 266).attr("x2", 486).attr("y1", sy(v)).attr("y2", sy(v)).attr("stroke", VC.grid);
      g.append("text").attr("x", 262).attr("y", sy(v) + 3).attr("text-anchor", "end").attr("font-size", 8.5).attr("fill", VC.muted).text(d3.format(".0e")(v));
    });
    [0.1, 1, 10].forEach(v => {
      g.append("line").attr("y1", 74).attr("y2", 250).attr("x1", sx(v)).attr("x2", sx(v)).attr("stroke", VC.grid);
      g.append("text").attr("x", sx(v)).attr("y", 264).attr("text-anchor", "middle").attr("font-size", 8.5).attr("fill", VC.muted).text(v);
    });
    const ms = VZ.linspace(Math.log(0.04), Math.log(32), 60).map(Math.exp);
    VZ.poly(g, ms.map(m => [sx(m), sy(err)]), { stroke: VC.bad, w: 2, close: false });
    VZ.poly(g, ms.map(m => {
      const e2 = dir === "along" ? [m + err, 0] : [m, err];
      return [sx(m), sy(VZ.clamp(ae(e2[0], e2[1], m, 0), 1e-3, 40))];
    }), { stroke: VC.violet, w: 2, close: false });
    g.append("line").attr("x1", sx(VZ.clamp(mag, 0.04, 32))).attr("x2", sx(VZ.clamp(mag, 0.04, 32)))
      .attr("y1", 74).attr("y2", 250).attr("stroke", VC.a2).attr("stroke-dasharray", "3 3");
    VZ.legend(g, [{ color: VC.bad, label: "endpoint error, px" }, { color: VC.violet, label: "angular error, degrees" }], 272, 88, { gap: 13, font: 9.5 });

    /* ---- a field, its EPE map and histogram -------------------------------- */
    VZ.panelBox(g, 518, 56, 100, 100, "a flow field");
    VZ.panelBox(g, 640, 56, 100, 100, "its EPE map");
    const GU = VZ.zeros2(NF, NF), GV = VZ.zeros2(NF, NF), EU = VZ.zeros2(NF, NF), EV = VZ.zeros2(NF, NF);
    const EM = VZ.zeros2(NF, NF), all = [];
    for (let i = 0; i < NF; i++) for (let j = 0; j < NF; j++) {
      const k = i * NF + j;
      const cx = (j - NF / 2) / NF, cy = (i - NF / 2) / NF;
      GU[i][j] = mag * (0.6 + 1.4 * (0.5 + cx)); GV[i][j] = mag * 1.2 * cy;
      /* mostly small perturbations, with a few gross outliers — the realistic shape */
      const gross = PERT[k][2] < 0.04 ? 12 : 1;
      EU[i][j] = GU[i][j] + err * gross * PERT[k][0];
      EV[i][j] = GV[i][j] + err * gross * PERT[k][1];
      EM[i][j] = Math.hypot(EU[i][j] - GU[i][j], EV[i][j] - GV[i][j]);
      all.push(EM[i][j]);
    }
    const fmx = d3.max([d3.max(GU, r => d3.max(r)), 0.5]);
    VZ.cells(g, 518, 56, 100 / NF, NF, NF, (jx, iy) => AM.flowColor(GU[iy][jx], GV[iy][jx], fmx * 1.2));
    VZ.raster(g, EM, 640, 56, 100, 100, { lo: 0, hi: Math.max(xth * 2, 0.5) });

    VZ.panelBox(g, 518, 196, 222, 130, "EPE histogram, log count");
    const hx = d3.scaleLinear().domain([0, Math.max(d3.max(all), xth * 1.4)]).range([528, 734]);
    const bins = d3.histogram().domain(hx.domain()).thresholds(26)(all);
    const hy = d3.scaleLog().domain([0.8, Math.max(2, d3.max(bins, b => b.length))]).range([314, 206]).clamp(true);
    bins.forEach(b => {
      if (!b.length) return;
      g.append("rect").attr("x", hx(b.x0)).attr("y", hy(b.length))
        .attr("width", Math.max(1, hx(b.x1) - hx(b.x0) - 1)).attr("height", Math.max(1, 314 - hy(b.length)))
        .attr("fill", b.x0 >= xth ? VC.bad : VC.accent).attr("fill-opacity", 0.85);
    });
    g.append("line").attr("x1", hx(xth)).attr("x2", hx(xth)).attr("y1", 206).attr("y2", 314)
      .attr("stroke", VC.a2).attr("stroke-dasharray", "3 3");

    const aepe = d3.mean(all);
    const rx = 100 * all.filter(v => v > xth).length / all.length;
    let fl = 0, n2 = 0;
    for (let i = 0; i < NF; i++) for (let j = 0; j < NF; j++) {
      const gm = Math.hypot(GU[i][j], GV[i][j]);
      if (EM[i][j] > 3 && EM[i][j] > 0.05 * gm) fl++;
      n2++;
    }
    const kv = VZ.kv(g, 528, 348, { keyW: 128, size: 11, lead: 16 });
    kv("AEPE", VZ.fmt(aepe, 4) + " px", VC.accent, true);
    kv(`R${VZ.fmt(xth, 2)} (EPE > ${VZ.fmt(xth, 2)} px)`, VZ.fmt(rx, 2) + " %", VC.bad, true);
    kv("Fl (>3 px and >5%)", VZ.fmt(100 * fl / n2, 2) + " %", VC.a2);

    ui.out.innerHTML =
      `flow ‖u‖ = ${VZ.fmt(mag, 3)} px, an error of ${VZ.fmt(err, 3)} px ${dir === "along" ? "along" : "across"} it · ` +
      `endpoint error <b style="color:${VC.bad}">${VZ.fmt(epe, 4)} px</b>, angular error <b style="color:${VC.violet}">${VZ.fmt(ang, 4)}°</b>.` +
      `<br />Hold the error slider still and sweep the flow magnitude: the endpoint error does not move at all, while the angular error goes from ` +
      `${VZ.fmt(ae(0.1 + err, 0, 0.1, 0), 3)}° at ‖u‖ = 0.1 px to ${VZ.fmt(ae(10 + err, 0, 10, 0), 4)}° at ‖u‖ = 10 px — a factor of ` +
      `${VZ.fmt(ae(0.1 + err, 0, 0.1, 0) / Math.max(ae(10 + err, 0, 10, 0), 1e-9), 0)} for an identical mistake. ` +
      `The field on the right has 4% gross outliers: its AEPE is ${VZ.fmt(aepe, 3)} px while ${VZ.fmt(rx, 1)}% of pixels exceed ${VZ.fmt(xth, 2)} px, and those two numbers move almost independently.`;
  }

  [ui.e, ui.m, ui.d, ui.x].forEach(el => { el.addEventListener("input", draw); el.addEventListener("change", draw); });
  draw();
})();
