/* image-formation.viz.js — the eighteen visualizations on vision/image-formation.html.
   Loaded after ../data.js → ../notes.js → vision-viz.js.
   Each block is an IIFE that exits quietly if its container is not on the page.

     1  #homog-svg  a point of the plane IS a ray through the origin: drag the ray
                    flat and the pierce point escapes; slide λ and it does not move
     2  #hier-svg   one rectangle under all five levels of the 2D hierarchy, with
                    lengths, angles, parallelism, area and cross ratio measured live
     3  #gimbal-svg three nested Euler rings; at pitch = 90° two of them coincide
                    and one whole degree of freedom is gone — the default IS locked
     4  #rod-svg    Rodrigues by construction: v∥, v⊥, v× and the rotated u, with a
                    draggable axis and the matrix printed
     5  #slerp-svg  slerp against naive Euler-angle interpolation between the same
                    two orientations, plus the double cover as a switchable detour
     6  #projfam-svg a top-down comparison of orthographic, scaled orthographic,
                    para-perspective and perspective on the same scene
     7  #persp-svg  divide-by-z, vanishing points and the horizon, on draggable rails
     8  #intr-svg   the five parameters of K acting on one image: f, cₓ, c_y, aspect, skew
     9  #fov-svg    focal length ↔ field of view ↔ sensor size, and the dolly zoom
    10  #cam-svg    the flagship: orbit the camera round a 3D scene and watch the
                    image, with P = K[R|t] recomputed every frame
    11  #dist-svg   radial and tangential distortion, the undistortion that inverts
                    it, and the exact radius at which the inverse stops existing
    12  #lens-svg   the thin lens: focus, circle of confusion, f-number, depth of field
    13  #brdf-svg   Phong lobes in polar coordinates: ambient, diffuse, specular
    14  #shade-svg  a Lambertian + specular sphere with a movable light
    15  #cie-svg    colour matching functions, the chromaticity diagram and a gamut
    16  #gamma-svg  gamma encoding and where the 8-bit code values actually go
    17  #bayer-svg  the Bayer mosaic, bilinear demosaicing and the zipper artefact
    18  #noise-svg  photon shot noise, read noise, gain, and SNR against exposure

   Every number these print is recomputed from the geometry they draw, so the
   pictures and the prose cannot drift apart.                                    */

/* ══════════ page-local helpers (deliberately NOT in vision-viz.js) ══════════ */
const IF = (function () {

  /* format a matrix as monospace <tspan> rows, for printing K, R or P inside an svg */
  function matText(g, M, x, y, opt) {
    const o = Object.assign({ size: 10.5, dp: 3, fill: VC.ink, lead: 13, label: null, pad: 8 }, opt || {});
    const rows = M.map(r => r.map(v => {
      const s = (Math.abs(v) < 5e-7) ? "0" : v.toFixed(o.dp);
      return s.padStart(o.pad);
    }).join(" "));
    const br = ["⎡", "⎢", "⎣"];
    const bl = ["⎤", "⎥", "⎦"];
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    if (o.label) gg.append("text").attr("x", 0).attr("y", -o.lead).attr("font-size", 10)
      .attr("fill", VC.muted).text(o.label);
    rows.forEach((r, i) => {
      const b = rows.length === 1 ? ["[", "]"] : [br[i === 0 ? 0 : (i === rows.length - 1 ? 2 : 1)],
      bl[i === 0 ? 0 : (i === rows.length - 1 ? 2 : 1)]];
      gg.append("text").attr("x", 0).attr("y", i * o.lead)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("font-size", o.size)
        /* SVG collapses runs of whitespace in <text> unless told not to, which would
           throw away the padStart alignment and leave a ragged matrix */
        .attr("xml:space", "preserve")
        .attr("fill", o.fill).text(b[0] + r + " " + b[1]);
    });
    return gg;
  }

  /* a dashed rectangle standing for the image sensor, in screen coordinates */
  function sensorRect(g, x, y, w, h, opt) {
    const o = Object.assign({ stroke: VC.muted, dash: "4 3", fill: "none", op: 1 }, opt || {});
    return g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h)
      .attr("fill", o.fill).attr("fill-opacity", o.fill === "none" ? 0 : 0.5)
      .attr("stroke", o.stroke).attr("stroke-dasharray", o.dash).attr("stroke-opacity", o.op);
  }

  /* draw a 3D polyline through view.project, optionally with depth culling */
  function path3(g, view, pts, opt) {
    const o = Object.assign({ stroke: VC.accent, w: 1.4, dash: null, op: 1, close: false }, opt || {});
    const P = pts.map(p => view.project(p));
    if (P.some(p => p.z <= 0.05)) return g.append("path");     // behind the preview camera
    const d = P.map((p, i) => (i ? "L" : "M") + p.x.toFixed(2) + "," + p.y.toFixed(2)).join(" ") + (o.close ? " Z" : "");
    const el = g.append("path").attr("d", d).attr("fill", "none")
      .attr("stroke", o.stroke).attr("stroke-width", o.w).attr("stroke-opacity", o.op);
    if (o.dash) el.attr("stroke-dasharray", o.dash);
    return el;
  }
  function quad3(g, view, pts, opt) {
    const o = Object.assign({ fill: VC.accent, op: 0.14, stroke: null, w: 1 }, opt || {});
    const P = pts.map(p => view.project(p));
    if (P.some(p => p.z <= 0.05)) return g.append("path");
    const d = P.map((p, i) => (i ? "L" : "M") + p.x.toFixed(2) + "," + p.y.toFixed(2)).join(" ") + " Z";
    const el = g.append("path").attr("d", d).attr("fill", o.fill).attr("fill-opacity", o.op);
    if (o.stroke) el.attr("stroke", o.stroke).attr("stroke-width", o.w);
    return el;
  }
  function dot3(g, view, p, opt) {
    const o = Object.assign({ r: 4, fill: VC.a2, stroke: null }, opt || {});
    const q = view.project(p);
    if (q.z <= 0.05) return g.append("circle");
    const el = g.append("circle").attr("cx", q.x).attr("cy", q.y).attr("r", o.r).attr("fill", o.fill);
    if (o.stroke) el.attr("stroke", o.stroke).attr("stroke-width", 1);
    return el;
  }

  /* one wireframe "house" used by several of the camera figures: a box with a roof,
     given in the series' world frame (X right, Y forward, Z up)                    */
  function house(x0, y0, s, h) {
    const V = [
      [x0 - s, y0 - s, 0], [x0 + s, y0 - s, 0], [x0 + s, y0 + s, 0], [x0 - s, y0 + s, 0],
      [x0 - s, y0 - s, h], [x0 + s, y0 - s, h], [x0 + s, y0 + s, h], [x0 - s, y0 + s, h],
      [x0, y0 - s, h + s * 0.9], [x0, y0 + s, h + s * 0.9]
    ];
    const E = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 8], [6, 9], [7, 9], [8, 9]];
    return { verts: V, edges: E };
  }

  return { matText, sensorRect, path3, quad3, dot3, house };
})();

/* ───────── 1 · a point of the plane is a ray through the origin ───────── */
(function () {
  const svg = d3.select("#homog-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 440;
  const eEl = document.getElementById("hg-el"), eElv = document.getElementById("hg-elv");
  const eAz = document.getElementById("hg-az"), eAzv = document.getElementById("hg-azv");
  const eLam = document.getElementById("hg-lam"), eLamv = document.getElementById("hg-lamv");
  const eYaw = document.getElementById("hg-yaw");
  const ePar = document.getElementById("hg-par"), eW0 = document.getElementById("hg-w0");
  const out = document.getElementById("homog-readout");

  const LIM = 2.6;                       // half-width of the drawn patch of the w = 1 plane
  const M = 3.2;                         // half-width of the right-hand 2D viewport

  function draw() {
    const psi = VZ.rad(+eEl.value), phi = VZ.rad(+eAz.value);
    const lam = +eLam.value, yaw = VZ.rad(+eYaw.value);

    /* the ray direction. Elevation is measured from the w = 0 plane, so w̃ = sin ψ. */
    const d = [Math.cos(psi) * Math.cos(phi), Math.cos(psi) * Math.sin(phi), Math.sin(psi)];
    const finite = Math.abs(d[2]) > 1e-6;
    const pierce = finite ? [d[0] / d[2], d[1] / d[2], 1] : null;      // cot ψ · (cos φ, sin φ, 1)
    const rho = finite ? Math.hypot(pierce[0], pierce[1]) : Infinity;  // = cot ψ

    const f = VZ.frame(svg, W, H, { l: 8, r: 8, t: 8, b: 8 });
    const g = f.g;
    const leftW = 452, rightW = f.iw - leftW - 16;

    /* ─── left: the ℝ³ picture ─── */
    const gl = g.append("g");
    const view = VZ.view3({
      target: [0, 0, 0.35], yaw: yaw, pitch: VZ.rad(18), dist: 8.2,
      f: 330, cx: leftW / 2, cy: f.ih / 2 + 10
    });
    gl.append("rect").attr("x", 0).attr("y", 0).attr("width", leftW).attr("height", f.ih)
      .attr("rx", 8).attr("fill", VC.bg).attr("fill-opacity", 0.35).attr("stroke", VC.line);

    /* the w̃ = 0 plane, faint, drawn first so everything sits on top of it */
    if (eW0.checked) {
      IF.quad3(gl, view, [[-LIM, -LIM, 0], [LIM, -LIM, 0], [LIM, LIM, 0], [-LIM, LIM, 0]],
        { fill: VC.violet, op: 0.07 });
      for (let i = -2; i <= 2; i++) {
        IF.path3(gl, view, [[-LIM, i * 1.3, 0], [LIM, i * 1.3, 0]], { stroke: VC.violet, w: 0.6, op: 0.4 });
        IF.path3(gl, view, [[i * 1.3, -LIM, 0], [i * 1.3, LIM, 0]], { stroke: VC.violet, w: 0.6, op: 0.4 });
      }
      const lab0 = view.project([LIM, LIM, 0]);
      if (lab0.z > 0) gl.append("text").attr("x", lab0.x + 4).attr("y", lab0.y + 12)
        .attr("font-size", 10).attr("fill", VC.violet).text("w̃ = 0  (the line at infinity)");
    }

    /* two parallel lines, each lifted to a plane through the origin */
    if (ePar.checked) {
      const m = 0.55;
      [[0.0, VC.good], [1.1, VC.teal]].forEach(([c, col]) => {
        const A = [-LIM, -LIM * m + c, 1], B = [LIM, LIM * m + c, 1];
        const s = 1.35;
        IF.quad3(gl, view, [VZ.scale(A, s), VZ.scale(B, s), VZ.scale(A, -s), VZ.scale(B, -s)],
          { fill: col, op: 0.13 });
        IF.path3(gl, view, [A, B], { stroke: col, w: 2 });
      });
      /* the two planes meet along the common direction, which lies in w̃ = 0 */
      const dir = VZ.unit([1, m, 0]);
      IF.path3(gl, view, [VZ.scale(dir, -2.4), VZ.scale(dir, 2.4)], { stroke: VC.a2, w: 2.4, dash: "6 3" });
      const dl = view.project(VZ.scale(dir, 2.4));
      if (dl.z > 0) gl.append("text").attr("x", dl.x + 5).attr("y", dl.y + 3)
        .attr("font-size", 10).attr("fill", VC.a2).text("they meet here — an ideal point");
    }

    /* the plane w̃ = 1, as a grid */
    IF.quad3(gl, view, [[-LIM, -LIM, 1], [LIM, -LIM, 1], [LIM, LIM, 1], [-LIM, LIM, 1]],
      { fill: VC.accent, op: 0.09, stroke: VC.accent, w: 1 });
    for (let i = -2; i <= 2; i++) {
      IF.path3(gl, view, [[-LIM, i * 1.3, 1], [LIM, i * 1.3, 1]], { stroke: VC.accent, w: 0.6, op: 0.45 });
      IF.path3(gl, view, [[i * 1.3, -LIM, 1], [i * 1.3, LIM, 1]], { stroke: VC.accent, w: 0.6, op: 0.45 });
    }
    const lab1 = view.project([-LIM, LIM, 1]);
    if (lab1.z > 0) gl.append("text").attr("x", lab1.x - 4).attr("y", lab1.y - 6).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", VC.accent).text("w̃ = 1");

    /* the coordinate frame of ℝ³ */
    VZ.axes3(gl, view, 1.9, { labels: ["x̃", "ỹ", "w̃"], colors: [VC.bad, VC.good, VC.violet], w: 1.3 });
    IF.dot3(gl, view, [0, 0, 0], { r: 3.5, fill: VC.ink });
    const org = view.project([0, 0, 0]);
    if (org.z > 0) gl.append("text").attr("x", org.x - 12).attr("y", org.y + 14)
      .attr("font-size", 10).attr("fill", VC.muted).text("O");

    /* the ray itself — the FULL line, both directions, because ±λ name one point */
    const L = 2.9;
    IF.path3(gl, view, [VZ.scale(d, -L), VZ.scale(d, L)], { stroke: VC.a2, w: 2.2, op: 0.9 });
    IF.dot3(gl, view, VZ.scale(d, lam * 1.6), { r: 5, fill: VC.a2, stroke: VC.bg });
    const mk = view.project(VZ.scale(d, lam * 1.6));
    if (mk.z > 0) gl.append("text").attr("x", mk.x + 8).attr("y", mk.y - 6)
      .attr("font-size", 10.5).attr("fill", VC.a2).text("λ x̃");

    if (finite && rho <= LIM * 1.02) {
      IF.dot3(gl, view, pierce, { r: 5.5, fill: VC.good, stroke: VC.bg });
      const pp = view.project(pierce);
      gl.append("text").attr("x", pp.x + 8).attr("y", pp.y + 4)
        .attr("font-size", 10.5).attr("fill", VC.good).text("the point");
    } else {
      gl.append("text").attr("x", 12).attr("y", f.ih - 12).attr("font-size", 11)
        .attr("fill", finite ? VC.a2 : VC.bad)
        .text(finite ? "the pierce point is off the drawn patch — |x| = " + VZ.fmt(rho, 2)
          : "the ray lies in w̃ = 0 — it never pierces the plane");
    }
    gl.append("text").attr("x", 10).attr("y", 16).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("ℝ³, with ℙ² as its lines through O");

    /* ─── right: the plane itself ─── */
    const gr = g.append("g").attr("transform", `translate(${leftW + 16},0)`);
    gr.append("rect").attr("x", 0).attr("y", 0).attr("width", rightW).attr("height", f.ih)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.6).attr("stroke", VC.line);
    const sz = Math.min(rightW, f.ih) - 46;
    const ox = (rightW - sz) / 2, oy = (f.ih - sz) / 2 + 8;
    const sx = d3.scaleLinear().domain([-M, M]).range([ox, ox + sz]);
    const sy = d3.scaleLinear().domain([-M, M]).range([oy + sz, oy]);
    gr.append("rect").attr("x", ox).attr("y", oy).attr("width", sz).attr("height", sz)
      .attr("fill", VC.accent).attr("fill-opacity", 0.05).attr("stroke", VC.accent).attr("stroke-opacity", 0.5);
    for (let i = -3; i <= 3; i++) {
      gr.append("line").attr("x1", sx(-M)).attr("x2", sx(M)).attr("y1", sy(i)).attr("y2", sy(i))
        .attr("stroke", VC.grid).attr("stroke-opacity", i === 0 ? 1 : 0.55).attr("stroke-width", i === 0 ? 1.2 : 0.7);
      gr.append("line").attr("y1", sy(-M)).attr("y2", sy(M)).attr("x1", sx(i)).attr("x2", sx(i))
        .attr("stroke", VC.grid).attr("stroke-opacity", i === 0 ? 1 : 0.55).attr("stroke-width", i === 0 ? 1.2 : 0.7);
    }
    if (ePar.checked) {
      const m = 0.55;
      [[0.0, VC.good], [1.1, VC.teal]].forEach(([c, col]) => {
        gr.append("line").attr("x1", sx(-M)).attr("y1", sy(-M * m + c))
          .attr("x2", sx(M)).attr("y2", sy(M * m + c)).attr("stroke", col).attr("stroke-width", 1.8);
      });
    }
    gr.append("text").attr("x", 10).attr("y", 16).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("the plane, dehomogenised");

    if (finite && rho < M * 0.98) {
      gr.append("circle").attr("cx", sx(pierce[0])).attr("cy", sy(pierce[1])).attr("r", 5.5)
        .attr("fill", VC.good).attr("stroke", VC.bg);
      gr.append("text").attr("x", sx(pierce[0]) + 9).attr("y", sy(pierce[1]) + 4)
        .attr("font-size", 10.5).attr("fill", VC.good)
        .text("(" + VZ.fmt(pierce[0], 2) + ", " + VZ.fmt(pierce[1], 2) + ")");
    } else {
      /* an ideal point has no location — only a direction. Draw it as one. */
      const u = [Math.cos(phi), Math.sin(phi)];
      const cx0 = ox + sz / 2, cy0 = oy + sz / 2, rr = sz / 2 - 4;
      VZ.arrow(gr, cx0, cy0, cx0 + u[0] * rr, cy0 - u[1] * rr, { color: VC.bad, w: 2, head: 8 });
      gr.append("text").attr("x", cx0 + u[0] * rr * 0.62 + 8).attr("y", cy0 - u[1] * rr * 0.62 - 6)
        .attr("font-size", 10.5).attr("fill", VC.bad).text("ideal point — a direction only");
    }
    gr.append("text").attr("x", ox + sz).attr("y", oy + sz + 16).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", VC.muted).text("viewport ±" + M);

    /* ─── readout ─── */
    const xt = VZ.scale(d, lam * 1.6);
    out.innerHTML =
      `ray direction x̃ = (<b>${VZ.fmt(d[0], 4)}, ${VZ.fmt(d[1], 4)}, ${VZ.fmt(d[2], 4)}</b>) · `
      + `the marker sits at <span class="keep">λ</span>x̃ = (${VZ.fmt(xt[0], 3)}, ${VZ.fmt(xt[1], 3)}, ${VZ.fmt(xt[2], 3)}) `
      + `with <span class="keep">λ</span> = ${VZ.fmt(lam * 1.6, 3)}<br>`
      + (finite
        ? `w̃ = sin <span class="keep">ψ</span> = <b>${VZ.fmt(d[2], 5)} ≠ 0</b>, so dehomogenising gives the finite point `
        + `(x̃/w̃, ỹ/w̃) = <b>(${VZ.fmt(d[0] / d[2], 4)}, ${VZ.fmt(d[1] / d[2], 4)})</b>, at distance `
        + `cot <span class="keep">ψ</span> = <b>${VZ.fmt(rho, 4)}</b> from the origin. Moving <span class="keep">λ</span> does not change it.`
        : `w̃ = <b>0</b> exactly. There is no finite point: this class is the ideal point in direction `
        + `(${VZ.fmt(Math.cos(phi), 4)}, ${VZ.fmt(Math.sin(phi), 4)}), one of the points making up the line at infinity.`)
      + (ePar.checked
        ? `<br>The two parallel lines are two planes through O; their intersection is the direction `
        + `(1, 0.55, 0)/‖·‖, which lies in w̃ = 0 — hence they meet at infinity, and only there.`
        : "");
  }

  eEl.oninput = () => { eElv.textContent = (+eEl.value).toFixed(2) + "°"; draw(); };
  eAz.oninput = () => { eAzv.textContent = eAz.value + "°"; draw(); };
  eLam.oninput = () => { eLamv.textContent = (+eLam.value * 1.6).toFixed(2); draw(); };
  eYaw.oninput = draw; ePar.onchange = draw; eW0.onchange = draw;
  eElv.textContent = (+eEl.value).toFixed(2) + "°";
  eLamv.textContent = (+eLam.value * 1.6).toFixed(2);
  draw();
})();

/* ───────── 2 · one rectangle through the five levels of the hierarchy ───────── */
(function () {
  const svg = d3.select("#hier-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 470;
  const eLv = document.getElementById("hi-lv");
  const eTh = document.getElementById("hi-th"), eThv = document.getElementById("hi-thv");
  const eS = document.getElementById("hi-s"), eSv = document.getElementById("hi-sv");
  const eSh = document.getElementById("hi-sh"), eShv = document.getElementById("hi-shv");
  const eP1 = document.getElementById("hi-p1"), eP1v = document.getElementById("hi-p1v");
  const eP2 = document.getElementById("hi-p2"), eP2v = document.getElementById("hi-p2v");
  const eG = document.getElementById("hi-grid"), eCr = document.getElementById("hi-cr");
  const out = document.getElementById("hier-readout");

  const RECT = [[0, 0], [2, 0], [2, 1], [0, 1]];
  const COLL = [[0, 0], [1, 0], [2.5, 0], [4, 0]];          // four collinear points, on y = 0
  const TX = 3, TY = -1;

  /* the five levels, all as 3×3 homogeneous matrices so one code path draws them all */
  function build(level, th, s, sh, p1, p2) {
    if (level === 0) return VZ.T2(TX, TY);
    if (level === 1) return VZ.R2(th, TX, TY);
    if (level === 2) return VZ.S2(s, th, TX, TY);
    const A = [[1.2, sh], [-0.3, 0.9]];                     // the running affine block
    if (level === 3) return [[A[0][0], A[0][1], TX], [A[1][0], A[1][1], TY], [0, 0, 1]];
    return [[A[0][0], A[0][1], TX], [A[1][0], A[1][1], TY], [p1, p2, 1]];
  }

  const ang = (a, b, c) => {                                 // angle at b, in degrees
    const u = VZ.sub(a, b), v = VZ.sub(c, b);
    return VZ.deg(Math.acos(VZ.clamp(VZ.dot(u, v) / (VZ.norm(u) * VZ.norm(v)), -1, 1)));
  };
  function measure(Q) {
    const L = [0, 1, 2, 3].map(i => VZ.norm(VZ.sub(Q[(i + 1) % 4], Q[i])));
    const A = [0, 1, 2, 3].map(i => ang(Q[(i + 3) % 4], Q[i], Q[(i + 1) % 4]));
    const d0 = VZ.sub(Q[1], Q[0]), d2 = VZ.sub(Q[2], Q[3]);   // the two "long" sides
    const par = VZ.deg(Math.abs(Math.atan2(d0[0] * d2[1] - d0[1] * d2[0], VZ.dot(d0, d2))));
    let area = 0;
    for (let i = 0; i < 4; i++) area += Q[i][0] * Q[(i + 1) % 4][1] - Q[(i + 1) % 4][0] * Q[i][1];
    return { L: L, A: A, ratio: L[0] / L[1], par: par, area: Math.abs(area) / 2 };
  }
  const crossRatio = P => {
    const d = (a, b) => VZ.norm(VZ.sub(P[b], P[a]));
    return (d(0, 2) * d(1, 3)) / (d(1, 2) * d(0, 3));
  };

  function draw() {
    const level = +eLv.value;
    const th = VZ.rad(+eTh.value), s = +eS.value, sh = +eSh.value;
    const p1 = +eP1.value, p2 = +eP2.value;
    const Hm = build(level, th, s, sh, p1, p2);

    const Q = RECT.map(p => VZ.applyH(Hm, p) || [0, 0]);
    const Qc = COLL.map(p => VZ.applyH(Hm, p) || [0, 0]);
    const m0 = measure(RECT), m1 = measure(Q);
    const cr0 = crossRatio(COLL), cr1 = crossRatio(Qc);

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const panelW = (f.iw - 16) / 2, panelH = 268;

    /* one shared world→screen scale so both panels are comparable */
    const all = RECT.concat(COLL).concat(Q).concat(Qc);
    const xr = d3.extent(all, p => p[0]), yr = d3.extent(all, p => p[1]);
    const cx = (xr[0] + xr[1]) / 2, cy = (yr[0] + yr[1]) / 2;
    const span = Math.max(xr[1] - xr[0], yr[1] - yr[0], 3) * 1.16;
    const k = Math.min(panelW, panelH) / span;

    function panel(x0, title, shape, coll, colour, useH) {
      const gg = g.append("g").attr("transform", `translate(${x0},0)`);
      gg.append("rect").attr("x", 0).attr("y", 0).attr("width", panelW).attr("height", panelH)
        .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.5).attr("stroke", VC.line);
      const sx = v => panelW / 2 + (v - cx) * k, sy = v => panelH / 2 - (v - cy) * k;
      const cid = "hi-clip-" + (useH ? "b" : "a");
      const cp = VZ.clip(svg, cid, x0, 0, panelW, panelH);
      const gi = gg.append("g").attr("clip-path", cp)
        .attr("transform", `translate(${-x0},0)`).append("g").attr("transform", `translate(${x0},0)`);

      if (eG.checked) {
        /* a grid in SOURCE coordinates, then mapped — so the picture shows what the
           transformation does to the plane, not just to the four corners */
        const N = 9, step = 1;
        for (let i = -N; i <= N; i++) {
          for (const horiz of [true, false]) {
            const pts = [];
            for (let t = -N; t <= N; t += 0.25) {
              const p = horiz ? [t * step, i * step] : [i * step, t * step];
              const q = useH ? VZ.applyH(Hm, p) : p;
              if (q) pts.push([sx(q[0]), sy(q[1])]);
            }
            if (pts.length > 1) VZ.poly(gi, pts, { stroke: VC.grid, w: 0.7, close: false, op: 1 });
          }
        }
      }
      VZ.poly(gi, shape.map(p => [sx(p[0]), sy(p[1])]),
        { stroke: colour, w: 2.2, fill: colour, fillOp: 0.15 });
      shape.forEach((p, i) => gi.append("circle").attr("cx", sx(p[0])).attr("cy", sy(p[1])).attr("r", 3)
        .attr("fill", colour));
      if (eCr.checked) {
        VZ.poly(gi, coll.map(p => [sx(p[0]), sy(p[1])]), { stroke: VC.a2, w: 1.2, close: false, dash: "3 3" });
        coll.forEach((p, i) => {
          gi.append("circle").attr("cx", sx(p[0])).attr("cy", sy(p[1])).attr("r", 3.4)
            .attr("fill", VC.a2).attr("stroke", VC.bg).attr("stroke-width", 0.8);
          gi.append("text").attr("x", sx(p[0])).attr("y", sy(p[1]) - 7).attr("text-anchor", "middle")
            .attr("font-size", 9.5).attr("fill", VC.a2).text("ABCD"[i]);
        });
      }
      gg.append("text").attr("x", 10).attr("y", 17).attr("font-size", 11.5).attr("fill", VC.ink)
        .attr("font-weight", 600).text(title);
      return gg;
    }

    const names = ["translation · 2 DoF", "rigid · 3 DoF", "similarity · 4 DoF", "affine · 6 DoF", "projective · 8 DoF"];
    panel(0, "original", RECT, COLL, VC.accent, false);
    panel(panelW + 16, "after " + names[level], Q, Qc, VC.good, true);

    /* ── the measurement table, coloured by whether each quantity moved ── */
    const gt = g.append("g").attr("transform", `translate(0,${panelH + 20})`);
    const eq = (a, b, tol) => Math.abs(a - b) <= (tol === undefined ? 1e-6 : tol);
    const rows = [
      ["side lengths", m0.L.map(v => VZ.fmt(v, 3)).join(", "), m1.L.map(v => VZ.fmt(v, 3)).join(", "),
        m0.L.every((v, i) => eq(v, m1.L[i], 1e-4))],
      ["long / short", VZ.fmt(m0.ratio, 4), VZ.fmt(m1.ratio, 4), eq(m0.ratio, m1.ratio, 1e-4)],
      ["corner angles", m0.A.map(v => VZ.fmt(v, 1)).join(", "), m1.A.map(v => VZ.fmt(v, 1)).join(", "),
        m0.A.every((v, i) => eq(v, m1.A[i], 5e-3))],
      ["opposite sides parallel?", "yes (0.00°)",
        m1.par < 1e-4 ? "yes (0.00°)" : "no (" + VZ.fmt(m1.par, 2) + "° apart)", m1.par < 1e-4],
      ["enclosed area", VZ.fmt(m0.area, 4), VZ.fmt(m1.area, 4), eq(m0.area, m1.area, 1e-4)],
      ["cross ratio of A B C D", VZ.fmt(cr0, 6), VZ.fmt(cr1, 6), eq(cr0, cr1, 1e-6)]
    ];
    const colX = [4, 190, 350, 560];
    gt.append("text").attr("x", colX[0]).attr("y", 0).attr("font-size", 10.5).attr("fill", VC.muted).text("quantity");
    gt.append("text").attr("x", colX[1]).attr("y", 0).attr("font-size", 10.5).attr("fill", VC.muted).text("original");
    gt.append("text").attr("x", colX[2]).attr("y", 0).attr("font-size", 10.5).attr("fill", VC.muted).text("transformed");
    gt.append("text").attr("x", colX[3]).attr("y", 0).attr("font-size", 10.5).attr("fill", VC.muted).text("verdict");
    rows.forEach((r, i) => {
      const y = 18 + i * 17, col = r[3] ? VC.good : VC.bad;
      gt.append("text").attr("x", colX[0]).attr("y", y).attr("font-size", 11).attr("fill", VC.ink).text(r[0]);
      gt.append("text").attr("x", colX[1]).attr("y", y).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", VC.muted).text(r[1]);
      gt.append("text").attr("x", colX[2]).attr("y", y).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col).text(r[2]);
      gt.append("text").attr("x", colX[3]).attr("y", y).attr("font-size", 10.5).attr("fill", col)
        .text(r[3] ? "preserved" : "destroyed");
    });

    /* ── readout ── */
    const active = ["t only", "t, θ", "t, θ, s", "t, A (2×2)", "t, A, and the bottom row"][level];
    const detA = level >= 3 ? (1.2 * 0.9 - sh * (-0.3)) : (level === 2 ? s * s : 1);
    out.innerHTML =
      `<b>${names[level]}</b> — the sliders that do anything at this level: ${active}.`
      + ` H = [[${VZ.fmt(Hm[0][0], 3)}, ${VZ.fmt(Hm[0][1], 3)}, ${VZ.fmt(Hm[0][2], 3)}], `
      + `[${VZ.fmt(Hm[1][0], 3)}, ${VZ.fmt(Hm[1][1], 3)}, ${VZ.fmt(Hm[1][2], 3)}], `
      + `[${VZ.fmt(Hm[2][0], 3)}, ${VZ.fmt(Hm[2][1], 3)}, ${VZ.fmt(Hm[2][2], 3)}]]<br>`
      + (level <= 2
        ? `Area went ${VZ.fmt(m0.area, 4)} → ${VZ.fmt(m1.area, 4)}, a factor of <b>${VZ.fmt(m1.area / m0.area, 4)}</b>`
        + (level === 2 ? ` = s² = ${VZ.fmt(s * s, 4)} ✓` : ` — a rigid motion cannot change area.`)
        : level === 3
          ? `Area went ${VZ.fmt(m0.area, 4)} → ${VZ.fmt(m1.area, 4)}, a factor of <b>${VZ.fmt(m1.area / m0.area, 4)}</b> = |det A| = ${VZ.fmt(Math.abs(detA), 4)} ✓ — the determinant IS the area scale.`
          : `Opposite sides are now <b>${VZ.fmt(m1.par, 3)}°</b> apart: parallelism is gone, and with it any single "area scale factor" — the local scale differs at every point.`)
      + `<br>Cross ratio: <b>${VZ.fmt(cr0, 8)}</b> → <b>${VZ.fmt(cr1, 8)}</b>`
      + (Math.abs(cr0 - cr1) < 1e-6 ? " — unchanged, as it must be at every level." : " — ONLY a non-projective map could do this.");
  }

  eLv.onchange = draw;
  eTh.oninput = () => { eThv.textContent = eTh.value + "°"; draw(); };
  eS.oninput = () => { eSv.textContent = (+eS.value).toFixed(2); draw(); };
  eSh.oninput = () => { eShv.textContent = (+eSh.value).toFixed(2); draw(); };
  eP1.oninput = () => { eP1v.textContent = (+eP1.value).toFixed(3); draw(); };
  eP2.oninput = () => { eP2v.textContent = (+eP2.value).toFixed(3); draw(); };
  eG.onchange = draw; eCr.onchange = draw;
  draw();
})();

/* ───────── 3 · gimbal lock, demonstrated rather than described ───────── */
(function () {
  const svg = d3.select("#gimbal-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 430;
  const eA = document.getElementById("gm-a"), eAv = document.getElementById("gm-av");
  const eB = document.getElementById("gm-b"), eBv = document.getElementById("gm-bv");
  const eG = document.getElementById("gm-g"), eGv = document.getElementById("gm-gv");
  const eYaw = document.getElementById("gm-yaw");
  const out = document.getElementById("gimbal-readout");

  /* the last object frame, so a "nudge" can be measured against it rather than asserted */
  let lastR = null, lastMsg = "";

  /* a circle of radius r in the plane normal to `ax`, carried by the matrix M */
  function ring(M, ax, r, n) {
    const u = VZ.unit(Math.abs(ax[0]) < 0.9 ? VZ.cross(ax, [1, 0, 0]) : VZ.cross(ax, [0, 1, 0]));
    const v = VZ.cross(ax, u);
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const t = 2 * Math.PI * i / n;
      const p = VZ.add(VZ.scale(u, r * Math.cos(t)), VZ.scale(v, r * Math.sin(t)));
      pts.push(VZ.mv(M, p));
    }
    return pts;
  }
  function detJ(a, b) {
    const wa = [0, 0, 1];
    const wb = VZ.mv(VZ.Rz(a), [0, 1, 0]);
    const wg = VZ.mv(VZ.mul(VZ.Rz(a), VZ.Ry(b)), [1, 0, 0]);
    return { d: VZ.det3(VZ.T([wa, wb, wg])), wa: wa, wb: wb, wg: wg };
  }

  function draw() {
    const a = VZ.rad(+eA.value), b = VZ.rad(+eB.value), g = VZ.rad(+eG.value);
    const R = VZ.eulerZYX(a, b, g);
    const J = detJ(a, b);

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const gg = f.g;
    const leftW = 452, rightW = f.iw - leftW - 16;

    /* ─── left: the three rings ─── */
    const gl = gg.append("g");
    gl.append("rect").attr("x", 0).attr("y", 0).attr("width", leftW).attr("height", f.ih)
      .attr("rx", 8).attr("fill", VC.bg).attr("fill-opacity", 0.35).attr("stroke", VC.line);
    const view = VZ.view3({
      target: [0, 0, 0], yaw: VZ.rad(+eYaw.value), pitch: VZ.rad(20), dist: 9.5,
      f: 330, cx: leftW / 2, cy: f.ih / 2
    });

    /* outer ring turns about world Z by α; middle about the yawed Y by β;
       inner about the yawed-and-pitched X by γ — the physical gimbal, in order */
    const M0 = VZ.eye(3);
    const M1 = VZ.Rz(a);
    const M2 = VZ.mul(M1, VZ.Ry(b));
    const M3 = VZ.mul(M2, VZ.Rx(g));

    const rings = [
      { M: M0, ax: [0, 0, 1], r: 2.5, col: VC.accent, name: "outer · yaw about Z" },
      { M: M1, ax: [0, 1, 0], r: 2.1, col: VC.a2, name: "middle · pitch about Y" },
      { M: M2, ax: [1, 0, 0], r: 1.7, col: VC.violet, name: "inner · roll about X" }
    ];
    /* the world frame, faint, for reference */
    VZ.axes3(gl, view, 3.1, { labels: ["X", "Y", "Z"], colors: [VC.line, VC.line, VC.line], w: 1 });

    rings.forEach(rg => {
      IF.path3(gl, view, ring(rg.M, rg.ax, rg.r, 96), { stroke: rg.col, w: 2, close: true });
      /* the ring's own axle, which is the direction that ring can rotate about */
      const ax = VZ.mv(rg.M, rg.ax);
      IF.path3(gl, view, [VZ.scale(ax, -rg.r * 1.12), VZ.scale(ax, rg.r * 1.12)],
        { stroke: rg.col, w: 1.2, dash: "4 3", op: 0.85 });
    });

    /* the object the gimbal is carrying: its own little frame */
    VZ.axes3(gl, view, 1.15, { labels: ["x", "y", "z"], colors: [VC.bad, VC.good, VC.teal], w: 2.4 });
    const gobj = gl.append("g");
    [[1.15, 0, 0], [0, 1.15, 0], [0, 0, 1.15]].forEach((d, i) => {
      const p1 = view.project(VZ.mv(R, d)), p0 = view.project([0, 0, 0]);
      gobj.append("line").attr("x1", p0.x).attr("y1", p0.y).attr("x2", p1.x).attr("y2", p1.y)
        .attr("stroke", [VC.bad, VC.good, VC.teal][i]).attr("stroke-width", 2.6);
      gobj.append("circle").attr("cx", p1.x).attr("cy", p1.y).attr("r", 3)
        .attr("fill", [VC.bad, VC.good, VC.teal][i]);
    });
    /* redraw the world frame faintly on top so the two are distinguishable */
    gl.append("text").attr("x", 10).attr("y", 17).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("three rings, and the frame they carry");
    VZ.legend(gl, rings.map(r => ({ label: r.name, color: r.col })), 10, f.ih - 76, { gap: 14 });

    /* the visual signature of the lock: outer and inner axles collinear */
    const axO = VZ.mv(M0, [0, 0, 1]), axI = VZ.mv(M2, [1, 0, 0]);
    const align = Math.abs(VZ.dot(VZ.unit(axO), VZ.unit(axI)));
    if (align > 0.999) {
      gl.append("text").attr("x", leftW / 2).attr("y", f.ih - 12).attr("text-anchor", "middle")
        .attr("font-size", 12).attr("fill", VC.bad).attr("font-weight", 600)
        .text("LOCKED — the outer and inner axles are the same line");
    }

    /* ─── right: the three achievable directions and det J ─── */
    const gr = gg.append("g").attr("transform", `translate(${leftW + 16},0)`);
    gr.append("rect").attr("x", 0).attr("y", 0).attr("width", rightW).attr("height", f.ih)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.6).attr("stroke", VC.line);
    gr.append("text").attr("x", 10).attr("y", 17).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("what the three controls can reach");

    const bx = 14, bw = rightW - 28;
    const items = [["ω from α̇ (yaw)", J.wa, VC.accent], ["ω from β̇ (pitch)", J.wb, VC.a2], ["ω from γ̇ (roll)", J.wg, VC.violet]];
    items.forEach((it, i) => {
      const y = 44 + i * 44;
      gr.append("text").attr("x", bx).attr("y", y).attr("font-size", 10.5).attr("fill", VC.muted).text(it[0]);
      gr.append("text").attr("x", bx).attr("y", y + 15).attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", it[2])
        .text("(" + it[1].map(v => VZ.fmt(v, 3)).join(", ") + ")");
    });
    /* the pairwise alignment that IS the lock */
    const cosAI = Math.abs(VZ.dot(VZ.unit(J.wa), VZ.unit(J.wg)));
    gr.append("text").attr("x", bx).attr("y", 190).attr("font-size", 10.5).attr("fill", VC.muted)
      .text("|cos∠(yaw axis, roll axis)|");
    gr.append("text").attr("x", bx).attr("y", 206).attr("font-size", 13)
      .attr("font-family", "SF Mono, Menlo, monospace")
      .attr("fill", cosAI > 0.99 ? VC.bad : VC.good).text(VZ.fmt(cosAI, 6));

    const dv = Math.abs(J.d);
    gr.append("text").attr("x", bx).attr("y", 238).attr("font-size", 10.5).attr("fill", VC.muted)
      .text("|det J| = |cos β|  — the reachable volume");
    gr.append("rect").attr("x", bx).attr("y", 246).attr("width", bw).attr("height", 16).attr("rx", 3)
      .attr("fill", VC.grid);
    gr.append("rect").attr("x", bx).attr("y", 246).attr("width", Math.max(1, bw * dv)).attr("height", 16).attr("rx", 3)
      .attr("fill", dv < 0.1 ? VC.bad : (dv < 0.4 ? VC.a2 : VC.good));
    gr.append("text").attr("x", bx).attr("y", 280).attr("font-size", 15)
      .attr("font-family", "SF Mono, Menlo, monospace")
      .attr("fill", dv < 0.1 ? VC.bad : VC.ink).text(VZ.fmt(dv, 6));
    gr.append("text").attr("x", bx).attr("y", 300).attr("font-size", 10).attr("fill", VC.muted)
      .text("0 means one whole degree of freedom");
    gr.append("text").attr("x", bx).attr("y", 313).attr("font-size", 10).attr("fill", VC.muted)
      .text("is unreachable — not merely stiff");
    IF.matText(gr, R, bx, 344, { label: "R(α, β, γ)", dp: 4, size: 10, pad: 8 });

    /* ─── readout ─── */
    const locked = dv < 1e-3;
    out.innerHTML =
      `<span class="keep">α</span> = ${eA.value}°, <span class="keep">β</span> = ${eB.value}°, <span class="keep">γ</span> = ${eG.value}° · `
      + `|det J| = |cos <span class="keep">β</span>| = <b>${VZ.fmt(dv, 6)}</b>`
      + (locked
        ? ` — <b style="color:var(--bad)">locked</b>. Only ${(+eB.value) > 0 ? "<span class='keep'>α</span> − <span class='keep'>γ</span>" : "<span class='keep'>α</span> + <span class='keep'>γ</span>"} = `
        + `${VZ.fmt((+eB.value) > 0 ? (+eA.value) - (+eG.value) : (+eA.value) + (+eG.value), 1)}° affects the orientation; the split between the two is free.`
        : dv < 0.15
          ? ` — badly conditioned. A rotation of 1° about the failing direction already needs about `
          + `${VZ.fmt(1 / dv, 1)}° of coordinated yaw and roll.`
          : ` — healthy. All three controls move the frame in independent directions.`)
      + (lastMsg ? "<br>" + lastMsg : "");
    lastR = R;
  }

  function nudge() {
    const before = VZ.eulerZYX(VZ.rad(+eA.value), VZ.rad(+eB.value), VZ.rad(+eG.value));
    eA.value = Math.max(-180, Math.min(180, +eA.value + 20));
    eG.value = Math.max(-180, Math.min(180, +eG.value + 20));
    eAv.textContent = eA.value + "°"; eGv.textContent = eG.value + "°";
    const after = VZ.eulerZYX(VZ.rad(+eA.value), VZ.rad(+eB.value), VZ.rad(+eG.value));
    /* measure the change the honest way: the angle of the relative rotation */
    const rel = VZ.mul(VZ.T(before), after);
    const ang = VZ.deg(VZ.axisAngleFromR(rel).angle);
    lastMsg = `Nudged <span class="keep">α</span> and <span class="keep">γ</span> each by +20°. The frame actually rotated by `
      + `<b>${VZ.fmt(ang, 6)}°</b>`
      + (ang < 1e-3 ? " — that is, <b>not at all</b>: 40° of control input, zero motion."
        : ", which is what a working parameterisation looks like.");
    draw();
  }

  eA.oninput = () => { eAv.textContent = eA.value + "°"; lastMsg = ""; draw(); };
  eB.oninput = () => { eBv.textContent = (+eB.value).toFixed(1) + "°"; lastMsg = ""; draw(); };
  eG.oninput = () => { eGv.textContent = eG.value + "°"; lastMsg = ""; draw(); };
  eYaw.oninput = draw;
  document.getElementById("gm-couple").onclick = nudge;
  document.getElementById("gm-free").onclick = () => {
    eB.value = 0; eBv.textContent = "0.0°"; lastMsg = ""; draw();
  };
  eBv.textContent = (+eB.value).toFixed(1) + "°";
  draw();
})();

/* ───────── 4 · Rodrigues, one term at a time ───────── */
(function () {
  const svg = d3.select("#rod-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 430;
  const eAz = document.getElementById("rd-az"), eAzv = document.getElementById("rd-azv");
  const eEl = document.getElementById("rd-el"), eElv = document.getElementById("rd-elv");
  const eTh = document.getElementById("rd-th"), eThv = document.getElementById("rd-thv");
  const eYaw = document.getElementById("rd-yaw");
  const eP = document.getElementById("rd-parts"), eC = document.getElementById("rd-cone");
  const out = document.getElementById("rod-readout");

  const V = [1.9, 0.35, 0.55];                    // the vector being rotated, fixed

  function draw() {
    const az = VZ.rad(+eAz.value), el = VZ.rad(+eEl.value), th = VZ.rad(+eTh.value);
    const n = VZ.unit([Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)]);

    /* the five pieces of the derivation, each computed the way the text derives it */
    const vpar = VZ.scale(n, VZ.dot(n, V));            // v∥ = n̂(n̂·v)
    const vperp = VZ.sub(V, vpar);                     // v⊥ = v − v∥
    const vcross = VZ.cross(n, V);                     // v× = n̂ × v
    const uperp = VZ.add(VZ.scale(vperp, Math.cos(th)), VZ.scale(vcross, Math.sin(th)));
    const U = VZ.add(uperp, vpar);
    const R = VZ.rodrigues(n, th);
    const Um = VZ.mv(R, V);                            // the same thing, via the matrix

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const leftW = 470, rightW = f.iw - leftW - 16;

    const gl = g.append("g");
    gl.append("rect").attr("x", 0).attr("y", 0).attr("width", leftW).attr("height", f.ih)
      .attr("rx", 8).attr("fill", VC.bg).attr("fill-opacity", 0.35).attr("stroke", VC.line);
    const view = VZ.view3({
      target: [0, 0, 0.15], yaw: VZ.rad(+eYaw.value), pitch: VZ.rad(20), dist: 8.2,
      f: 330, cx: leftW / 2, cy: f.ih / 2 + 6
    });
    VZ.axes3(gl, view, 2.4, { labels: ["x", "y", "z"], colors: [VC.line, VC.line, VC.line], w: 1 });

    /* the cone that v sweeps: the circle of radius ‖v⊥‖ centred on v∥, in the spin plane */
    if (eC.checked && VZ.norm(vperp) > 1e-6) {
      const pts = [];
      for (let i = 0; i <= 96; i++) {
        const t = 2 * Math.PI * i / 96;
        pts.push(VZ.add(vpar, VZ.add(VZ.scale(vperp, Math.cos(t)), VZ.scale(vcross, Math.sin(t)))));
      }
      IF.path3(gl, view, pts, { stroke: VC.muted, w: 1, dash: "3 3", close: true, op: 0.8 });
      IF.path3(gl, view, [[0, 0, 0], vpar], { stroke: VC.muted, w: 0.9, dash: "2 3", op: 0.7 });
    }

    /* the axis, drawn through the origin both ways */
    const A = 2.5;
    IF.path3(gl, view, [VZ.scale(n, -A * 0.4), VZ.scale(n, A)], { stroke: VC.violet, w: 2.4 });
    const np = view.project(VZ.scale(n, A));
    gl.append("text").attr("x", np.x + 6).attr("y", np.y - 4).attr("font-size", 11)
      .attr("fill", VC.violet).text("n̂");

    function vec(p, col, lab, w, dash) {
      const a = view.project([0, 0, 0]), b = view.project(p);
      if (b.z <= 0.05) return;
      VZ.arrow(gl, a.x, a.y, b.x, b.y, { color: col, w: w || 2, head: 7, dash: dash || null });
      if (lab) gl.append("text").attr("x", b.x + 6).attr("y", b.y - 5).attr("font-size", 11)
        .attr("fill", col).text(lab);
    }
    function seg(p0, p1, col, lab, dash) {
      const a = view.project(p0), b = view.project(p1);
      if (a.z <= 0.05 || b.z <= 0.05) return;
      gl.append("line").attr("x1", a.x).attr("y1", a.y).attr("x2", b.x).attr("y2", b.y)
        .attr("stroke", col).attr("stroke-width", 1.4).attr("stroke-dasharray", dash || "4 3")
        .attr("stroke-opacity", 0.85);
      if (lab) gl.append("text").attr("x", (a.x + b.x) / 2 + 5).attr("y", (a.y + b.y) / 2 - 4)
        .attr("font-size", 10).attr("fill", col).text(lab);
    }

    if (eP.checked) {
      vec(vpar, VC.teal, "v∥", 1.8);
      vec(vcross, VC.a2, "v× = n̂ × v", 1.6, "5 3");
      seg(vpar, V, VC.good, "v⊥");
      seg(vpar, U, VC.bad, null);
    }
    vec(V, VC.accent, "v", 2.4);
    vec(U, VC.bad, "u = Rv", 2.6);

    gl.append("text").attr("x", 10).attr("y", 17).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("u = v∥ + cos θ·v⊥ + sin θ·v×");
    VZ.legend(gl, [
      { label: "v — the input", color: VC.accent },
      { label: "v∥ — fixed by the rotation", color: VC.teal },
      { label: "v⊥ — the part that spins", color: VC.good },
      { label: "v× = n̂ × v — v⊥ turned 90°", color: VC.a2 },
      { label: "u = Rv", color: VC.bad }
    ], 10, f.ih - 78, { gap: 14 });

    /* ─── right: the matrix and its certificates ─── */
    const gr = g.append("g").attr("transform", `translate(${leftW + 16},0)`);
    gr.append("rect").attr("x", 0).attr("y", 0).attr("width", rightW).attr("height", f.ih)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.6).attr("stroke", VC.line);
    gr.append("text").attr("x", 12).attr("y", 17).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("R = I + sin θ[n̂]ₓ + (1−cos θ)[n̂]ₓ²");
    IF.matText(gr, R, 12, 48, { dp: 5, size: 10.5, pad: 9 });

    /* certificates, all recomputed */
    const G = VZ.mul(VZ.T(R), R);
    let orth = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) orth = Math.max(orth, Math.abs(G[i][j] - (i === j ? 1 : 0)));
    const det = VZ.det3(R), tr = R[0][0] + R[1][1] + R[2][2];
    const thTr = VZ.deg(Math.acos(VZ.clamp((tr - 1) / 2, -1, 1)));
    const fixed = VZ.norm(VZ.sub(VZ.mv(R, n), n));
    const agree = VZ.norm(VZ.sub(U, Um));
    const rows = [
      ["‖RᵀR − I‖max", orth.toExponential(1), orth < 1e-12],
      ["det R", VZ.fmt(det, 8), Math.abs(det - 1) < 1e-9],
      ["tr R = 1 + 2cos θ", VZ.fmt(tr, 6), Math.abs(tr - (1 + 2 * Math.cos(th))) < 1e-9],
      ["θ from the trace", VZ.fmt(thTr, 4) + "°", Math.abs(thTr - Math.abs(+eTh.value)) < 1e-3],
      ["‖Rn̂ − n̂‖ (axis fixed)", fixed.toExponential(1), fixed < 1e-12],
      ["‖(built by hand) − Rv‖", agree.toExponential(1), agree < 1e-12]
    ];
    rows.forEach((r, i) => {
      const y = 132 + i * 21;
      gr.append("text").attr("x", 12).attr("y", y).attr("font-size", 10.5).attr("fill", VC.muted).text(r[0]);
      gr.append("text").attr("x", rightW - 12).attr("y", y).attr("text-anchor", "end").attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", r[2] ? VC.good : VC.a2).text(r[1]);
    });
    gr.append("text").attr("x", 12).attr("y", 278).attr("font-size", 10.5).attr("fill", VC.muted)
      .text("as a quaternion (x, y, z, w):");
    const q = VZ.qFromAxisAngle(n, th);
    gr.append("text").attr("x", 12).attr("y", 296).attr("font-size", 10.5)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", VC.violet)
      .text("(" + q.map(v => VZ.fmt(v, 4)).join(", ") + ")");
    const Rq = VZ.qToR(q);
    let dq = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) dq = Math.max(dq, Math.abs(R[i][j] - Rq[i][j]));
    gr.append("text").attr("x", 12).attr("y", 316).attr("font-size", 10).attr("fill", dq < 1e-12 ? VC.good : VC.bad)
      .text("‖R(quaternion) − R(Rodrigues)‖ = " + dq.toExponential(1));

    /* ─── readout ─── */
    const rec = VZ.axisAngleFromR(R);
    const flip = VZ.dot(rec.axis, n) < 0;
    out.innerHTML =
      `n̂ = (${n.map(v => VZ.fmt(v, 4)).join(", ")}), <span class="keep">θ</span> = ${eTh.value}° · `
      + `‖v∥‖ = ${VZ.fmt(VZ.norm(vpar), 4)}, ‖v⊥‖ = ${VZ.fmt(VZ.norm(vperp), 4)}, `
      + `‖v×‖ = ${VZ.fmt(VZ.norm(vcross), 4)} — and ‖v⊥‖ = ‖v×‖ to ${Math.abs(VZ.norm(vperp) - VZ.norm(vcross)).toExponential(0)}, `
      + `which is why {v⊥, v×} is an orthogonal frame for the spin plane.<br>`
      + `‖v‖ = ${VZ.fmt(VZ.norm(V), 6)} and ‖u‖ = ${VZ.fmt(VZ.norm(U), 6)} — a rotation preserves length. `
      + `Recovering the axis from R gives (${rec.axis.map(v => VZ.fmt(v, 4)).join(", ")}) at `
      + `${VZ.fmt(VZ.deg(rec.angle), 3)}°`
      + (flip ? ` — the sign-flipped axis with the sign-flipped angle, which names the same rotation.` : `.`)
      + (Math.abs(+eTh.value) < 3 ? ` <b style="color:var(--bad)">Near θ = 0 the recovered axis is meaningless</b> — every direction fixes the identity.` : "")
      + (Math.abs(Math.abs(+eTh.value) - 180) < 3 ? ` <b style="color:var(--bad)">Near θ = 180° the antisymmetric part vanishes</b> and the axis must be dug out of the symmetric part, sign and all.` : "");
  }

  eAz.oninput = () => { eAzv.textContent = eAz.value + "°"; draw(); };
  eEl.oninput = () => { eElv.textContent = eEl.value + "°"; draw(); };
  eTh.oninput = () => { eThv.textContent = eTh.value + "°"; draw(); };
  eYaw.oninput = draw; eP.onchange = draw; eC.onchange = draw;
  draw();
})();

/* ───────── 5 · slerp against the alternatives ───────── */
(function () {
  const svg = d3.select("#slerp-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 450, N = 40;
  const eA = document.getElementById("sl-a"), eAv = document.getElementById("sl-av");
  const eAng = document.getElementById("sl-ang"), eAngv = document.getElementById("sl-angv");
  const eYaw = document.getElementById("sl-yaw");
  const eS = document.getElementById("sl-slerp"), eN = document.getElementById("sl-nlerp");
  const eE = document.getElementById("sl-euler"), eL = document.getElementById("sl-long");
  const out = document.getElementById("slerp-readout");

  const AXIS = VZ.unit([0.3, 0.8, 0.52]);
  const R0 = VZ.eulerZYX(VZ.rad(-40), VZ.rad(20), VZ.rad(15));
  const PROBE = [0, 0, 1.55];                        // the body axis whose track is drawn

  function draw() {
    const a = +eA.value, endAng = VZ.rad(+eAng.value);
    const R1 = VZ.rodrigues(AXIS, endAng);
    const q0 = VZ.qFromR(R0);
    let q1 = VZ.qFromR(R1);
    const longWay = eL.checked;
    if (longWay) q1 = q1.map(v => -v);

    const e0 = VZ.eulerFromZYX(R0), e1 = VZ.eulerFromZYX(R1);

    const fSlerp = t => VZ.qToR(VZ.slerp(q0, q1, t, longWay));
    const fNlerp = t => {
      let b = q1;
      if (!longWay && VZ.dot(q0, q1) < 0) b = q1.map(v => -v);
      return VZ.qToR(VZ.qNorm(q0.map((v, i) => (1 - t) * v + t * b[i])));
    };
    const fEuler = t => VZ.eulerZYX(e0.a + t * (e1.a - e0.a), e0.b + t * (e1.b - e0.b), e0.g + t * (e1.g - e0.g));

    /* measure each path: the geodesic angle between consecutive samples */
    function steps(fn) {
      const s = [];
      let prev = fn(0);
      for (let i = 1; i <= N; i++) {
        const c = fn(i / N);
        s.push(VZ.deg(VZ.axisAngleFromR(VZ.mul(VZ.T(prev), c)).angle));
        prev = c;
      }
      return s;
    }
    const methods = [
      { on: eS.checked, name: "slerp", fn: fSlerp, col: VC.good },
      { on: eN.checked, name: "nlerp", fn: fNlerp, col: VC.a2 },
      { on: eE.checked, name: "Euler lerp", fn: fEuler, col: VC.bad }
    ].map(m => {
      if (!m.on) return m;
      const st = steps(m.fn);
      return Object.assign(m, { steps: st, total: st.reduce((x, y) => x + y, 0), max: Math.max(...st), min: Math.min(...st) });
    });
    const geo = VZ.deg(VZ.axisAngleFromR(VZ.mul(VZ.T(R0), R1)).angle);

    const f = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = f.g;
    const leftW = 430, rightW = f.iw - leftW - 16;

    /* ─── left: the tracks on a sphere ─── */
    const gl = g.append("g");
    gl.append("rect").attr("x", 0).attr("y", 0).attr("width", leftW).attr("height", f.ih)
      .attr("rx", 8).attr("fill", VC.bg).attr("fill-opacity", 0.35).attr("stroke", VC.line);
    const view = VZ.view3({
      target: [0, 0, 0], yaw: VZ.rad(+eYaw.value), pitch: VZ.rad(18), dist: 7.4,
      f: 300, cx: leftW / 2, cy: f.ih / 2
    });
    /* a wire sphere of radius ‖PROBE‖ for reference */
    const rad = VZ.norm(PROBE);
    for (let k = -2; k <= 2; k++) {
      const z = rad * Math.sin(VZ.rad(k * 30)), rr = Math.sqrt(Math.max(0, rad * rad - z * z));
      const pts = [];
      for (let i = 0; i <= 72; i++) { const t = 2 * Math.PI * i / 72; pts.push([rr * Math.cos(t), rr * Math.sin(t), z]); }
      IF.path3(gl, view, pts, { stroke: VC.grid, w: 0.7, close: true });
    }
    for (let k = 0; k < 6; k++) {
      const ph = Math.PI * k / 6, pts = [];
      for (let i = 0; i <= 72; i++) {
        const t = 2 * Math.PI * i / 72;
        pts.push([rad * Math.cos(t) * Math.cos(ph), rad * Math.cos(t) * Math.sin(ph), rad * Math.sin(t)]);
      }
      IF.path3(gl, view, pts, { stroke: VC.grid, w: 0.7, close: true });
    }
    VZ.axes3(gl, view, 2.05, { labels: ["x", "y", "z"], colors: [VC.line, VC.line, VC.line], w: 1 });

    methods.forEach(m => {
      if (!m.on) return;
      const pts = [];
      for (let i = 0; i <= 120; i++) pts.push(VZ.mv(m.fn(i / 120), PROBE));
      IF.path3(gl, view, pts, { stroke: m.col, w: 2, op: 0.95 });
    });
    /* endpoints */
    IF.dot3(gl, view, VZ.mv(R0, PROBE), { r: 5, fill: VC.accent, stroke: VC.bg });
    IF.dot3(gl, view, VZ.mv(R1, PROBE), { r: 5, fill: VC.violet, stroke: VC.bg });
    const p0 = view.project(VZ.mv(R0, PROBE)), p1 = view.project(VZ.mv(R1, PROBE));
    gl.append("text").attr("x", p0.x + 7).attr("y", p0.y - 5).attr("font-size", 10.5).attr("fill", VC.accent).text("start");
    gl.append("text").attr("x", p1.x + 7).attr("y", p1.y - 5).attr("font-size", 10.5).attr("fill", VC.violet).text("end");

    /* the current interpolated frame, from whichever method is topmost */
    const live = methods.find(m => m.on) || methods[0];
    const Rc = live.fn(a);
    [[1.15, 0, 0], [0, 1.15, 0], [0, 0, 1.15]].forEach((d, i) => {
      const q = view.project(VZ.mv(Rc, d)), o = view.project([0, 0, 0]);
      gl.append("line").attr("x1", o.x).attr("y1", o.y).attr("x2", q.x).attr("y2", q.y)
        .attr("stroke", [VC.bad, VC.good, VC.teal][i]).attr("stroke-width", 2.6);
    });
    IF.dot3(gl, view, VZ.mv(Rc, PROBE), { r: 5.5, fill: live.col, stroke: VC.bg });
    gl.append("text").attr("x", 10).attr("y", 17).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("the track of one body axis");
    VZ.legend(gl, methods.filter(m => m.on).map(m => ({ label: m.name, color: m.col })), 10, f.ih - 46, { gap: 14 });

    /* ─── right: measured step sizes ─── */
    const gr = g.append("g").attr("transform", `translate(${leftW + 16},0)`);
    gr.append("rect").attr("x", 0).attr("y", 0).attr("width", rightW).attr("height", f.ih)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.6).attr("stroke", VC.line);
    gr.append("text").attr("x", 12).attr("y", 17).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("measured step size, frame to frame");
    gr.append("text").attr("x", 12).attr("y", 33).attr("font-size", 10).attr("fill", VC.muted)
      .text("direct geodesic distance start → end: " + VZ.fmt(geo, 4) + "°");

    const shown = methods.filter(m => m.on);
    const gmax = Math.max(0.001, ...shown.map(m => m.max));
    const bw = rightW - 30, bh = 52;
    shown.forEach((m, k) => {
      const y0 = 60 + k * 96;
      gr.append("text").attr("x", 14).attr("y", y0).attr("font-size", 11).attr("fill", m.col)
        .attr("font-weight", 600).text(m.name);
      gr.append("text").attr("x", rightW - 14).attr("y", y0).attr("text-anchor", "end").attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", VC.muted)
        .text("total " + VZ.fmt(m.total, 3) + "°");
      const w1 = bw / N;
      m.steps.forEach((s, i) => {
        const hh = Math.max(1, bh * s / gmax);
        gr.append("rect").attr("x", 14 + i * w1).attr("y", y0 + 10 + (bh - hh))
          .attr("width", Math.max(1, w1 - 0.8)).attr("height", hh)
          .attr("fill", m.col).attr("fill-opacity", 0.75);
      });
      /* the marker for the current fraction a */
      gr.append("line").attr("x1", 14 + a * bw).attr("x2", 14 + a * bw)
        .attr("y1", y0 + 6).attr("y2", y0 + 10 + bh + 3).attr("stroke", VC.ink).attr("stroke-width", 1);
      const spread = m.max / Math.max(1e-9, m.min);
      gr.append("text").attr("x", 14).attr("y", y0 + bh + 26).attr("font-size", 10)
        .attr("fill", spread < 1.001 ? VC.good : VC.muted)
        .text(spread < 1.001
          ? "every step identical — constant angular speed"
          : "fastest step is " + VZ.fmt(spread, 3) + "× the slowest"
          + (m.total > geo + 1e-3 ? ", and the path is " + VZ.fmt(100 * (m.total / geo - 1), 1) + "% longer than the geodesic" : ""));
    });

    /* ─── readout ─── */
    const dotq = VZ.dot(VZ.qFromR(R0), VZ.qFromR(R1));
    const sl = methods[0], nl = methods[1], el = methods[2];
    out.innerHTML =
      `a = ${VZ.fmt(a, 3)} · q₀·q₁ = <b>${VZ.fmt(dotq, 6)}</b> · <span class="keep">Ω</span> = `
      + `${VZ.fmt(VZ.deg(Math.acos(VZ.clamp(Math.abs(dotq), -1, 1))), 4)}° · geodesic separation of the two poses = `
      + `<b>${VZ.fmt(geo, 4)}°</b> = 2<span class="keep">Ω</span>.<br>`
      + (sl.on ? `slerp travels <b>${VZ.fmt(sl.total, 4)}°</b>${Math.abs(sl.total - (longWay ? 360 - geo : geo)) < 1e-2 ? " — exactly the geodesic" + (longWay ? " the long way, and " + VZ.fmt(sl.total, 4) + " + " + VZ.fmt(geo, 4) + " = 360.0000" : "") : ""}. ` : "")
      + (nl.on ? `nlerp travels ${VZ.fmt(nl.total, 4)}° along the <i>same</i> great circle but with steps ranging ${VZ.fmt(nl.min, 4)}°–${VZ.fmt(nl.max, 4)}°. ` : "")
      + (el.on ? `Euler-angle lerp travels <b>${VZ.fmt(el.total, 4)}°</b>, a genuinely different and ${VZ.fmt(100 * (el.total / geo - 1), 1)}% longer route.` : "")
      + (longWay ? `<br>The end matrices for q₁ and −q₁ differ by <b>0.0 exactly</b> — same orientation, opposite journey.` : "");
  }

  eA.oninput = () => { eAv.textContent = (+eA.value).toFixed(2); draw(); };
  eAng.oninput = () => { eAngv.textContent = eAng.value + "°"; draw(); };
  eYaw.oninput = draw;
  [eS, eN, eE, eL].forEach(el => el.onchange = draw);
  draw();
})();

/* ───────── 6 · the four projection models on one scene ───────── */
(function () {
  const svg = d3.select("#projfam-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 420;
  const eZ = document.getElementById("pf-z"), eZv = document.getElementById("pf-zv");
  const eO = document.getElementById("pf-off"), eOv = document.getElementById("pf-offv");
  const eS = document.getElementById("pf-sz"), eSv = document.getElementById("pf-szv");
  const eF = document.getElementById("pf-f"), eFv = document.getElementById("pf-fv");
  const out = document.getElementById("projfam-readout");

  /* a box in CAMERA coordinates: x right, y down, z forward */
  function box(cx, cy, cz, s) {
    const h = s / 2, V = [];
    for (const dz of [-h, h]) for (const dy of [-h, h]) for (const dx of [-h, h]) V.push([cx + dx, cy + dy, cz + dz]);
    const E = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    return { V: V, E: E, c: [cx, cy, cz] };
  }

  function draw() {
    const z0 = +eZ.value, off = +eO.value, sz = +eS.value, f = +eF.value;
    /* two boxes: one near and off-axis, one further back and further out */
    /* the two boxes straddle the offset symmetrically in x and sit on y = 0, so that
       setting the offset slider to zero really does put the scene centroid on the
       optical axis — which is the configuration where para-perspective degenerates
       to scaled orthography, and the figure has to be able to reach it. */
    const B = [box(off - sz * 0.95, 0, z0, sz), box(off + sz * 0.95, 0, z0 + sz * 2.6, sz)];
    /* the projection models all share ONE reference centre — the scene centroid,
       which is what "the object" means for scaled orthography and para-perspective */
    const all = B.reduce((a, b) => a.concat(b.V), []);
    const c = [0, 1, 2].map(k => all.reduce((s, p) => s + p[k], 0) / all.length);

    const persp = p => [f * p[0] / p[2], f * p[1] / p[2]];
    const sortho = p => [f * p[0] / c[2], f * p[1] / c[2]];
    const para = p => {
      const d = p[2] - c[2];
      return [(f / c[2]) * (p[0] - c[0] * d / c[2]), (f / c[2]) * (p[1] - c[1] * d / c[2])];
    };
    /* plain orthography has no depth-derived scale at all; it is shown at the same
       nominal scale so the panels are comparable, which is exactly the point:
       without a scale it cannot be compared with a real image */
    const ortho = p => [(f / c[2]) * 1.0 * p[0] * 1.0, (f / c[2]) * p[1]];

    const models = [
      { name: "orthography", fn: ortho, col: VC.muted, note: "drop Z" },
      { name: "scaled orthography", fn: sortho, col: VC.teal, note: "one global scale f/z̄" },
      { name: "para-perspective", fn: para, col: VC.a2, note: "+ first-order depth" },
      { name: "perspective", fn: persp, col: VC.good, note: "exact — divide by z" }
    ];
    /* RMS disagreement with perspective, over all sixteen corners */
    models.forEach(m => {
      let s = 0;
      all.forEach(p => { const A = m.fn(p), Bq = persp(p); s += (A[0] - Bq[0]) ** 2 + (A[1] - Bq[1]) ** 2; });
      m.rms = Math.sqrt(s / all.length);
    });

    const fr = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = fr.g;
    const pw = (fr.iw - 3 * 8) / 4, ph = 210;

    models.forEach((m, i) => {
      const gg = g.append("g").attr("transform", `translate(${i * (pw + 8)},0)`);
      gg.append("rect").attr("x", 0).attr("y", 0).attr("width", pw).attr("height", ph)
        .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.55).attr("stroke", VC.line);
      gg.append("text").attr("x", 8).attr("y", 16).attr("font-size", 11).attr("fill", m.col)
        .attr("font-weight", 600).text(m.name);
      gg.append("text").attr("x", 8).attr("y", 30).attr("font-size", 9.5).attr("fill", VC.muted).text(m.note);

      /* one shared image-space scale across all four panels, from the perspective one */
      const ref = all.map(persp);
      const xr = d3.extent(ref, p => p[0]), yr = d3.extent(ref, p => p[1]);
      const cxr = (xr[0] + xr[1]) / 2, cyr = (yr[0] + yr[1]) / 2;
      const span = Math.max(xr[1] - xr[0], yr[1] - yr[0], 1) * 2.15;
      const k = Math.min(pw - 16, ph - 62) / span;
      const sx = v => pw / 2 + (v - cxr) * k, sy = v => 38 + (ph - 62) / 2 + (v - cyr) * k;

      B.forEach((bx, bi) => {
        const P = bx.V.map(m.fn);
        bx.E.forEach(e => {
          gg.append("line").attr("x1", sx(P[e[0]][0])).attr("y1", sy(P[e[0]][1]))
            .attr("x2", sx(P[e[1]][0])).attr("y2", sy(P[e[1]][1]))
            .attr("stroke", bi === 0 ? m.col : VC.violet).attr("stroke-width", 1.3)
            .attr("stroke-opacity", 0.9);
        });
      });
      gg.append("text").attr("x", pw / 2).attr("y", ph - 10).attr("text-anchor", "middle")
        .attr("font-size", 10.5).attr("font-family", "SF Mono, Menlo, monospace")
        .attr("fill", m.rms < 1e-9 ? VC.good : (m.rms < 3 ? VC.a2 : VC.bad))
        .text(m.rms < 1e-9 ? "reference" : "RMS " + VZ.fmt(m.rms, 2) + " px");
    });

    /* ─── bottom: the top-down ray diagram ─── */
    const gb = g.append("g").attr("transform", `translate(0,${ph + 14})`);
    const bh = fr.ih - ph - 14;
    gb.append("rect").attr("x", 0).attr("y", 0).attr("width", fr.iw).attr("height", bh)
      .attr("rx", 8).attr("fill", VC.bg).attr("fill-opacity", 0.35).attr("stroke", VC.line);
    const zmax = Math.max(...all.map(p => p[2])) * 1.12;
    const xext = Math.max(2.4, ...all.map(p => Math.abs(p[0]))) * 1.5;
    const bx0 = 40, bx1 = fr.iw - 150;
    const zs = d3.scaleLinear().domain([0, zmax]).range([bx0, bx1]);
    const xs = d3.scaleLinear().domain([-xext, xext]).range([bh - 14, 14]);

    /* the image plane, drawn at z = f/(f/c[2]) = c[2] scaled — put it at a fixed small z */
    const zimg = zmax * 0.10;
    gb.append("line").attr("x1", zs(zimg)).attr("x2", zs(zimg)).attr("y1", xs(-xext * 0.75)).attr("y2", xs(xext * 0.75))
      .attr("stroke", VC.accent).attr("stroke-width", 2);
    gb.append("text").attr("x", zs(zimg) + 4).attr("y", xs(xext * 0.75) - 4).attr("font-size", 9.5)
      .attr("fill", VC.accent).text("image plane");
    gb.append("circle").attr("cx", zs(0)).attr("cy", xs(0)).attr("r", 4).attr("fill", VC.ink);
    gb.append("text").attr("x", zs(0) - 4).attr("y", xs(0) + 16).attr("text-anchor", "end")
      .attr("font-size", 9.5).attr("fill", VC.muted).text("C");
    gb.append("line").attr("x1", zs(0)).attr("x2", zs(zmax)).attr("y1", xs(0)).attr("y2", xs(0))
      .attr("stroke", VC.grid).attr("stroke-width", 1).attr("stroke-dasharray", "4 4");

    /* rays: perspective through C, orthographic family parallel to the axis (or to
       the line of sight to the centre, for para-perspective) */
    const rays = [
      { name: "perspective — through C", col: VC.good, kind: "p" },
      { name: "orthographic family — parallel", col: VC.teal, kind: "o" },
      { name: "para-perspective — parallel to C→centre", col: VC.a2, kind: "a" }
    ];
    all.forEach(p => {
      gb.append("circle").attr("cx", zs(p[2])).attr("cy", xs(p[0])).attr("r", 2)
        .attr("fill", VC.violet).attr("fill-opacity", 0.85);
      /* perspective ray */
      gb.append("line").attr("x1", zs(0)).attr("y1", xs(0)).attr("x2", zs(p[2])).attr("y2", xs(p[0]))
        .attr("stroke", VC.good).attr("stroke-width", 0.55).attr("stroke-opacity", 0.5);
      /* orthographic ray: straight back along z */
      gb.append("line").attr("x1", zs(zimg)).attr("y1", xs(p[0])).attr("x2", zs(p[2])).attr("y2", xs(p[0]))
        .attr("stroke", VC.teal).attr("stroke-width", 0.55).attr("stroke-opacity", 0.5);
      /* para-perspective ray: parallel to the line of sight to the object centre */
      const dz = p[2] - c[2], px = p[0] - (c[0] / c[2]) * dz;
      gb.append("line").attr("x1", zs(c[2])).attr("y1", xs(px)).attr("x2", zs(p[2])).attr("y2", xs(p[0]))
        .attr("stroke", VC.a2).attr("stroke-width", 0.55).attr("stroke-opacity", 0.55);
    });
    gb.append("circle").attr("cx", zs(c[2])).attr("cy", xs(c[0])).attr("r", 3.5).attr("fill", VC.a2);
    gb.append("text").attr("x", zs(c[2])).attr("y", xs(c[0]) - 8).attr("text-anchor", "middle")
      .attr("font-size", 9.5).attr("fill", VC.a2).text("object centre");
    VZ.legend(gb, rays.map(r => ({ label: r.name, color: r.col })), fr.iw - 142, 22, { gap: 14 });
    gb.append("text").attr("x", bx1).attr("y", bh - 4).attr("text-anchor", "end").attr("font-size", 9.5)
      .attr("fill", VC.muted).text("top-down view · depth z →");

    /* ─── readout ─── */
    const ratio = models[1].rms / Math.max(1e-12, models[2].rms);
    const onAxis = Math.abs(c[0]) < 1e-6 && Math.abs(c[1]) < 1e-6;
    out.innerHTML =
      `object centre (${c.map(v => VZ.fmt(v, 2)).join(", ")}) m, f = ${f} px, depth spread `
      + `${VZ.fmt(Math.max(...all.map(p => p[2])) - Math.min(...all.map(p => p[2])), 2)} m `
      + `(${VZ.fmt(100 * (Math.max(...all.map(p => p[2])) - Math.min(...all.map(p => p[2]))) / c[2], 1)}% of the distance).<br>`
      + `RMS error against exact perspective — scaled orthography <b>${VZ.fmt(models[1].rms, 3)} px</b>, `
      + `para-perspective <b>${VZ.fmt(models[2].rms, 3)} px</b>`
      + (onAxis
        ? ` — <b>identical</b>, because the para-perspective correction carries cₓ and c_y, which are zero on the optical axis.`
        : `, so para-perspective is ${VZ.fmt(ratio, 2)}× more accurate here.`)
      + `<br>Push the distance slider out and both errors fall as 1/z²: the affine models become exact in the limit, `
      + `which is the regime in which factorisation methods are allowed to be used at all.`;
  }

  eZ.oninput = () => { eZv.textContent = (+eZ.value).toFixed(1) + " m"; draw(); };
  eO.oninput = () => { eOv.textContent = (+eO.value).toFixed(2) + " m"; draw(); };
  eS.oninput = () => { eSv.textContent = (+eS.value).toFixed(2) + " m"; draw(); };
  eF.oninput = () => { eFv.textContent = eF.value + " px"; draw(); };
  draw();
})();

/* ───────── 7 · vanishing points, the horizon, and depth from a row ───────── */
(function () {
  const svg = d3.select("#persp-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 450, IW = 640, IH = 480;
  const eD = document.getElementById("pv-dir"), eDv = document.getElementById("pv-dirv");
  const eS = document.getElementById("pv-sep"), eSv = document.getElementById("pv-sepv");
  const eH = document.getElementById("pv-h"), eHv = document.getElementById("pv-hv");
  const eP = document.getElementById("pv-pitch"), ePv = document.getElementById("pv-pitchv");
  const eR = document.getElementById("pv-roll"), eRv = document.getElementById("pv-rollv");
  const eF = document.getElementById("pv-f"), eFv = document.getElementById("pv-fv");
  const ePr = document.getElementById("pv-probe");
  const out = document.getElementById("persp-readout");

  let probeY = 8;                                   // the probe's world Y, dragged along the rails

  function draw() {
    const dirD = VZ.rad(+eD.value), sep = +eS.value, h = +eH.value;
    const pitch = VZ.rad(+eP.value), roll = VZ.rad(+eR.value), f = +eF.value;
    const K = VZ.K({ f: f, cx: IW / 2, cy: IH / 2 });

    /* world: X right, Y forward, Z up. The base rotation is Rₓ(90°) as in the text;
       pitch tilts about the camera x axis, roll spins about the camera z axis. */
    const R0 = [[1, 0, 0], [0, 0, -1], [0, 1, 0]];
    const R = VZ.mul(VZ.mul(VZ.Rz(roll), VZ.Rx(pitch)), R0);
    const C = [0, -5, h];
    const t = VZ.neg(VZ.mv(R, C));
    const cam = VZ.camera({ K: K, R: R, t: t });

    /* the rail direction, in the ground plane */
    const d = [Math.sin(dirD), Math.cos(dirD), 0];
    const perp = [Math.cos(dirD), -Math.sin(dirD), 0];

    const fr = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = fr.g;
    const imgW = 560, imgH = imgW * IH / IW;
    const ox = 8, oy = (fr.ih - imgH) / 2;
    const sx = u => ox + u * imgW / IW, sy = v => oy + v * imgH / IH;

    g.append("rect").attr("x", ox).attr("y", oy).attr("width", imgW).attr("height", imgH)
      .attr("rx", 4).attr("fill", VC.bg).attr("stroke", VC.line);
    const cp = VZ.clip(svg, "pv-clip", ox, oy, imgW, imgH);
    const gi = g.append("g").attr("clip-path", cp);

    /* ── the horizon: the image of the ground plane's line at infinity ──
       computed, not assumed: two independent ground directions, and the line
       through their two vanishing points.                                     */
    function vp(dir) {
      const x = VZ.mv(K, VZ.mv(R, dir));
      return Math.abs(x[2]) < 1e-9 ? null : [x[0] / x[2], x[1] / x[2]];
    }
    const vA = vp([1, 0, 0]), vB = vp([0, 1, 0]), vD = vp(d);
    let horiz = null;
    if (vA && vB) horiz = VZ.lineThrough(vA, vB);
    else {
      /* one of them is at infinity — build the line from the finite one plus the
         image direction of the infinite one, which is the projective way */
      const inf = vA ? VZ.mv(K, VZ.mv(R, [0, 1, 0])) : VZ.mv(K, VZ.mv(R, [1, 0, 0]));
      const fin = vA || vB;
      horiz = VZ.cross([fin[0], fin[1], 1], inf);
    }
    if (horiz) {
      const [a, b, c] = horiz;
      const pts = [];
      if (Math.abs(b) > 1e-9) { pts.push([0, -c / b]); pts.push([IW, -(a * IW + c) / b]); }
      else if (Math.abs(a) > 1e-9) { pts.push([-c / a, 0]); pts.push([-c / a, IH]); }
      if (pts.length === 2) {
        gi.append("line").attr("x1", sx(pts[0][0])).attr("y1", sy(pts[0][1]))
          .attr("x2", sx(pts[1][0])).attr("y2", sy(pts[1][1]))
          .attr("stroke", VC.a2).attr("stroke-width", 1.6).attr("stroke-dasharray", "7 4");
      }
    }

    /* ── the ground: a faint grid, then the two rails and the posts ── */
    for (let i = -6; i <= 6; i++) {
      const pts = [];
      for (let s = -3; s <= 60; s += 1) {
        const p = VZ.add(VZ.scale(perp, i * 1.5), VZ.scale(d, s));
        const q = cam.project(p);
        if (q[2] > 0.05) pts.push([sx(q[0]), sy(q[1])]); else if (pts.length) break;
      }
      if (pts.length > 1) VZ.poly(gi, pts, { stroke: VC.grid, w: 0.8, close: false });
    }
    [-1, 1].forEach(side => {
      const pts = [];
      for (let s = -3; s <= 220; s += 1) {
        const p = VZ.add(VZ.scale(perp, side * sep / 2), VZ.scale(d, s));
        const q = cam.project(p);
        if (q[2] > 0.05) pts.push([sx(q[0]), sy(q[1])]); else if (pts.length) break;
      }
      if (pts.length > 1) VZ.poly(gi, pts, { stroke: VC.accent, w: 2, close: false });
    });
    for (let s = 2; s <= 40; s += 3) {
      const base = VZ.add(VZ.scale(perp, sep / 2 + 0.5), VZ.scale(d, s));
      const top = VZ.add(base, [0, 0, 1.2]);
      const a = cam.project(base), b = cam.project(top);
      if (a[2] > 0.05 && b[2] > 0.05) {
        gi.append("line").attr("x1", sx(a[0])).attr("y1", sy(a[1])).attr("x2", sx(b[0])).attr("y2", sy(b[1]))
          .attr("stroke", VC.violet).attr("stroke-width", 1.6).attr("stroke-opacity", 0.85);
      }
    }

    /* the vanishing point of the rails */
    if (vD) {
      gi.append("circle").attr("cx", sx(vD[0])).attr("cy", sy(vD[1])).attr("r", 5)
        .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 2);
      gi.append("circle").attr("cx", sx(vD[0])).attr("cy", sy(vD[1])).attr("r", 1.8).attr("fill", VC.good);
      gi.append("text").attr("x", sx(vD[0]) + 8).attr("y", sy(vD[1]) - 6).attr("font-size", 10)
        .attr("fill", VC.good).text("vanishing point");
    }
    /* the principal point */
    gi.append("path").attr("d", `M${sx(IW / 2) - 6},${sy(IH / 2)} h12 M${sx(IW / 2)},${sy(IH / 2) - 6} v12`)
      .attr("stroke", VC.muted).attr("stroke-width", 1.2);
    gi.append("text").attr("x", sx(IW / 2) + 8).attr("y", sy(IH / 2) + 14).attr("font-size", 9.5)
      .attr("fill", VC.muted).text("(cₓ, c_y)");

    /* ── the depth probe ── */
    let probe = null, zTrue = null, zEst = null, dzdv = null;
    if (ePr.checked) {
      const pw = VZ.scale(d, probeY);
      const q = cam.project(pw);
      if (q[2] > 0.05) {
        probe = q;
        zTrue = q[2];
        const dv = q[1] - IH / 2;
        zEst = Math.abs(dv) > 1e-9 ? f * h / dv : Infinity;
        dzdv = Math.abs(dv) > 1e-9 ? f * h / (dv * dv) : Infinity;
        gi.append("line").attr("x1", ox).attr("x2", ox + imgW).attr("y1", sy(q[1])).attr("y2", sy(q[1]))
          .attr("stroke", VC.bad).attr("stroke-width", 1).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.8);
        gi.append("circle").attr("cx", sx(q[0])).attr("cy", sy(q[1])).attr("r", 5.5)
          .attr("fill", VC.bad).attr("stroke", VC.bg).attr("stroke-width", 1);
      }
    }
    g.append("text").attr("x", ox + 6).attr("y", oy + 15).attr("font-size", 10.5).attr("fill", VC.muted)
      .text("640 × 480 · f = " + f + " px");

    /* ── side panel ── */
    const px0 = ox + imgW + 14, pwid = fr.iw - imgW - 22;
    const gp = g.append("g").attr("transform", `translate(${px0},${oy})`);
    gp.append("rect").attr("x", 0).attr("y", 0).attr("width", pwid).attr("height", imgH)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.6).attr("stroke", VC.line);
    let yy = 22;
    const line = (lab, val, col) => {
      gp.append("text").attr("x", 10).attr("y", yy).attr("font-size", 10).attr("fill", VC.muted).text(lab);
      gp.append("text").attr("x", 10).attr("y", yy + 15).attr("font-size", 11)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col || VC.ink).text(val);
      yy += 38;
    };
    line("vanishing point K R d", vD ? "(" + VZ.fmt(vD[0], 1) + ", " + VZ.fmt(vD[1], 1) + ")" : "at infinity",
      vD ? VC.good : VC.a2);
    const horizAtCx = horiz && Math.abs(horiz[1]) > 1e-9
      ? -(horiz[0] * IW / 2 + horiz[2]) / horiz[1] : NaN;
    line("horizon row at x = cₓ",
      isFinite(horizAtCx) ? VZ.fmt(horizAtCx, 2) + " px" : "—",
      Math.abs(horizAtCx - IH / 2) < 1e-6 ? VC.good : VC.a2);
    const tilt = horiz ? VZ.deg(Math.atan2(-horiz[0], horiz[1])) : NaN;
    line("horizon tilt", isFinite(tilt) ? VZ.fmt(tilt, 3) + "°" : "—",
      Math.abs(tilt - (+eR.value)) < 1e-6 ? VC.good : VC.ink);
    if (probe) {
      line("probe image row v", VZ.fmt(probe[1], 2) + " px", VC.bad);
      line("true depth", VZ.fmt(zTrue, 4) + " m", VC.ink);
      line("z = f·h/(v − c_y)", isFinite(zEst) ? VZ.fmt(zEst, 4) + " m" : "∞",
        (isFinite(zEst) && Math.abs(zEst - zTrue) < 1e-6) ? VC.good : VC.a2);
      line("uncertainty per pixel", isFinite(dzdv) ? VZ.fmt(dzdv, 4) + " m" : "∞",
        dzdv > 1 ? VC.bad : VC.good);
    }
    gp.append("text").attr("x", 10).attr("y", imgH - 10).attr("font-size", 9.5).attr("fill", VC.muted)
      .text("drag in the image to move the probe");

    /* drag: move the probe along the rails */
    svg.on("mousedown", null);
    svg.style("cursor", "crosshair").on("mousedown", function (ev) {
      const m = d3.pointer(ev, svg.node());
      if (m[0] < ox + 10 || m[0] > ox + imgW + 10) return;
      const vpx = (m[1] - 10 - oy) * IH / imgH;
      const dv = vpx - IH / 2;
      if (dv > 2) {
        /* the click names a DEPTH; convert it to a distance along the rails, since a
           ground point at along-rail distance s sits at depth s·cos(dir) + 5        */
        const zWant = f * h / dv;
        probeY = VZ.clamp((zWant - 5) / Math.max(0.2, Math.cos(dirD)), 1.2, 400);
      }
      draw();
    });

    /* ── readout ── */
    const level = Math.abs(+eP.value) < 1e-9 && Math.abs(+eR.value) < 1e-9;
    out.innerHTML =
      `rail direction ${eD.value}° · separation ${VZ.fmt(sep, 2)} m · camera height ${VZ.fmt(h, 2)} m · f = ${f} px<br>`
      + (vD
        ? `The rails converge at <b>(${VZ.fmt(vD[0], 2)}, ${VZ.fmt(vD[1], 2)})</b> = K R d, which does not depend on the separation slider at all.`
        : `The rails are parallel to the image plane, so their vanishing point is <b>at infinity</b> and they never converge in the picture.`)
      + `<br>The horizon crosses x = cₓ at row <b>${isFinite(horizAtCx) ? VZ.fmt(horizAtCx, 3) : "—"}</b>`
      + (level
        ? ` — exactly c_y = ${IH / 2}, because the camera is level. Move the height slider: it does not move.`
        : ` — off c_y by ${VZ.fmt(horizAtCx - IH / 2, 2)} px, and tilted by ${VZ.fmt(tilt, 3)}°, which equals the roll setting.`)
      + (probe
        ? `<br>Probe at true depth <b>${VZ.fmt(zTrue, 4)} m</b>; the row formula returns <b>${isFinite(zEst) ? VZ.fmt(zEst, 4) : "∞"} m</b>`
        + (level ? ` — exact, since the derivation assumed a level camera.` : ` — <b style="color:var(--bad)">wrong</b>, because the formula assumed no pitch and no roll and the camera has some.`)
        + ` One pixel of row error is worth <b>${isFinite(dzdv) ? VZ.fmt(dzdv, 3) : "∞"} m</b> of depth here.`
        : "");
  }

  eD.oninput = () => { eDv.textContent = eD.value + "°"; draw(); };
  eS.oninput = () => { eSv.textContent = (+eS.value).toFixed(1) + " m"; draw(); };
  eH.oninput = () => { eHv.textContent = (+eH.value).toFixed(2) + " m"; draw(); };
  eP.oninput = () => { ePv.textContent = (+eP.value).toFixed(1) + "°"; draw(); };
  eR.oninput = () => { eRv.textContent = (+eR.value).toFixed(1) + "°"; draw(); };
  eF.oninput = () => { eFv.textContent = eF.value + " px"; draw(); };
  ePr.onchange = draw;
  draw();
})();

/* ───────── 8 · the five entries of K, acting alone ───────── */
(function () {
  const svg = d3.select("#intr-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 420, IW = 640, IH = 480;
  const ids = ["fx", "fy", "cx", "cy", "s"];
  const el = {}, elv = {};
  ids.forEach(k => { el[k] = document.getElementById("kv-" + k); elv[k] = document.getElementById("kv-" + k + "v"); });
  const out = document.getElementById("intr-readout");
  const DEF = { fx: 500, fy: 500, cx: 320, cy: 240, s: 0 };

  /* one fixed scene and one fixed pose — only K moves */
  const R = [[1, 0, 0], [0, 0, -1], [0, 1, 0]];
  const C = [0, -5, 1.5], t = VZ.neg(VZ.mv(R, C));
  const GRID = [];
  for (let i = -3; i <= 3; i++) for (let j = 1; j <= 7; j++) GRID.push([i * 1.1, j * 1.6, 0]);
  const BOX = IF.house(-1.4, 6.5, 0.9, 1.5);

  function draw() {
    const v = {};
    ids.forEach(k => v[k] = +el[k].value);
    const K = [[v.fx, v.s, v.cx], [0, v.fy, v.cy], [0, 0, 1]];
    const cam = VZ.camera({ K: K, R: R, t: t });

    const fr = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = fr.g;
    const imgW = 470, imgH = imgW * IH / IW, ox = 4, oy = (fr.ih - imgH) / 2;
    const sx = u => ox + u * imgW / IW, sy = u => oy + u * imgH / IH;

    g.append("rect").attr("x", ox).attr("y", oy).attr("width", imgW).attr("height", imgH)
      .attr("rx", 4).attr("fill", VC.bg).attr("stroke", VC.line);
    const gi = g.append("g").attr("clip-path", VZ.clip(svg, "kv-clip", ox, oy, imgW, imgH));

    /* the ground grid, drawn as lines so the skew is visible on the verticals */
    for (let i = -3; i <= 3; i++) {
      const pts = [];
      for (let j = 0; j <= 8; j++) {
        const q = cam.project([i * 1.1, j * 1.6, 0]);
        if (q[2] > 0.05) pts.push([sx(q[0]), sy(q[1])]);
      }
      if (pts.length > 1) VZ.poly(gi, pts, { stroke: VC.accent, w: 1.1, close: false, op: 0.75 });
    }
    for (let j = 0; j <= 8; j++) {
      const pts = [];
      for (let i = -3; i <= 3; i++) {
        const q = cam.project([i * 1.1, j * 1.6, 0]);
        if (q[2] > 0.05) pts.push([sx(q[0]), sy(q[1])]);
      }
      if (pts.length > 1) VZ.poly(gi, pts, { stroke: VC.teal, w: 1.1, close: false, op: 0.7 });
    }
    BOX.edges.forEach(e => {
      const a = cam.project(BOX.verts[e[0]]), b = cam.project(BOX.verts[e[1]]);
      if (a[2] > 0.05 && b[2] > 0.05) {
        gi.append("line").attr("x1", sx(a[0])).attr("y1", sy(a[1])).attr("x2", sx(b[0])).attr("y2", sy(b[1]))
          .attr("stroke", VC.violet).attr("stroke-width", 1.5);
      }
    });
    /* the principal point, which is the fixed point of every f change */
    gi.append("path").attr("d", `M${sx(v.cx) - 7},${sy(v.cy)} h14 M${sx(v.cx)},${sy(v.cy) - 7} v14`)
      .attr("stroke", VC.a2).attr("stroke-width", 1.6);
    gi.append("circle").attr("cx", sx(v.cx)).attr("cy", sy(v.cy)).attr("r", 11)
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 1).attr("stroke-dasharray", "3 3");
    gi.append("text").attr("x", sx(v.cx) + 14).attr("y", sy(v.cy) + 4).attr("font-size", 10)
      .attr("fill", VC.a2).text("(cₓ, c_y)");
    /* the true image centre, for comparison */
    gi.append("circle").attr("cx", sx(IW / 2)).attr("cy", sy(IH / 2)).attr("r", 2.5)
      .attr("fill", VC.muted).attr("fill-opacity", 0.8);
    g.append("text").attr("x", ox + 6).attr("y", oy + 15).attr("font-size", 10).attr("fill", VC.muted)
      .text("640 × 480 · small grey dot = the image centre");

    /* ─── right: K and its readings ─── */
    const px0 = ox + imgW + 14, pw = fr.iw - imgW - 22;
    const gp = g.append("g").attr("transform", `translate(${px0},${oy})`);
    gp.append("rect").attr("x", 0).attr("y", 0).attr("width", pw).attr("height", imgH)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.6).attr("stroke", VC.line);
    IF.matText(gp, K, 12, 40, { dp: 1, size: 11.5, pad: 6, label: "K" });
    let yy = 106;
    const line = (lab, val, col) => {
      gp.append("text").attr("x", 12).attr("y", yy).attr("font-size", 10).attr("fill", VC.muted).text(lab);
      gp.append("text").attr("x", pw - 12).attr("y", yy).attr("text-anchor", "end").attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col || VC.ink).text(val);
      yy += 20;
    };
    line("horizontal field of view", VZ.fmt(VZ.fovFromF(v.fx, IW), 2) + "°");
    line("vertical field of view", VZ.fmt(VZ.fovFromF(v.fy, IH), 2) + "°");
    line("pixel aspect a = f_y/fₓ", VZ.fmt(v.fy / v.fx, 4),
      Math.abs(v.fy / v.fx - 1) < 1e-6 ? VC.good : VC.a2);
    /* skew, as the angle the sensor axes make away from perpendicular */
    const skewAng = VZ.deg(Math.atan2(v.s, v.fy));
    line("sensor axes off perpendicular", VZ.fmt(skewAng, 3) + "°",
      Math.abs(v.s) < 1e-9 ? VC.good : VC.a2);
    line("principal point offset", "(" + VZ.fmt(v.cx - IW / 2, 0) + ", " + VZ.fmt(v.cy - IH / 2, 0) + ") px",
      (v.cx === IW / 2 && v.cy === IH / 2) ? VC.good : VC.a2);
    line("det K = fₓ · f_y", VZ.fmt(v.fx * v.fy, 0));
    yy += 8;
    gp.append("text").attr("x", 12).attr("y", yy).attr("font-size", 10).attr("fill", VC.muted).text("K⁻¹ — pixels back to rays");
    const Ki = VZ.inv3(K);
    if (Ki) IF.matText(gp, Ki, 12, yy + 22, { dp: 6, size: 9, pad: 10 });

    /* the running example, recomputed through the CURRENT K so the reader can watch it move */
    const test = cam.project([1, 3, 0]);
    gp.append("text").attr("x", 12).attr("y", imgH - 30).attr("font-size", 10).attr("fill", VC.muted)
      .text("world (1, 3, 0), depth 8 m, lands at");
    gp.append("text").attr("x", 12).attr("y", imgH - 14).attr("font-size", 11)
      .attr("font-family", "SF Mono, Menlo, monospace")
      .attr("fill", (Math.abs(test[0] - 382.5) < 1e-6 && Math.abs(test[1] - 333.75) < 1e-6) ? VC.good : VC.ink)
      .text("(" + VZ.fmt(test[0], 3) + ", " + VZ.fmt(test[1], 3) + ")");

    /* ─── readout ─── */
    const changed = ids.filter(k => v[k] !== DEF[k]);
    out.innerHTML =
      `K = [[${v.fx}, ${v.s}, ${v.cx}], [0, ${v.fy}, ${v.cy}], [0, 0, 1]] · `
      + `fov ${VZ.fmt(VZ.fovFromF(v.fx, IW), 2)}° × ${VZ.fmt(VZ.fovFromF(v.fy, IH), 2)}°`
      + (changed.length === 0
        ? ` — the running camera exactly, and the check point lands on (382.5, 333.75) as §01 promised.`
        : ` — changed from the running camera in: <b>${changed.join(", ")}</b>.`)
      + `<br>The camera-frame ray of that check point is (1, 1.5, 8), i.e. the normalised coordinates `
      + `(0.125, 0.1875). Those never move — every slider here changes only how a ray is written down in pixels.`;
  }

  ids.forEach(k => el[k].oninput = () => { elv[k].textContent = el[k].value; draw(); });
  document.getElementById("kv-reset").onclick = () => {
    ids.forEach(k => { el[k].value = DEF[k]; elv[k].textContent = DEF[k]; });
    draw();
  };
  draw();
})();

/* ───────── 9 · focal length, field of view, and the dolly zoom ───────── */
(function () {
  const svg = d3.select("#fov-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 430;
  const eF = document.getElementById("fv-f"), eFv = document.getElementById("fv-fv");
  const eZ = document.getElementById("fv-z"), eZv = document.getElementById("fv-zv");
  const eW = document.getElementById("fv-w"), eWv = document.getElementById("fv-wv");
  const eSW = document.getElementById("fv-sw"), eSWv = document.getElementById("fv-swv");
  const eHold = document.getElementById("fv-hold");
  const out = document.getElementById("fov-readout");

  const H_SUB = 1.8, H_BG = 10, D_BG = 20, TARGET = 200;   // metres, metres, metres, pixels

  function draw() {
    const IWpx = +eW.value, IHpx = Math.round(IWpx * 3 / 4);
    const z = +eZ.value;
    /* the dolly zoom: choose f so the subject's image height is exactly TARGET px,
       scaled to the current image width so the demonstration survives a resize */
    const target = TARGET * IWpx / 640;
    let f = +eF.value;
    if (eHold.checked) {
      f = target * z / H_SUB;
      eF.value = VZ.clamp(f, 120, 2400);
      eFv.textContent = VZ.fmt(f, 0) + " px";
    }
    const hSub = f * H_SUB / z, hBg = f * H_BG / (z + D_BG);

    const fr = VZ.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const g = fr.g;

    /* ── top: the top-down cone diagram ── */
    const th = 176;
    const gt = g.append("g");
    gt.append("rect").attr("x", 0).attr("y", 0).attr("width", fr.iw).attr("height", th)
      .attr("rx", 8).attr("fill", VC.bg).attr("fill-opacity", 0.35).attr("stroke", VC.line);
    const zmax = z + D_BG + 4;
    const zs = d3.scaleLinear().domain([0, zmax]).range([46, fr.iw - 20]);
    const half = VZ.rad(VZ.fovFromF(f, IWpx)) / 2;
    const xext = Math.max(zmax * Math.tan(half), H_BG / 2) * 1.1;
    const xs = d3.scaleLinear().domain([-xext, xext]).range([th - 14, 14]);

    /* the field-of-view wedge */
    const wedge = [[zs(0), xs(0)],
    [zs(zmax), xs(zmax * Math.tan(half))],
    [zs(zmax), xs(-zmax * Math.tan(half))]];
    VZ.poly(gt, wedge, { stroke: VC.accent, w: 1.2, fill: VC.accent, fillOp: 0.09 });
    gt.append("line").attr("x1", zs(0)).attr("x2", zs(zmax)).attr("y1", xs(0)).attr("y2", xs(0))
      .attr("stroke", VC.grid).attr("stroke-dasharray", "4 4");
    gt.append("circle").attr("cx", zs(0)).attr("cy", xs(0)).attr("r", 4).attr("fill", VC.ink);
    /* the subject and the background wall, seen from above as bars */
    gt.append("line").attr("x1", zs(z)).attr("x2", zs(z)).attr("y1", xs(-0.35)).attr("y2", xs(0.35))
      .attr("stroke", VC.good).attr("stroke-width", 5);
    gt.append("text").attr("x", zs(z)).attr("y", xs(0.35) - 7).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", VC.good).text("subject " + VZ.fmt(z, 1) + " m");
    gt.append("line").attr("x1", zs(z + D_BG)).attr("x2", zs(z + D_BG))
      .attr("y1", xs(-H_BG / 2)).attr("y2", xs(H_BG / 2))
      .attr("stroke", VC.violet).attr("stroke-width", 4);
    gt.append("text").attr("x", zs(z + D_BG) - 6).attr("y", xs(H_BG / 2) - 6).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", VC.violet).text("background wall");
    /* the half-angle arc */
    const ra = 54;
    gt.append("path").attr("d", d3.arc()({
      innerRadius: ra, outerRadius: ra + 1.4, startAngle: Math.PI / 2 - half, endAngle: Math.PI / 2
    })).attr("transform", `translate(${zs(0)},${xs(0)})`).attr("fill", VC.a2);
    gt.append("text").attr("x", zs(0) + ra * 0.85).attr("y", xs(0) - ra * 0.34)
      .attr("font-size", 10.5).attr("fill", VC.a2).text("θ/2 = " + VZ.fmt(VZ.deg(half), 2) + "°");
    gt.append("text").attr("x", 10).attr("y", 16).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("top-down · the cone is set by f and the sensor width together");

    /* ── bottom left: the resulting image ── */
    const bh = fr.ih - th - 12;
    const gb = g.append("g").attr("transform", `translate(0,${th + 12})`);
    const iw = 400, ih = iw * 3 / 4;
    const scale = iw / IWpx;                     // draw the IWpx-wide image at iw on screen
    gb.append("rect").attr("x", 0).attr("y", (bh - ih) / 2).attr("width", iw).attr("height", ih)
      .attr("rx", 4).attr("fill", VC.bg).attr("stroke", VC.line);
    const gi = gb.append("g").attr("clip-path", VZ.clip(svg, "fv-clip", 0, (bh - ih) / 2, iw, ih));
    const cy0 = (bh - ih) / 2 + ih * 0.72;       // a notional ground line
    /* background wall */
    gi.append("rect")
      .attr("x", iw / 2 - (hBg * scale) / 2).attr("y", cy0 - hBg * scale)
      .attr("width", hBg * scale).attr("height", hBg * scale)
      .attr("fill", VC.violet).attr("fill-opacity", 0.22).attr("stroke", VC.violet);
    /* subject */
    gi.append("rect").attr("x", iw / 2 - hSub * scale * 0.18).attr("y", cy0 - hSub * scale)
      .attr("width", hSub * scale * 0.36).attr("height", hSub * scale)
      .attr("fill", VC.good).attr("fill-opacity", 0.35).attr("stroke", VC.good).attr("stroke-width", 1.6);
    gi.append("line").attr("x1", 0).attr("x2", iw).attr("y1", cy0).attr("y2", cy0)
      .attr("stroke", VC.grid).attr("stroke-width", 1);
    gb.append("text").attr("x", 6).attr("y", (bh - ih) / 2 + 14).attr("font-size", 10).attr("fill", VC.muted)
      .text(IWpx + " × " + IHpx + " px");
    gb.append("text").attr("x", iw / 2).attr("y", (bh - ih) / 2 + ih - 8).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", VC.good)
      .text("subject " + VZ.fmt(hSub, 1) + " px · background " + VZ.fmt(hBg, 1) + " px");

    /* ── bottom right: the numbers ── */
    const gp = gb.append("g").attr("transform", `translate(${iw + 14},0)`);
    const pw = fr.iw - iw - 14;
    gp.append("rect").attr("x", 0).attr("y", 0).attr("width", pw).attr("height", bh)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.6).attr("stroke", VC.line);
    let yy = 20;
    const line = (lab, val, col) => {
      gp.append("text").attr("x", 10).attr("y", yy).attr("font-size", 10).attr("fill", VC.muted).text(lab);
      gp.append("text").attr("x", pw - 10).attr("y", yy).attr("text-anchor", "end").attr("font-size", 10.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col || VC.ink).text(val);
      yy += 19;
    };
    const swMM = +eSW.value;
    line("f", VZ.fmt(f, 1) + " px");
    line("horizontal fov", VZ.fmt(VZ.fovFromF(f, IWpx), 3) + "°", VC.accent);
    line("vertical fov", VZ.fmt(VZ.fovFromF(f, IHpx), 3) + "°");
    line("diagonal fov", VZ.fmt(VZ.fovFromF(f, Math.hypot(IWpx, IHpx)), 3) + "°");
    line("pixel pitch", VZ.fmt(1000 * swMM / IWpx, 3) + " µm");
    line("f on this sensor", VZ.fmt(f * swMM / IWpx, 2) + " mm");
    line("35 mm equivalent", VZ.fmt(f * 36 / IWpx, 2) + " mm", VC.a2);
    yy += 6;
    line("subject height", VZ.fmt(hSub, 2) + " px", VC.good);
    line("background height", VZ.fmt(hBg, 2) + " px", VC.violet);
    line("background / subject", VZ.fmt(hBg / hSub, 4), VC.violet);
    line("predicted (h_b/h_s)(z/z_b)", VZ.fmt((H_BG / H_SUB) * (z / (z + D_BG)), 4),
      Math.abs(hBg / hSub - (H_BG / H_SUB) * (z / (z + D_BG))) < 1e-9 ? VC.good : VC.bad);

    /* ── readout ── */
    out.innerHTML =
      `f = <b>${VZ.fmt(f, 1)} px</b> on a ${IWpx}-wide image → <b>${VZ.fmt(VZ.fovFromF(f, IWpx), 3)}°</b> horizontally`
      + ` = ${VZ.fmt(f * swMM / IWpx, 2)} mm on a ${VZ.fmt(swMM, 1)} mm sensor = <b>${VZ.fmt(f * 36 / IWpx, 1)} mm</b> in 35 mm terms.<br>`
      + `Background/subject size ratio = <b>${VZ.fmt(hBg / hSub, 4)}</b>, and the formula (h_b/h_s)(z_s/z_b) predicts `
      + `<b>${VZ.fmt((H_BG / H_SUB) * (z / (z + D_BG)), 4)}</b> — f has cancelled out of it entirely.<br>`
      + (eHold.checked
        ? `<b>Dolly zoom on.</b> f is being driven to hold the subject at ${VZ.fmt(target, 0)} px. Move the distance slider: `
        + `the subject does not change size, and the background does — that difference is the only evidence in the image of which happened.`
        : `Move the focal-length slider: the ratio above does not move. Move the distance slider: it does.`);
  }

  eF.oninput = () => { eFv.textContent = eF.value + " px"; if (eHold.checked) eHold.checked = false; draw(); };
  eZ.oninput = () => { eZv.textContent = (+eZ.value).toFixed(1) + " m"; draw(); };
  eW.oninput = () => { eWv.textContent = eW.value + " px"; draw(); };
  eSW.oninput = () => { eSWv.textContent = (+eSW.value).toFixed(1) + " mm"; draw(); };
  eHold.onchange = draw;
  draw();
})();

/* ───────── 10 · the flagship: the whole chain, live ───────── */
(function () {
  const svg = d3.select("#cam-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 470, IW = 640, IH = 480;
  const gid = i => document.getElementById(i);
  const eY = gid("cm-yaw"), eYv = gid("cm-yawv"), eP = gid("cm-pitch"), ePv = gid("cm-pitchv");
  const eR = gid("cm-roll"), eRv = gid("cm-rollv"), eD = gid("cm-d"), eDv = gid("cm-dv");
  const eH = gid("cm-h"), eHv = gid("cm-hv"), eF = gid("cm-f"), eFv = gid("cm-fv");
  const eVW = gid("cm-vw"), eFr = gid("cm-frust");
  const out = gid("cam-readout");

  const HOUSE = IF.house(0, 0, 1.3, 1.8);
  const POSTS = [[-2.6, 2.2], [2.8, 1.4], [-1.2, -2.9], [3.2, -2.4]];
  const MARK = [0, 0, 1.8 + 1.3 * 0.9];            // the roof ridge, used as the check point

  function draw() {
    const yaw = VZ.rad(+eY.value), pit = VZ.rad(+eP.value), rol = VZ.rad(+eR.value);
    const dist = +eD.value, hgt = +eH.value, f = +eF.value;

    /* the camera stands on a circle of radius `dist` about the origin, at height hgt,
       looking back at the scene centre; yaw/pitch/roll are applied on top of that. */
    const C = [dist * Math.sin(yaw), -dist * Math.cos(yaw), hgt];
    const base = VZ.lookAt(C, [0, 0, 1.0], [0, 0, 1]);
    const R = VZ.mul(VZ.mul(VZ.Rz(rol), VZ.Rx(pit)), base);
    const t = VZ.neg(VZ.mv(R, C));
    const K = VZ.K({ f: f, cx: IW / 2, cy: IH / 2 });
    const cam = VZ.camera({ K: K, R: R, t: t });

    const fr = VZ.frame(svg, W, H, { l: 8, r: 8, t: 8, b: 8 });
    const g = fr.g;
    const leftW = 402, rightW = fr.iw - leftW - 12;

    /* ─── left: the world, with the camera drawn in it ─── */
    const gl = g.append("g");
    const lh = fr.ih;
    gl.append("rect").attr("x", 0).attr("y", 0).attr("width", leftW).attr("height", lh)
      .attr("rx", 8).attr("fill", VC.bg).attr("fill-opacity", 0.4).attr("stroke", VC.line);
    const view = VZ.view3({
      target: [0, 0, 1.2], yaw: VZ.rad(+eVW.value), pitch: VZ.rad(24),
      dist: Math.max(16, dist * 1.9), f: 430, cx: leftW / 2, cy: lh / 2 + 20
    });
    /* ground */
    VZ.gridPlane(view, 14, 1.2, 0).forEach(sgm => {
      if (sgm[0].z > 0.05 && sgm[1].z > 0.05)
        gl.append("line").attr("x1", sgm[0].x).attr("y1", sgm[0].y).attr("x2", sgm[1].x).attr("y2", sgm[1].y)
          .attr("stroke", VC.grid).attr("stroke-width", 0.7);
    });
    VZ.axes3(gl, view, 2.2, { labels: ["X", "Y", "Z"], colors: [VC.bad, VC.good, VC.teal], w: 1.2 });
    HOUSE.edges.forEach(e => IF.path3(gl, view, [HOUSE.verts[e[0]], HOUSE.verts[e[1]]],
      { stroke: VC.violet, w: 1.5 }));
    POSTS.forEach(p => IF.path3(gl, view, [[p[0], p[1], 0], [p[0], p[1], 1.1]], { stroke: VC.a2, w: 1.6 }));
    IF.dot3(gl, view, MARK, { r: 4, fill: VC.good, stroke: VC.bg });

    /* the camera itself: apex at C, base the four corners of the image plane pushed
       out to a visible depth, so the pyramid IS the frustum                        */
    const Rt = VZ.T(R);                                 // camera → world
    const corners = [[0, 0], [IW, 0], [IW, IH], [0, IH]];
    const zf = Math.max(2.2, dist * 0.55);
    const world = corners.map(c => {
      const ray = VZ.mv(VZ.inv3(K), [c[0], c[1], 1]);   // a direction in camera coords
      const s = zf / ray[2];
      return VZ.add(C, VZ.mv(Rt, VZ.scale(ray, s)));
    });
    if (eFr.checked) {
      world.forEach(w => IF.path3(gl, view, [C, w], { stroke: VC.accent, w: 1, op: 0.8 }));
      IF.path3(gl, view, world.concat([world[0]]), { stroke: VC.accent, w: 1.6 });
      IF.quad3(gl, view, world, { fill: VC.accent, op: 0.10 });
    }
    /* the camera's own three axes */
    [[0.9, 0, 0, VC.bad, "x"], [0, 0.9, 0, VC.good, "y"], [0, 0, 1.3, VC.teal, "z"]].forEach(d => {
      const p1 = VZ.add(C, VZ.mv(Rt, [d[0], d[1], d[2]]));
      const a = view.project(C), b = view.project(p1);
      if (a.z > 0.05 && b.z > 0.05) {
        gl.append("line").attr("x1", a.x).attr("y1", a.y).attr("x2", b.x).attr("y2", b.y)
          .attr("stroke", d[3]).attr("stroke-width", 2.2);
        gl.append("text").attr("x", b.x + 4).attr("y", b.y - 3).attr("font-size", 9.5)
          .attr("fill", d[3]).text(d[4]);
      }
    });
    IF.dot3(gl, view, C, { r: 5, fill: VC.accent, stroke: VC.bg });
    const cp = view.project(C);
    gl.append("text").attr("x", cp.x + 8).attr("y", cp.y + 14).attr("font-size", 10)
      .attr("fill", VC.accent).text("C");
    gl.append("text").attr("x", 10).attr("y", 17).attr("font-size", 11.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("the world, with the camera in it");

    /* ─── right: the image, then P ─── */
    const gr = g.append("g").attr("transform", `translate(${leftW + 12},0)`);
    const imW = rightW, imH = imW * IH / IW;
    gr.append("rect").attr("x", 0).attr("y", 0).attr("width", imW).attr("height", imH)
      .attr("rx", 4).attr("fill", VC.bg).attr("stroke", VC.line);
    const sx = u => u * imW / IW, sy = u => u * imH / IH;
    const gi = gr.append("g").attr("clip-path", VZ.clip(svg, "cm-clip", 0, 0, imW, imH));

    const seen = p => { const q = cam.project(p); return q[2] > 0.05 ? q : null; };
    for (let i = -7; i <= 7; i++) {
      [[[i * 1.2, -8.4, 0], [i * 1.2, 8.4, 0]], [[-8.4, i * 1.2, 0], [8.4, i * 1.2, 0]]].forEach(sgm => {
        const a = seen(sgm[0]), b = seen(sgm[1]);
        if (a && b) gi.append("line").attr("x1", sx(a[0])).attr("y1", sy(a[1]))
          .attr("x2", sx(b[0])).attr("y2", sy(b[1])).attr("stroke", VC.grid).attr("stroke-width", 0.8);
      });
    }
    HOUSE.edges.forEach(e => {
      const a = seen(HOUSE.verts[e[0]]), b = seen(HOUSE.verts[e[1]]);
      if (a && b) gi.append("line").attr("x1", sx(a[0])).attr("y1", sy(a[1]))
        .attr("x2", sx(b[0])).attr("y2", sy(b[1])).attr("stroke", VC.violet).attr("stroke-width", 1.6);
    });
    POSTS.forEach(p => {
      const a = seen([p[0], p[1], 0]), b = seen([p[0], p[1], 1.1]);
      if (a && b) gi.append("line").attr("x1", sx(a[0])).attr("y1", sy(a[1]))
        .attr("x2", sx(b[0])).attr("y2", sy(b[1])).attr("stroke", VC.a2).attr("stroke-width", 1.6);
    });
    const mk = seen(MARK);
    if (mk) {
      gi.append("circle").attr("cx", sx(mk[0])).attr("cy", sy(mk[1])).attr("r", 5)
        .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 2);
    }
    gi.append("path").attr("d", `M${sx(IW / 2) - 6},${sy(IH / 2)} h12 M${sx(IW / 2)},${sy(IH / 2) - 6} v12`)
      .attr("stroke", VC.muted).attr("stroke-width", 1.2);
    gr.append("text").attr("x", 6).attr("y", 14).attr("font-size", 10).attr("fill", VC.muted)
      .text("what the camera records · 640 × 480");

    /* P and its certificates */
    const gp = gr.append("g").attr("transform", `translate(0,${imH + 12})`);
    const gph = fr.ih - imH - 12;
    gp.append("rect").attr("x", 0).attr("y", 0).attr("width", imW).attr("height", gph)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.6).attr("stroke", VC.line);
    IF.matText(gp, cam.P, 10, 26, { dp: 2, size: 9.5, pad: 9, label: "P = K[R|t]" });
    const zero = VZ.mv(cam.P, C.concat([1]));
    const nz = Math.max(...zero.map(Math.abs));
    let yy = 82;
    const line = (lab, val, col) => {
      gp.append("text").attr("x", 10).attr("y", yy).attr("font-size", 9.5).attr("fill", VC.muted).text(lab);
      gp.append("text").attr("x", imW - 10).attr("y", yy).attr("text-anchor", "end").attr("font-size", 9.5)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col || VC.ink).text(val);
      yy += 15;
    };
    line("t (world origin, in camera coords)", "(" + t.map(v => VZ.fmt(v, 3)).join(", ") + ")", VC.a2);
    line("C = −Rᵀt (camera centre, in world)", "(" + cam.C.map(v => VZ.fmt(v, 3)).join(", ") + ")", VC.accent);
    line("‖P·(C, 1)ᵀ‖∞", nz.toExponential(1), nz < 1e-9 ? VC.good : VC.bad);
    line("optical axis r₃", "(" + R[2].map(v => VZ.fmt(v, 3)).join(", ") + ")");
    line("marked point → pixel", mk ? "(" + VZ.fmt(mk[0], 2) + ", " + VZ.fmt(mk[1], 2) + ")" : "behind the camera",
      mk ? VC.good : VC.bad);
    line("its depth", mk ? VZ.fmt(mk[2], 3) + " m" : "—");
    line("horizontal fov", VZ.fmt(VZ.fovFromF(f, IW), 2) + "°");

    /* ─── readout ─── */
    const inFrame = mk && mk[0] >= 0 && mk[0] <= IW && mk[1] >= 0 && mk[1] <= IH;
    out.innerHTML =
      `C = (${cam.C.map(v => VZ.fmt(v, 3)).join(", ")}) m, t = (${t.map(v => VZ.fmt(v, 3)).join(", ")}) — `
      + `different vectors, and only the first one is where the camera is standing. `
      + `P·(C, 1)ᵀ = (${zero.map(v => v.toExponential(0)).join(", ")}), i.e. the zero vector: `
      + `<b>the camera centre is the null space of P</b>.<br>`
      + `The marked roof point is at depth <b>${mk ? VZ.fmt(mk[2], 3) : "—"} m</b> and lands at `
      + `<b>${mk ? "(" + VZ.fmt(mk[0], 2) + ", " + VZ.fmt(mk[1], 2) + ")" : "—"}</b>`
      + (mk ? (inFrame ? " — inside the frame." : " — <b>outside the 640 × 480 frame</b>: projection is defined there, the sensor is not.") : ".")
      + ` Field of view ${VZ.fmt(VZ.fovFromF(f, IW), 2)}° × ${VZ.fmt(VZ.fovFromF(f, IH), 2)}°.`;
  }

  eY.oninput = () => { eYv.textContent = eY.value + "°"; draw(); };
  eP.oninput = () => { ePv.textContent = (+eP.value).toFixed(1) + "°"; draw(); };
  eR.oninput = () => { eRv.textContent = (+eR.value).toFixed(1) + "°"; draw(); };
  eD.oninput = () => { eDv.textContent = (+eD.value).toFixed(1) + " m"; draw(); };
  eH.oninput = () => { eHv.textContent = (+eH.value).toFixed(1) + " m"; draw(); };
  eF.oninput = () => { eFv.textContent = eF.value + " px"; draw(); };
  eVW.oninput = draw; eFr.onchange = draw;
  draw();
})();

/* ───────── 11 · distortion, undistortion, and the fold ───────── */
(function () {
  const svg = d3.select("#dist-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 470, IW = 640, IH = 480, F = 500, CX = 320, CY = 240;
  const gid = i => document.getElementById(i);
  const eK1 = gid("ds-k1"), eK1v = gid("ds-k1v"), eK2 = gid("ds-k2"), eK2v = gid("ds-k2v");
  const eP1 = gid("ds-p1"), eP1v = gid("ds-p1v"), eP2 = gid("ds-p2"), eP2v = gid("ds-p2v");
  const eU = gid("ds-undo"), eFl = gid("ds-field");
  const out = gid("dist-readout");

  const RMAX = Math.hypot(CX, CY) / F;          // 0.8 exactly, for this K
  const px = (x, y) => [F * x + CX, F * y + CY];
  const nm = (u, v) => [(u - CX) / F, (v - CY) / F];

  function draw() {
    const k = { k1: +eK1.value, k2: +eK2.value, p1: +eP1.value, p2: +eP2.value };

    const fr = VZ.frame(svg, W, H, { l: 8, r: 8, t: 8, b: 8 });
    const g = fr.g;
    const leftW = 452, rightW = fr.iw - leftW - 14;
    const imH = leftW * IH / IW;
    const oy = 14;

    /* ── left: the grid, distorted, plus the undistortion overlay ── */
    const gl = g.append("g").attr("transform", `translate(0,${oy})`);
    gl.append("rect").attr("x", 0).attr("y", 0).attr("width", leftW).attr("height", imH)
      .attr("rx", 4).attr("fill", VC.bg).attr("stroke", VC.line);
    const sx = u => u * leftW / IW, sy = v => v * imH / IH;
    const gi = gl.append("g").attr("clip-path", VZ.clip(svg, "ds-clip", 0, 0, leftW, imH));

    const NX = 13, NY = 10;
    /* the straight reference grid */
    const lines = [];
    for (let i = 0; i <= NX; i++) lines.push({ horiz: false, t: i / NX });
    for (let j = 0; j <= NY; j++) lines.push({ horiz: true, t: j / NY });
    lines.forEach(L => {
      const straight = [], distorted = [], undone = [];
      for (let s = 0; s <= 60; s++) {
        const u = L.horiz ? (s / 60) * IW : L.t * IW;
        const v = L.horiz ? L.t * IH : (s / 60) * IH;
        const n = nm(u, v);
        straight.push([sx(u), sy(v)]);
        const d = VZ.distort(n[0], n[1], k);
        const dp = px(d[0], d[1]);
        distorted.push([sx(dp[0]), sy(dp[1])]);
        const un = VZ.undistort(d[0], d[1], k, 20);
        const up = px(un[0], un[1]);
        if (isFinite(up[0]) && isFinite(up[1]) && Math.abs(up[0]) < 1e6 && Math.abs(up[1]) < 1e6)
          undone.push([sx(up[0]), sy(up[1])]);
      }
      VZ.poly(gi, straight, { stroke: VC.grid, w: 0.8, close: false });
      VZ.poly(gi, distorted, { stroke: VC.accent, w: 1.5, close: false });
      if (eU.checked && undone.length > 2)
        VZ.poly(gi, undone, { stroke: VC.good, w: 1.2, close: false, dash: "4 3" });
    });

    if (eFl.checked) {
      for (let i = 1; i < 10; i++) for (let j = 1; j < 8; j++) {
        const u = i * IW / 10, v = j * IH / 8, n = nm(u, v);
        const d = VZ.distort(n[0], n[1], k), dp = px(d[0], d[1]);
        VZ.arrow(gi, sx(u), sy(v), sx(u + (dp[0] - u) * 2.2), sy(v + (dp[1] - v) * 2.2),
          { color: VC.a2, w: 1, head: 4, op: 0.9 });
      }
      gl.append("text").attr("x", 8).attr("y", imH - 8).attr("font-size", 9.5).attr("fill", VC.a2)
        .text("displacement × 2.2 for visibility");
    }
    /* the principal point, which is the centre of the radial part */
    gi.append("path").attr("d", `M${sx(CX) - 6},${sy(CY)} h12 M${sx(CX)},${sy(CY) - 6} v12`)
      .attr("stroke", VC.muted).attr("stroke-width", 1.2);
    gl.append("rect").attr("x", 4).attr("y", 6).attr("width", 158).attr("height", 50).attr("rx", 5)
      .attr("fill", VC.bg).attr("fill-opacity", 0.82).attr("stroke", VC.line);
    VZ.legend(gl, [
      { label: "ideal, straight", color: VC.grid },
      { label: "as the lens records it", color: VC.accent },
      { label: "after undistortion", color: VC.good, dash: "4 3" }
    ], 12, 20, { gap: 14 });

    /* ── measure the round trip over the whole frame ── */
    let worst = 0, worstAt = null;
    for (let i = 0; i <= 40; i++) for (let j = 0; j <= 30; j++) {
      const u = i * IW / 40, v = j * IH / 30, n = nm(u, v);
      const d = VZ.distort(n[0], n[1], k);
      const un = VZ.undistort(d[0], d[1], k, 20);
      const e = F * Math.hypot(un[0] - n[0], un[1] - n[1]);
      if (isFinite(e) && e > worst) { worst = e; worstAt = [u, v]; }
      if (!isFinite(e)) { worst = Infinity; worstAt = [u, v]; }
    }
    /* The fold radius: the smallest positive r where dr̂/dr = 1 + 3κ₁r² + 5κ₂r⁴ turns
       negative. With κ₂ = 0 this is the closed form 1/√(−3κ₁); with κ₂ ≠ 0 it moves,
       so it is found numerically rather than quoted from the cubic-only formula.   */
    let fold = Infinity;
    for (let i = 1; i <= 6000; i++) {
      const r = i * 4 / 6000;
      if (1 + 3 * k.k1 * r * r + 5 * k.k2 * Math.pow(r, 4) <= 0) { fold = r; break; }
    }
    const foldCubic = k.k1 < 0 ? 1 / Math.sqrt(-3 * k.k1) : Infinity;

    /* ── right: the radial map ── */
    const gr = g.append("g").attr("transform", `translate(${leftW + 14},${oy})`);
    const rh = imH;
    gr.append("rect").attr("x", 0).attr("y", 0).attr("width", rightW).attr("height", rh)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.6).attr("stroke", VC.line);
    const m = { l: 40, r: 14, t: 26, b: 34 };
    const pw = rightW - m.l - m.r, phh = rh - m.t - m.b - 92;
    const gg = gr.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const rMaxPlot = Math.max(1.05, Math.min(2.0, isFinite(fold) ? fold * 1.35 : 1.4));
    const X = d3.scaleLinear().domain([0, rMaxPlot]).range([0, pw]);
    const rr = r => r * (1 + k.k1 * r * r + k.k2 * r ** 4);
    const ys = [];
    for (let i = 0; i <= 200; i++) ys.push(rr(i * rMaxPlot / 200));
    const Y = d3.scaleLinear().domain([0, Math.max(1.05, ...ys) * 1.05]).range([phh, 0]);
    VZ.gridY(gg, Y, pw, 4); VZ.axisB(gg, X, phh, 4, "ideal r"); VZ.axisL(gg, Y, 4);
    /* the identity, for reference */
    gg.append("line").attr("x1", X(0)).attr("y1", Y(0)).attr("x2", X(rMaxPlot)).attr("y2", Y(rMaxPlot))
      .attr("stroke", VC.grid).attr("stroke-dasharray", "4 4");
    const curve = [];
    for (let i = 0; i <= 200; i++) { const r = i * rMaxPlot / 200; curve.push([X(r), Y(rr(r))]); }
    VZ.poly(gg, curve, { stroke: VC.accent, w: 2, close: false });
    /* the frame's maximum radius */
    gg.append("line").attr("x1", X(RMAX)).attr("x2", X(RMAX)).attr("y1", 0).attr("y2", phh)
      .attr("stroke", VC.violet).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
    gg.append("text").attr("x", X(RMAX) + 4).attr("y", 11).attr("font-size", 9.5)
      .attr("fill", VC.violet).text("frame corner r = " + VZ.fmt(RMAX, 2));
    if (isFinite(fold) && fold < rMaxPlot) {
      gg.append("line").attr("x1", X(fold)).attr("x2", X(fold)).attr("y1", 0).attr("y2", phh)
        .attr("stroke", VC.bad).attr("stroke-width", 1.6);
      gg.append("text").attr("x", X(fold) + 4).attr("y", 26).attr("font-size", 9.5)
        .attr("fill", VC.bad).text("fold at " + VZ.fmt(fold, 3));
      gg.append("rect").attr("x", X(fold)).attr("y", 0).attr("width", Math.max(0, pw - X(fold)))
        .attr("height", phh).attr("fill", VC.bad).attr("fill-opacity", 0.10);
    }
    gr.append("text").attr("x", 12).attr("y", 16).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("r̂(r) = r(1 + κ₁r² + κ₂r⁴)");
    gr.append("text").attr("x", rightW - 12).attr("y", 16).attr("text-anchor", "end")
      .attr("font-size", 9.5).attr("fill", VC.muted).text("observed r̂ ↑");

    let yy = m.t + phh + 56;
    const line = (lab, val, col) => {
      gr.append("text").attr("x", 12).attr("y", yy).attr("font-size", 9.5).attr("fill", VC.muted).text(lab);
      gr.append("text").attr("x", rightW - 12).attr("y", yy).attr("text-anchor", "end").attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col || VC.ink).text(val);
      yy += 17;
    };
    const kind = k.k1 < -1e-9 ? "barrel" : (k.k1 > 1e-9 ? "pincushion" : "none");
    line("distortion type", kind, k.k1 < 0 ? VC.accent : (k.k1 > 0 ? VC.a2 : VC.good));
    line("fold radius (dr̂/dr = 0)", isFinite(fold) ? VZ.fmt(fold, 4) : "none · monotone",
      isFinite(fold) && fold <= RMAX ? VC.bad : VC.good);
    line("   the κ₂ = 0 closed form", isFinite(foldCubic) ? VZ.fmt(foldCubic, 4) : "—", VC.muted);
    line("frame's largest radius", VZ.fmt(RMAX, 4), VC.violet);
    line("corner shift", (function () {
      const n = nm(0, 0), d = VZ.distort(n[0], n[1], k), p = px(d[0], d[1]);
      return VZ.fmt(Math.hypot(p[0] - 0, p[1] - 0), 2) + " px";
    })(), VC.a2);
    line("worst round-trip error", isFinite(worst) ? (worst < 1e-6 ? worst.toExponential(1) + " px" : VZ.fmt(worst, 4) + " px") : "diverged",
      worst < 1e-6 ? VC.good : (worst < 0.5 ? VC.a2 : VC.bad));

    /* ── readout ── */
    const past = isFinite(fold) && fold <= RMAX;
    out.innerHTML =
      `<span class="keep">κ</span>₁ = ${VZ.fmt(k.k1, 3)}, <span class="keep">κ</span>₂ = ${VZ.fmt(k.k2, 3)}, `
      + `p₁ = ${VZ.fmt(k.p1, 3)}, p₂ = ${VZ.fmt(k.p2, 3)} · <b>${kind}</b> distortion.<br>`
      + (past
        ? `<b style="color:var(--bad)">The frame extends past the fold.</b> dr̂/dr first vanishes at r = ${VZ.fmt(fold, 4)} but the corners `
        + `reach r = ${VZ.fmt(RMAX, 4)}, so beyond radius ${VZ.fmt(fold, 3)} two different ideal radii produce the same `
        + `observed radius. The map is not injective there and the worst round-trip error is `
        + `<b>${isFinite(worst) ? VZ.fmt(worst, 2) : "∞"} px</b> — no iteration count fixes this.`
        : `The whole frame lies inside the invertible region`
        + (isFinite(fold) ? ` (fold at ${VZ.fmt(fold, 4)}, corners at ${VZ.fmt(RMAX, 4)})` : ` (dr̂/dr never vanishes — the radial map is increasing everywhere)`)
        + `, and twenty passes of the fixed point recover every ideal point to <b>${worst.toExponential(1)} px</b>.`);
  }

  const upd = (e, v, d) => { e.oninput = () => { v.textContent = (+e.value).toFixed(d).replace("-", "−"); draw(); }; };
  upd(eK1, eK1v, 3); upd(eK2, eK2v, 3); upd(eP1, eP1v, 3); upd(eP2, eP2v, 3);
  eU.onchange = draw; eFl.onchange = draw;
  gid("ds-break").onclick = () => { eK1.value = -0.8; eK1v.textContent = "−0.800"; draw(); };
  draw();
})();

/* ───────── 12 · the thin lens, focus and depth of field ───────── */
(function () {
  const svg = d3.select("#lens-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 440;
  const gid = i => document.getElementById(i);
  const eF = gid("ln-f"), eFv = gid("ln-fv"), eZ = gid("ln-z"), eZv = gid("ln-zv");
  const eN = gid("ln-N"), eZo = gid("ln-zo"), eZov = gid("ln-zov");
  const eC = gid("ln-c"), eCv = gid("ln-cv");
  const out = gid("lens-readout");

  function draw() {
    const f = +eF.value, N = +eN.value, c = +eC.value;         // mm, —, mm
    const zFocus = +eZ.value * 1000, zObj = +eZo.value * 1000; // mm
    const d = f / N;
    const zi = 1 / (1 / f - 1 / zFocus);                        // sensor position
    const ziObj = 1 / (1 / f - 1 / zObj);                       // where the test object focuses
    const coc = d * Math.abs(ziObj - zi) / ziObj;               // blur circle at the sensor
    const Hyp = f * f / (N * c) + f;
    const near = zFocus * (Hyp - f) / (Hyp + zFocus - 2 * f);
    const far = zFocus >= Hyp ? Infinity : zFocus * (Hyp - f) / (Hyp - zFocus);

    const fr = VZ.frame(svg, W, H, { l: 8, r: 8, t: 8, b: 8 });
    const g = fr.g;

    /* ── top: the ray diagram, drawn in a squashed but honest geometry ── */
    const th = 224;
    const gt = g.append("g");
    gt.append("rect").attr("x", 0).attr("y", 0).attr("width", fr.iw).attr("height", th)
      .attr("rx", 8).attr("fill", VC.bg).attr("fill-opacity", 0.35).attr("stroke", VC.line);
    const lensX = fr.iw * 0.60, axisY = th / 2;
    /* object side uses a log-ish scale so 0.4 m and 12 m both fit; image side is linear
       in millimetres, magnified, because that is where the interesting millimetre is  */
    const objScale = (z) => lensX - 40 - (fr.iw * 0.52 - 40) * Math.log(1 + z / 400) / Math.log(1 + 12000 / 400);
    const apPix = VZ.clamp(d * 1.6, 12, 88);
    /* The image side spans fractions of a millimetre — at any honest scale the sensor
       and the object's focal point would be one pixel apart. So the image side is
       magnified about the sensor position by whatever factor makes the separation
       visible, and everything drawn there uses that ONE factor, so the similar
       triangles in the picture are still the similar triangles in the algebra.      */
    const sensX = lensX + 150;
    const dz = ziObj - zi;
    const magImg = VZ.clamp(Math.abs(dz) > 1e-6 ? 78 / Math.abs(dz) : 1e4, 6, 4000);
    const imgScale = (z) => sensX + (z - zi) * magImg;

    /* the lens */
    gt.append("ellipse").attr("cx", lensX).attr("cy", axisY).attr("rx", 7).attr("ry", apPix / 2)
      .attr("fill", VC.accent).attr("fill-opacity", 0.18).attr("stroke", VC.accent).attr("stroke-width", 1.6);
    gt.append("line").attr("x1", 20).attr("x2", fr.iw - 14).attr("y1", axisY).attr("y2", axisY)
      .attr("stroke", VC.grid).attr("stroke-dasharray", "4 4");
    gt.append("text").attr("x", lensX).attr("y", axisY - apPix / 2 - 8).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", VC.accent).text("d = " + VZ.fmt(d, 1) + " mm");

    /* the sensor, at z_i */
    gt.append("line").attr("x1", sensX).attr("x2", sensX).attr("y1", axisY - 62).attr("y2", axisY + 62)
      .attr("stroke", VC.ink).attr("stroke-width", 2.4);
    gt.append("text").attr("x", sensX).attr("y", axisY - 68).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", VC.ink).text("sensor · z_i = " + VZ.fmt(zi, 3) + " mm");

    /* rays from the FOCUS plane: they converge exactly on the sensor */
    const oy1 = axisY - 40;
    [-1, 0, 1].forEach(k => {
      const ay = axisY + k * apPix / 2;
      gt.append("line").attr("x1", objScale(zFocus)).attr("y1", oy1).attr("x2", lensX).attr("y2", ay)
        .attr("stroke", VC.good).attr("stroke-width", 1).attr("stroke-opacity", 0.85);
      gt.append("line").attr("x1", lensX).attr("y1", ay).attr("x2", sensX).attr("y2", axisY + 34)
        .attr("stroke", VC.good).attr("stroke-width", 1).attr("stroke-opacity", 0.85);
    });
    gt.append("circle").attr("cx", objScale(zFocus)).attr("cy", oy1).attr("r", 4).attr("fill", VC.good);
    gt.append("text").attr("x", objScale(zFocus)).attr("y", oy1 - 8).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", VC.good).text("focus " + VZ.fmt(zFocus / 1000, 2) + " m");

    /* Rays from the TEST object. They converge at z_i(obj), which is NOT where the
       sensor is, so by the time they reach the sensor they have spread back out into
       a disc — and the two extreme aperture rays bracket exactly that disc. Both the
       cone and the blur segment are drawn from the same two lines, so the picture
       cannot disagree with the number.                                              */
    const convX = imgScale(ziObj);
    const oy2 = axisY + 44;
    const convY = axisY - 22;                              // the image point, inverted about the axis
    const ends = [];
    [-1, 1].forEach(k => {
      const ay = axisY + k * apPix / 2;
      gt.append("line").attr("x1", objScale(zObj)).attr("y1", oy2).attr("x2", lensX).attr("y2", ay)
        .attr("stroke", VC.a2).attr("stroke-width", 1).attr("stroke-opacity", 0.75);
      const far = Math.max(convX, sensX) + 26;
      const tFar = (far - lensX) / Math.max(1e-6, convX - lensX);
      gt.append("line").attr("x1", lensX).attr("y1", ay)
        .attr("x2", far).attr("y2", ay + tFar * (convY - ay))
        .attr("stroke", VC.a2).attr("stroke-width", 1).attr("stroke-opacity", 0.75);
      const tSen = (sensX - lensX) / Math.max(1e-6, convX - lensX);
      ends.push(ay + tSen * (convY - ay));
    });
    gt.append("circle").attr("cx", convX).attr("cy", convY).attr("r", 2.6)
      .attr("fill", VC.a2).attr("fill-opacity", 0.9);
    gt.append("text").attr("x", convX).attr("y", convY - 8).attr("text-anchor", "middle")
      .attr("font-size", 9).attr("fill", VC.a2).text("it focuses here");
    gt.append("circle").attr("cx", objScale(zObj)).attr("cy", oy2).attr("r", 4).attr("fill", VC.a2);
    gt.append("text").attr("x", objScale(zObj)).attr("y", oy2 + 16).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", VC.a2).text("object " + VZ.fmt(zObj / 1000, 2) + " m");
    /* the blur patch on the sensor — its ends are the two rays traced above */
    const sharp = coc <= c;
    gt.append("line").attr("x1", sensX).attr("x2", sensX)
      .attr("y1", Math.min(ends[0], ends[1])).attr("y2", Math.max(ends[0], ends[1]))
      .attr("stroke", sharp ? VC.good : VC.bad).attr("stroke-width", 5).attr("stroke-linecap", "round");
    /* keep the label inside the frame when the sensor sits near the right edge */
    const lblRight = sensX + 10 > fr.iw - 190;
    gt.append("text").attr("x", lblRight ? sensX - 10 : sensX + 10).attr("y", axisY - 46)
      .attr("text-anchor", lblRight ? "end" : "start").attr("font-size", 10)
      .attr("fill", sharp ? VC.good : VC.bad)
      .text("c = " + VZ.fmt(coc, 4) + " mm" + (sharp ? "  ✓ acceptable" : "  ✗ visibly soft"));
    gt.append("text").attr("x", 10).attr("y", 16).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("1/z_o + 1/z_i = 1/f");
    gt.append("text").attr("x", fr.iw - 12).attr("y", 16).attr("text-anchor", "end").attr("font-size", 9.5)
      .attr("fill", VC.muted).text("object side log-compressed · image side magnified ×" + VZ.fmt(magImg, 0));

    /* ── bottom left: the depth-of-field bar ── */
    const bh = fr.ih - th - 12;
    const gb = g.append("g").attr("transform", `translate(0,${th + 12})`);
    const bw = fr.iw * 0.60;
    gb.append("rect").attr("x", 0).attr("y", 0).attr("width", bw).attr("height", bh)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.6).attr("stroke", VC.line);
    const zmax = Math.max(13000, zFocus * 1.6, zObj * 1.25);
    const Z = d3.scaleLog().domain([300, zmax]).range([46, bw - 20]).clamp(true);
    const barY = bh / 2;
    gb.append("line").attr("x1", Z(300)).attr("x2", Z(zmax)).attr("y1", barY).attr("y2", barY)
      .attr("stroke", VC.grid).attr("stroke-width", 3);
    gb.append("rect").attr("x", Z(near)).attr("y", barY - 11)
      .attr("width", Math.max(2, Z(isFinite(far) ? far : zmax) - Z(near))).attr("height", 22)
      .attr("fill", VC.good).attr("fill-opacity", 0.3).attr("stroke", VC.good);
    [[zFocus, VC.ink, "focus"], [zObj, sharp ? VC.good : VC.bad, "object"]].forEach((it, i) => {
      gb.append("line").attr("x1", Z(it[0])).attr("x2", Z(it[0])).attr("y1", barY - 20).attr("y2", barY + 20)
        .attr("stroke", it[1]).attr("stroke-width", 2);
      gb.append("text").attr("x", Z(it[0])).attr("y", barY + (i ? 36 : -26)).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", it[1]).text(it[2]);
    });
    const ax = gb.append("g").attr("transform", `translate(0,${bh - 22})`);
    ax.call(d3.axisBottom(Z).tickValues([500, 1000, 2000, 5000, 10000]).tickFormat(v => (v / 1000) + " m"))
      .attr("class", "axis");
    gb.append("text").attr("x", 10).attr("y", 16).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("acceptably sharp zone");

    /* ── bottom right: numbers ── */
    const gp = gb.append("g").attr("transform", `translate(${bw + 12},0)`);
    const pw = fr.iw - bw - 12;
    gp.append("rect").attr("x", 0).attr("y", 0).attr("width", pw).attr("height", bh)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.6).attr("stroke", VC.line);
    let yy = 18;
    const line = (lab, val, col) => {
      gp.append("text").attr("x", 10).attr("y", yy).attr("font-size", 9.5).attr("fill", VC.muted).text(lab);
      gp.append("text").attr("x", pw - 10).attr("y", yy).attr("text-anchor", "end").attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col || VC.ink).text(val);
      yy += 16;
    };
    line("aperture d = f/N", VZ.fmt(d, 2) + " mm");
    line("sensor distance z_i", VZ.fmt(zi, 4) + " mm");
    line("hyperfocal H", VZ.fmt(Hyp / 1000, 3) + " m", VC.a2);
    line("near limit", VZ.fmt(near / 1000, 3) + " m", VC.good);
    line("far limit", isFinite(far) ? VZ.fmt(far / 1000, 3) + " m" : "∞", VC.good);
    line("total depth of field", isFinite(far) ? VZ.fmt((far - near) / 1000, 3) + " m" : "infinite");
    line("blur at the object", VZ.fmt(coc, 4) + " mm", sharp ? VC.good : VC.bad);
    line("… in 4 µm pixels", VZ.fmt(coc / 0.004, 1) + " px", sharp ? VC.good : VC.bad);
    line("relative light (1/N²)", VZ.fmt(1 / (N * N), 4));
    line("corner vignetting cos⁴α", VZ.fmt(Math.pow(Math.cos(Math.atan(400 / 500)), 4), 4), VC.a2);

    /* ── readout ── */
    out.innerHTML =
      `f = ${f} mm at f/${N} focused at ${VZ.fmt(zFocus / 1000, 2)} m: the sensor sits at `
      + `<b>${VZ.fmt(zi, 4)} mm</b>, which is ${VZ.fmt(zi - f, 4)} mm beyond the focal length.<br>`
      + `Depth of field runs <b>${VZ.fmt(near / 1000, 3)} m</b> to <b>${isFinite(far) ? VZ.fmt(far / 1000, 3) + " m" : "infinity"}</b>`
      + (isFinite(far) ? ` — a span of ${VZ.fmt((far - near) / 1000, 3)} m` : ` — the focus distance has reached the hyperfocal distance of ${VZ.fmt(Hyp / 1000, 2)} m`)
      + `. The test object at ${VZ.fmt(zObj / 1000, 2)} m blurs to <b>${VZ.fmt(coc, 4)} mm</b> `
      + (sharp ? `≤ the ${VZ.fmt(c, 3)} mm criterion, so it is inside the sharp zone.`
        : `> the ${VZ.fmt(c, 3)} mm criterion — <b style="color:var(--bad)">outside</b>, and every high-frequency feature on it is gone.`)
      + `<br>Stopping down one stop roughly doubles that span and costs half the light; the corner already `
      + `receives only ${VZ.fmt(100 * Math.pow(Math.cos(Math.atan(400 / 500)), 4), 1)}% of the centre's, from cos⁴<span class="keep">α</span> alone.`;
  }

  eF.oninput = () => { eFv.textContent = eF.value + " mm"; draw(); };
  eZ.oninput = () => { eZv.textContent = (+eZ.value).toFixed(2) + " m"; draw(); };
  eZo.oninput = () => { eZov.textContent = (+eZo.value).toFixed(2) + " m"; draw(); };
  eC.oninput = () => { eCv.textContent = (+eC.value).toFixed(3) + " mm"; draw(); };
  eN.onchange = draw;
  draw();
})();

/* ───────── 13 · the Phong lobes, Cartesian and polar ───────── */
(function () {
  const svg = d3.select("#brdf-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 410;
  const gid = i => document.getElementById(i);
  const eL = gid("br-li"), eLv = gid("br-liv"), eK = gid("br-ke"), eKv = gid("br-kev");
  const eA = gid("br-ka"), eAv = gid("br-kav"), eD = gid("br-kd"), eDv = gid("br-kdv");
  const eS = gid("br-ks"), eSv = gid("br-ksv"), eG = gid("br-gauss");
  const out = gid("brdf-readout");

  function draw() {
    const thI = VZ.rad(+eL.value), ke = +eK.value;
    const ka = +eA.value, kd = +eD.value, ks = +eS.value, gauss = eG.checked;

    /* everything in the plane containing the normal and the light. n̂ = (0, 1);
       the light arrives from angle thI, and the mirror direction is at −thI.    */
    const n = [0, 1];
    const vi = [Math.sin(thI), Math.cos(thI)];
    /* ŝ = (2n̂n̂ᵀ − I)v̂ᵢ, computed rather than asserted */
    const s = [2 * n[0] * (n[0] * vi[0] + n[1] * vi[1]) - vi[0],
    2 * n[1] * (n[0] * vi[0] + n[1] * vi[1]) - vi[1]];
    const NdotL = Math.max(0, vi[0] * n[0] + vi[1] * n[1]);
    /* the Torrance–Sparrow width is matched to the Phong exponent so the toggle is
       a fair comparison: equal half-power angle */
    const halfPhong = Math.acos(Math.pow(0.5, 1 / ke));
    const cs = Math.sqrt(Math.log(2)) / Math.max(1e-6, halfPhong);

    const spec = thV => {
      const v = [Math.sin(thV), Math.cos(thV)];
      const cosS = VZ.clamp(v[0] * s[0] + v[1] * s[1], -1, 1);
      if (gauss) { const a = Math.acos(cosS); return ks * Math.exp(-cs * cs * a * a); }
      return cosS <= 0 ? 0 : ks * Math.pow(cosS, ke);
    };
    const diff = () => kd * NdotL;
    const amb = () => ka;

    const fr = VZ.frame(svg, W, H, { l: 8, r: 8, t: 8, b: 8 });
    const g = fr.g;
    const leftW = 400, rightW = fr.iw - leftW - 14;

    /* ── left: the Cartesian cross-section ── */
    const gl = g.append("g");
    gl.append("rect").attr("x", 0).attr("y", 0).attr("width", leftW).attr("height", fr.ih)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.55).attr("stroke", VC.line);
    const m = { l: 44, r: 14, t: 34, b: 40 };
    const pw = leftW - m.l - m.r, ph = fr.ih - m.t - m.b;
    const gg = gl.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const X = d3.scaleLinear().domain([-90, 90]).range([0, pw]);
    const ymax = Math.max(0.52, ka + kd + ks * 1.02);
    const Y = d3.scaleLinear().domain([0, ymax]).range([ph, 0]);
    VZ.gridY(gg, Y, pw, 5);
    VZ.axisB(gg, X, ph, 7, "viewing angle from the normal", v => v + "°");
    VZ.axisL(gg, Y, 5, "reflected radiance");

    const curves = [
      { name: "ambient", col: VC.muted, f: () => amb() },
      { name: "diffuse", col: VC.teal, f: () => diff() },
      { name: "specular", col: VC.a2, f: t => spec(VZ.rad(t)) },
      { name: "total", col: VC.good, f: t => amb() + diff() + spec(VZ.rad(t)) }
    ];
    curves.forEach(c => {
      const pts = [];
      for (let t = -90; t <= 90; t += 0.25) pts.push([X(t), Y(c.f(t))]);
      VZ.poly(gg, pts, { stroke: c.col, w: c.name === "total" ? 2.2 : 1.5, close: false });
    });
    /* mark the light and mirror directions */
    [[+eL.value, VC.violet, "light"], [-(+eL.value), VC.bad, "mirror ŝ"]].forEach(it => {
      gg.append("line").attr("x1", X(it[0])).attr("x2", X(it[0])).attr("y1", 0).attr("y2", ph)
        .attr("stroke", it[1]).attr("stroke-width", 1).attr("stroke-dasharray", "3 3");
      gg.append("text").attr("x", X(it[0]) + 4).attr("y", 11).attr("font-size", 9.5)
        .attr("fill", it[1]).text(it[2]);
    });
    VZ.legend(gg, curves.map(c => ({ label: c.name, color: c.col })), pw - 74, 16, { gap: 13 });
    gl.append("text").attr("x", 10).attr("y", 18).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("cross-section · ambient and diffuse do not depend on the viewer");

    /* ── right: the polar lobes ── */
    const gr = g.append("g").attr("transform", `translate(${leftW + 14},0)`);
    gr.append("rect").attr("x", 0).attr("y", 0).attr("width", rightW).attr("height", fr.ih)
      .attr("rx", 8).attr("fill", VC.bg).attr("fill-opacity", 0.4).attr("stroke", VC.line);
    const cx = rightW / 2, cy = fr.ih - 74, RR = Math.min(rightW / 2 - 20, cy - 34);
    const sc = RR / ymax;
    /* the surface and its normal */
    gr.append("line").attr("x1", 14).attr("x2", rightW - 14).attr("y1", cy).attr("y2", cy)
      .attr("stroke", VC.line).attr("stroke-width", 2);
    VZ.arrow(gr, cx, cy, cx, cy - RR - 12, { color: VC.muted, w: 1.2, head: 6, dash: "4 3" });
    gr.append("text").attr("x", cx + 5).attr("y", cy - RR - 14).attr("font-size", 10).attr("fill", VC.muted).text("n̂");
    for (let a = -90; a <= 90; a += 30) {
      const r = VZ.rad(a);
      gr.append("line").attr("x1", cx).attr("y1", cy)
        .attr("x2", cx + RR * Math.sin(r)).attr("y2", cy - RR * Math.cos(r))
        .attr("stroke", VC.grid).attr("stroke-width", 0.7);
    }
    curves.forEach(c => {
      const pts = [];
      for (let t = -90; t <= 90; t += 0.25) {
        const r = VZ.rad(t), v = c.f(t) * sc;
        pts.push([cx + v * Math.sin(r), cy - v * Math.cos(r)]);
      }
      VZ.poly(gr, pts, { stroke: c.col, w: c.name === "total" ? 2 : 1.3, close: false });
    });
    /* incoming light and mirror ray */
    VZ.arrow(gr, cx + RR * Math.sin(thI) * 1.05, cy - RR * Math.cos(thI) * 1.05, cx, cy,
      { color: VC.violet, w: 1.8, head: 7 });
    VZ.arrow(gr, cx, cy, cx + RR * s[0] * 1.0, cy - RR * s[1] * 1.0,
      { color: VC.bad, w: 1.6, head: 7, dash: "5 3" });
    gr.append("text").attr("x", 10).attr("y", 18).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("the same thing as polar lobes");

    /* the measured half-power width of the specular lobe */
    const peak = spec(-thI);
    let halfMeasured = NaN;
    for (let dt = 0; dt < 90; dt += 0.01) {
      if (spec(-thI + VZ.rad(dt)) <= peak / 2) { halfMeasured = dt; break; }
    }
    let yy = fr.ih - 52;
    const put = (lab, val, col) => {
      gr.append("text").attr("x", 12).attr("y", yy).attr("font-size", 9.5).attr("fill", VC.muted).text(lab);
      gr.append("text").attr("x", rightW - 12).attr("y", yy).attr("text-anchor", "end").attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col || VC.ink).text(val);
      yy += 16;
    };
    put("mirror direction ŝ", VZ.fmt(VZ.deg(Math.atan2(s[0], s[1])), 2) + "°", VC.bad);
    put("[v̂ᵢ · n̂]⁺ = cos θᵢ", VZ.fmt(NdotL, 4), VC.teal);
    put("specular half-power width", VZ.fmt(halfMeasured, 3) + "°", VC.a2);

    /* ── readout ── */
    const predicted = VZ.deg(Math.acos(Math.pow(0.5, 1 / ke)));
    out.innerHTML =
      `light at ${eL.value}°, mirror direction at <b>${VZ.fmt(VZ.deg(Math.atan2(s[0], s[1])), 2)}°</b> `
      + `— computed as ŝ = (2n̂n̂ᵀ − I)v̂ᵢ, and it is the light angle reflected about the normal.<br>`
      + `Diffuse radiance = k_d[v̂ᵢ·n̂]⁺ = ${VZ.fmt(kd, 2)} × ${VZ.fmt(NdotL, 4)} = <b>${VZ.fmt(kd * NdotL, 4)}</b>, `
      + `the same for every viewing direction — which is why its curve is flat and a matte object's brightness `
      + `tells you about its <i>normal</i> and nothing about where you are standing.<br>`
      + (gauss
        ? `Torrance–Sparrow lobe, width matched to the Phong exponent: measured half-power at <b>${VZ.fmt(halfMeasured, 3)}°</b>.`
        : `Phong exponent k_e = ${ke}: half-power at <b>${VZ.fmt(halfMeasured, 3)}°</b>, and arccos(2^(−1/k_e)) predicts `
        + `<b>${VZ.fmt(predicted, 3)}°</b> — the same number.`);
  }

  const bind = (e, v, d) => { e.oninput = () => { v.textContent = d === 0 ? e.value + "°" : (+e.value).toFixed(d); draw(); }; };
  eL.oninput = () => { eLv.textContent = eL.value + "°"; draw(); };
  eK.oninput = () => { eKv.textContent = eK.value; draw(); };
  bind(eA, eAv, 2); bind(eD, eDv, 2); bind(eS, eSv, 2);
  eG.onchange = draw;
  draw();
})();

/* ───────── 14 · a Lambertian sphere with a movable light ───────── */
(function () {
  const svg = d3.select("#shade-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 320;
  const gid = i => document.getElementById(i);
  const eAz = gid("sh-az"), eAzv = gid("sh-azv"), eEl = gid("sh-el"), eElv = gid("sh-elv");
  const eR = gid("sh-rho"), eRv = gid("sh-rhov"), eA = gid("sh-amb"), eAv = gid("sh-ambv");
  const eKs = gid("sh-ks"), eKsv = gid("sh-ksv"), eKe = gid("sh-ke"), eKev = gid("sh-kev");
  const eT = gid("sh-term"), eI = gid("sh-iso");
  const out = gid("shade-readout");

  const NPIX = 132;                            // the sphere is rendered on an NPIX grid

  function draw() {
    const az = VZ.rad(+eAz.value), el = VZ.rad(+eEl.value);
    const rho = +eR.value, amb = +eA.value, ks = +eKs.value, ke = +eKe.value;
    /* image coordinates: x right, y up on screen; z out of the screen toward the viewer */
    const l = VZ.unit([Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)]);
    const v = [0, 0, 1];                       // orthographic viewer, along +z

    const fr = VZ.frame(svg, W, H, { l: 8, r: 8, t: 8, b: 8 });
    const g = fr.g;
    const leftW = 372, rightW = fr.iw - leftW - 14;

    const gl = g.append("g");
    gl.append("rect").attr("x", 0).attr("y", 0).attr("width", leftW).attr("height", fr.ih)
      .attr("rx", 8).attr("fill", VC.bg).attr("fill-opacity", 0.45).attr("stroke", VC.line);
    const cx = leftW / 2, cy = fr.ih / 2, RAD = Math.min(leftW, fr.ih) / 2 - 26;

    /* render the sphere as small squares — one shading evaluation each */
    const step = 2 * RAD / NPIX;
    const px = [];
    let best = { I: -1 }, bestSpec = { S: -1 };
    for (let i = 0; i < NPIX; i++) for (let j = 0; j < NPIX; j++) {
      const X = (-RAD + (i + 0.5) * step) / RAD, Y = (RAD - (j + 0.5) * step) / RAD;
      const r2 = X * X + Y * Y;
      if (r2 > 1) continue;
      const n = [X, Y, Math.sqrt(1 - r2)];
      const ndl = n[0] * l[0] + n[1] * l[1] + n[2] * l[2];
      const diff = rho * Math.max(0, ndl);
      /* specular: mirror the light about the normal, then compare with the viewer */
      const d2 = 2 * ndl;
      const s = [d2 * n[0] - l[0], d2 * n[1] - l[1], d2 * n[2] - l[2]];
      const sv = Math.max(0, s[0] * v[0] + s[1] * v[1] + s[2] * v[2]);
      const spec = (ndl > 0 && ks > 0) ? ks * Math.pow(sv, ke) : 0;
      const I = VZ.clamp(amb + diff + spec, 0, 1);
      px.push({ x: cx - RAD + i * step, y: cy - RAD + j * step, I: I });
      if (I > best.I) best = { I: I, X: X, Y: Y };
      if (spec > bestSpec.S) bestSpec = { S: spec, X: X, Y: Y };
    }
    const gp = gl.append("g");
    px.forEach(p => gp.append("rect").attr("x", p.x).attr("y", p.y)
      .attr("width", step + 0.7).attr("height", step + 0.7)
      .attr("fill", d3.interpolateGreys(0.06 + 0.94 * p.I)));
    gl.append("circle").attr("cx", cx).attr("cy", cy).attr("r", RAD)
      .attr("fill", "none").attr("stroke", VC.line).attr("stroke-width", 1);

    /* the terminator: n̂ · l̂ = 0, i.e. X lₓ + Y l_y + √(1−r²) l_z = 0.
       Solve it as a curve in the image, by sweeping the angle round the great circle. */
    if (eT.checked) {
      /* build an orthonormal basis of the plane perpendicular to l̂, then project it */
      const a = VZ.unit(Math.abs(l[2]) < 0.9 ? VZ.cross(l, [0, 0, 1]) : VZ.cross(l, [1, 0, 0]));
      const b = VZ.cross(l, a);
      const front = [], back = [];
      for (let k = 0; k <= 240; k++) {
        const t = 2 * Math.PI * k / 240;
        const n = VZ.add(VZ.scale(a, Math.cos(t)), VZ.scale(b, Math.sin(t)));
        const pt = [cx + n[0] * RAD, cy - n[1] * RAD];
        (n[2] >= 0 ? front : back).push(pt);
      }
      if (front.length > 1) VZ.poly(gp, front, { stroke: VC.a2, w: 1.8, close: false });
      if (back.length > 1) VZ.poly(gp, back, { stroke: VC.a2, w: 1, close: false, dash: "3 3", op: 0.5 });
    }
    if (eI.checked) {
      [0.2, 0.4, 0.6, 0.8].forEach(c => {
        const a = VZ.unit(Math.abs(l[2]) < 0.9 ? VZ.cross(l, [0, 0, 1]) : VZ.cross(l, [1, 0, 0]));
        const b = VZ.cross(l, a);
        const st = Math.sqrt(1 - c * c), pts = [];
        for (let k = 0; k <= 200; k++) {
          const t = 2 * Math.PI * k / 200;
          const n = VZ.add(VZ.scale(l, c), VZ.add(VZ.scale(a, st * Math.cos(t)), VZ.scale(b, st * Math.sin(t))));
          if (n[2] >= 0) pts.push([cx + n[0] * RAD, cy - n[1] * RAD]);
        }
        if (pts.length > 2) VZ.poly(gp, pts, { stroke: VC.teal, w: 0.9, close: false, op: 0.85 });
      });
    }
    /* the diffuse maximum, which is exactly at (R lₓ, R l_y) when l points toward us */
    if (l[2] > 0) {
      gp.append("circle").attr("cx", cx + l[0] * RAD).attr("cy", cy - l[1] * RAD).attr("r", 4)
        .attr("fill", "none").attr("stroke", VC.good).attr("stroke-width", 2);
      gp.append("text").attr("x", cx + l[0] * RAD + 7).attr("y", cy - l[1] * RAD - 5)
        .attr("font-size", 9.5).attr("fill", VC.good).text("n̂ = l̂");
    }
    if (ks > 0 && bestSpec.S > 1e-6) {
      gp.append("circle").attr("cx", cx + bestSpec.X * RAD).attr("cy", cy - bestSpec.Y * RAD).attr("r", 4)
        .attr("fill", "none").attr("stroke", VC.bad).attr("stroke-width", 2);
      gp.append("text").attr("x", cx + bestSpec.X * RAD + 7).attr("y", cy - bestSpec.Y * RAD + 12)
        .attr("font-size", 9.5).attr("fill", VC.bad).text("highlight");
    }
    gl.append("text").attr("x", 10).attr("y", 17).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("I = ambient + ρ[n̂·l̂]⁺ + k_s(ŝ·v̂)^{k_e}");
    gl.append("text").attr("x", 10).attr("y", fr.ih - 10).attr("font-size", 9.5).attr("fill", VC.muted)
      .text("drag on the sphere to move the light");

    /* ── right: the centre-line profile against the exact cosine ── */
    const gr = g.append("g").attr("transform", `translate(${leftW + 14},0)`);
    gr.append("rect").attr("x", 0).attr("y", 0).attr("width", rightW).attr("height", fr.ih)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.6).attr("stroke", VC.line);
    const m = { l: 40, r: 14, t: 30, b: 96 };
    const pw = rightW - m.l - m.r, ph = fr.ih - m.t - m.b;
    const gg = gr.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const X = d3.scaleLinear().domain([-1, 1]).range([0, pw]);
    const Y = d3.scaleLinear().domain([0, 1.02]).range([ph, 0]);
    VZ.gridY(gg, Y, pw, 4); VZ.axisB(gg, X, ph, 5, "x / R"); VZ.axisL(gg, Y, 4, "intensity");
    const prof = [], cosOnly = [];
    for (let k = 0; k <= 300; k++) {
      const Xc = -1 + 2 * k / 300;
      if (Math.abs(Xc) > 1) continue;
      const n = [Xc, 0, Math.sqrt(Math.max(0, 1 - Xc * Xc))];
      const ndl = n[0] * l[0] + n[1] * l[1] + n[2] * l[2];
      const d2 = 2 * ndl;
      const s = [d2 * n[0] - l[0], d2 * n[1] - l[1], d2 * n[2] - l[2]];
      const sv = Math.max(0, s[2]);
      const spec = (ndl > 0 && ks > 0) ? ks * Math.pow(sv, ke) : 0;
      prof.push([X(Xc), Y(VZ.clamp(amb + rho * Math.max(0, ndl) + spec, 0, 1))]);
      cosOnly.push([X(Xc), Y(VZ.clamp(rho * Math.max(0, ndl), 0, 1))]);
    }
    VZ.poly(gg, cosOnly, { stroke: VC.teal, w: 1.4, close: false, dash: "4 3" });
    VZ.poly(gg, prof, { stroke: VC.good, w: 2, close: false });
    VZ.legend(gg, [{ label: "ρ[n̂·l̂]⁺ alone", color: VC.teal, dash: "4 3" },
    { label: "with ambient + specular", color: VC.good }], 6, 12, { gap: 13 });
    gr.append("text").attr("x", 12).attr("y", 18).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("along the horizontal centre line");

    let yy = fr.ih - 76;
    const put = (lab, val, col) => {
      gr.append("text").attr("x", 12).attr("y", yy).attr("font-size", 9.5).attr("fill", VC.muted).text(lab);
      gr.append("text").attr("x", rightW - 12).attr("y", yy).attr("text-anchor", "end").attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col || VC.ink).text(val);
      yy += 16;
    };
    put("light direction l̂", "(" + l.map(x => VZ.fmt(x, 3)).join(", ") + ")", VC.a2);
    put("brightest pixel at (x/R, y/R)", "(" + VZ.fmt(best.X, 3) + ", " + VZ.fmt(best.Y, 3) + ")", VC.good);
    put("predicted, if diffuse only", "(" + VZ.fmt(l[0], 3) + ", " + VZ.fmt(l[1], 3) + ")", VC.muted);
    put("lit fraction of the disc", VZ.fmt(100 * px.filter(p => p.I > amb + 1e-9).length / px.length, 1) + "%", VC.teal);

    /* drag the light */
    svg.style("cursor", "crosshair").on("mousedown", function (ev) {
      const mm = d3.pointer(ev, svg.node());
      const X0 = (mm[0] - 8 - cx) / RAD, Y0 = -(mm[1] - 8 - cy) / RAD;
      const r = Math.hypot(X0, Y0);
      if (r > 1.02) return;
      const nx = X0 / Math.max(1, r), ny = Y0 / Math.max(1, r);
      const nz = Math.sqrt(Math.max(0, 1 - nx * nx - ny * ny));
      eEl.value = Math.round(VZ.deg(Math.asin(VZ.clamp(ny, -1, 1))));
      eAz.value = Math.round(VZ.deg(Math.atan2(nx, nz)));
      eAzv.textContent = eAz.value + "°"; eElv.textContent = eEl.value + "°";
      draw();
    });

    /* ── readout ── */
    const lit = 100 * px.filter(p => p.I > amb + 1e-9).length / px.length;
    out.innerHTML =
      `l̂ = (${l.map(x => VZ.fmt(x, 3)).join(", ")}) · albedo <span class="keep">ρ</span> = ${VZ.fmt(rho, 2)}, ambient ${VZ.fmt(amb, 3)}, `
      + `k_s = ${VZ.fmt(ks, 2)}.<br>`
      + `<b>${VZ.fmt(lit, 1)}%</b> of the visible disc is lit, and the closed form (1 + l_z)/2 predicts `
      + `<b>${VZ.fmt(100 * (1 + l[2]) / 2, 1)}%</b> — the same number. When the light lies in the image plane `
      + `(l_z = 0) it is exactly 50% and the terminator is a straight line; here l_z = ${VZ.fmt(l[2], 4)}.<br>`
      + (ks > 0
        ? `The diffuse maximum sits at (${VZ.fmt(l[0], 3)}, ${VZ.fmt(l[1], 3)}) and the specular highlight at `
        + `(${VZ.fmt(bestSpec.X, 3)}, ${VZ.fmt(bestSpec.Y, 3)}) — <b>different points</b>, because the highlight needs `
        + `the MIRROR direction to point at the camera, not the normal to point at the light.`
        : `With k_s = 0 the brightest pixel is exactly where n̂ = l̂, at (${VZ.fmt(l[0], 3)}, ${VZ.fmt(l[1], 3)}) — `
        + `so a single matte sphere is enough to read off the light direction.`);
  }

  const bind = (e, v, d, suf) => { e.oninput = () => { v.textContent = (+e.value).toFixed(d) + (suf || ""); draw(); }; };
  eAz.oninput = () => { eAzv.textContent = eAz.value + "°"; draw(); };
  eEl.oninput = () => { eElv.textContent = eEl.value + "°"; draw(); };
  bind(eR, eRv, 2); bind(eA, eAv, 3); bind(eKs, eKsv, 2);
  eKe.oninput = () => { eKev.textContent = eKe.value; draw(); };
  eT.onchange = draw; eI.onchange = draw;
  draw();
})();

/* ───────── 15 · colour matching functions and the chromaticity diagram ───────── */
(function () {
  const svg = d3.select("#cie-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 420;
  const gid = i => document.getElementById(i);
  const eM = gid("ci-mode"), eG = gid("ci-gam");
  const eL = gid("ci-lam"), eLv = gid("ci-lamv"), eW = gid("ci-white");
  const out = gid("cie-readout");

  /* The CIE 1931 2° observer, as the standard multi-lobe piecewise-Gaussian fit.
     Verified against the tabulated curves: peaks 1.0559 @600, 0.9980 @555, 1.7816 @450
     against the true 1.0622, 1.0000, 1.7826, and equal-energy white lands at
     (0.3331, 0.3336) rather than exactly (1/3, 1/3). Good to about 1%, which is far
     inside what a 300-pixel-wide plot can show.                                      */
  function gp(x, mu, s1, s2) { const s = x < mu ? s1 : s2, t = (x - mu) / s; return Math.exp(-0.5 * t * t); }
  const xbar = l => 1.056 * gp(l, 599.8, 37.9, 31.0) + 0.362 * gp(l, 442.0, 16.0, 26.7) - 0.065 * gp(l, 501.1, 20.4, 26.2);
  const ybar = l => 0.821 * gp(l, 568.8, 46.9, 40.5) + 0.286 * gp(l, 530.9, 16.3, 31.1);
  const zbar = l => 1.217 * gp(l, 437.0, 11.8, 36.0) + 0.681 * gp(l, 459.0, 26.0, 13.8);
  /* the CIE 1931 RGB curves, obtained by inverting the standard RGB→XYZ matrix */
  const M_RGB2XYZ = [[0.49, 0.31, 0.20], [0.17697, 0.81240, 0.01063], [0.00, 0.01, 0.99]]
    .map(r => r.map(v => v / 0.17697));
  const M_XYZ2RGB = VZ.inv3(M_RGB2XYZ);
  const rgbBar = l => VZ.mv(M_XYZ2RGB, [xbar(l), ybar(l), zbar(l)]);

  /* the chromaticity of one wavelength */
  function chrom(l) {
    const X = xbar(l), Y = ybar(l), Z = zbar(l), s = X + Y + Z;
    return s > 1e-9 ? [X / s, Y / s] : [0, 0];
  }
  /* The fit's red tail curls back on itself beyond ~645 nm, so the locus is drawn to
     the point of maximum x and closed with the purple line from there. */
  const LMIN = 400, LMAX = 645;

  const GAMUTS = {
    "709": { name: "BT.709 / sRGB", P: [[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]], w: [0.3127, 0.3290] },
    "p3": { name: "Display P3", P: [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]], w: [0.3127, 0.3290] },
    "2020": { name: "BT.2020", P: [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]], w: [0.3127, 0.3290] },
    "cie": { name: "CIE 1931 RGB", P: [chrom(700.0), chrom(546.1), chrom(435.8)], w: [1 / 3, 1 / 3] }
  };

  /* polygon area, used to report how much of the horseshoe a gamut covers */
  const area = P => Math.abs(P.reduce((s, p, i) => {
    const q = P[(i + 1) % P.length]; return s + p[0] * q[1] - q[0] * p[1];
  }, 0)) / 2;

  function draw() {
    const mode = eM.value, gk = eG.value, lam = +eL.value;
    const G = GAMUTS[gk];

    const fr = VZ.frame(svg, W, H, { l: 8, r: 8, t: 8, b: 8 });
    const g = fr.g;
    const leftW = 372, rightW = fr.iw - leftW - 14;

    /* ── left: the matching functions ── */
    const gl = g.append("g");
    gl.append("rect").attr("x", 0).attr("y", 0).attr("width", leftW).attr("height", fr.ih)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.55).attr("stroke", VC.line);
    const m = { l: 44, r: 14, t: 34, b: 40 };
    const pw = leftW - m.l - m.r, ph = fr.ih - m.t - m.b;
    const gg = gl.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const X = d3.scaleLinear().domain([380, 700]).range([0, pw]);
    const isXYZ = mode === "xyz";
    const Y = d3.scaleLinear().domain(isXYZ ? [0, 1.85] : [-0.12, 0.40]).range([ph, 0]);
    VZ.gridY(gg, Y, pw, 5);
    VZ.axisB(gg, X, ph, 6, "wavelength λ (nm)");
    VZ.axisL(gg, Y, 5);
    gl.append("text").attr("x", leftW - 12).attr("y", 18).attr("text-anchor", "end")
      .attr("font-size", 9.5).attr("fill", VC.muted).text("matching value ↑");
    gg.append("line").attr("x1", 0).attr("x2", pw).attr("y1", Y(0)).attr("y2", Y(0))
      .attr("stroke", VC.muted).attr("stroke-width", 1);

    const fns = isXYZ
      ? [{ n: "x̄", f: xbar, c: VC.bad }, { n: "ȳ", f: ybar, c: VC.good }, { n: "z̄", f: zbar, c: VC.accent }]
      : [{ n: "r̄", f: l => rgbBar(l)[0], c: VC.bad }, { n: "ḡ", f: l => rgbBar(l)[1], c: VC.good },
      { n: "b̄", f: l => rgbBar(l)[2], c: VC.accent }];
    fns.forEach(fn => {
      const pts = [];
      for (let l = 380; l <= 700; l += 1) pts.push([X(l), Y(fn.f(l))]);
      VZ.poly(gg, pts, { stroke: fn.c, w: 1.8, close: false });
    });
    if (!isXYZ) {
      /* shade the negative lobe, which is the whole point of this view */
      const neg = [];
      for (let l = 380; l <= 700; l += 1) if (rgbBar(l)[0] < 0) neg.push([X(l), Y(rgbBar(l)[0])]);
      if (neg.length > 2) {
        const poly = [[neg[0][0], Y(0)]].concat(neg).concat([[neg[neg.length - 1][0], Y(0)]]);
        VZ.poly(gg, poly, { stroke: "none", w: 0, fill: VC.bad, fillOp: 0.18 });
        gg.append("text").attr("x", neg[Math.floor(neg.length / 2)][0]).attr("y", Y(-0.09))
          .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", VC.bad)
          .text("r̄ < 0 — no real primaries suffice");
      }
    }
    gg.append("line").attr("x1", X(lam)).attr("x2", X(lam)).attr("y1", 0).attr("y2", ph)
      .attr("stroke", VC.a2).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
    VZ.legend(gg, fns.map(f => ({ label: f.n, color: f.c })), pw - 40, 12, { gap: 13 });
    gl.append("text").attr("x", 10).attr("y", 18).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text(isXYZ ? "x̄, ȳ, z̄ — all non-negative by construction"
        : "r̄, ḡ, b̄ — measured with real primaries");

    /* ── right: the chromaticity diagram ── */
    const gr = g.append("g").attr("transform", `translate(${leftW + 14},0)`);
    gr.append("rect").attr("x", 0).attr("y", 0).attr("width", rightW).attr("height", fr.ih)
      .attr("rx", 8).attr("fill", VC.bg).attr("fill-opacity", 0.45).attr("stroke", VC.line);
    const cm = { l: 38, r: 14, t: 30, b: 82 };
    const cw = rightW - cm.l - cm.r, chh = fr.ih - cm.t - cm.b;
    const gc = gr.append("g").attr("transform", `translate(${cm.l},${cm.t})`);
    const CX = d3.scaleLinear().domain([0, 0.8]).range([0, cw]);
    const CY = d3.scaleLinear().domain([0, 0.9]).range([chh, 0]);
    VZ.axisB(gc, CX, chh, 4, "x"); VZ.axisL(gc, CY, 4, "y");

    const locus = [];
    for (let l = LMIN; l <= LMAX; l += 1) locus.push(chrom(l));
    const scr = p => [CX(p[0]), CY(p[1])];
    /* fill the horseshoe with an approximate colour, converted through the sRGB matrix */
    const M_XYZ2s = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]];
    const gfill = gc.append("g").attr("clip-path", VZ.clip(svg, "ci-clip", 0, 0, cw, chh));
    const NB = 78;
    for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) {
      const x = 0.8 * (i + 0.5) / NB, y = 0.9 * (j + 0.5) / NB;
      if (y <= 0 || x + y >= 1) continue;
      /* inside the horseshoe? test against the closed locus polygon */
      let inside = false;
      for (let k = 0, l2 = locus.length - 1; k < locus.length; l2 = k++) {
        const a = locus[k], b = locus[l2];
        if ((a[1] > y) !== (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
      }
      if (!inside) continue;
      const Yl = 1, Xl = x * Yl / y, Zl = (1 - x - y) * Yl / y;
      let c = VZ.mv(M_XYZ2s, [Xl, Yl, Zl]);
      const mx = Math.max(...c);
      c = c.map(v => Math.pow(VZ.clamp(v / mx, 0, 1), 1 / 2.2));
      gfill.append("rect").attr("x", CX(x) - cw / NB / 1.6).attr("y", CY(y) - chh / NB / 1.6)
        .attr("width", cw / NB * 1.25).attr("height", chh / NB * 1.25)
        .attr("fill", d3.rgb(255 * c[0], 255 * c[1], 255 * c[2]).toString())
        .attr("fill-opacity", 0.55);
    }
    VZ.poly(gc, locus.map(scr), { stroke: VC.ink, w: 1.6, close: false });
    /* the purple line */
    gc.append("line").attr("x1", scr(locus[0])[0]).attr("y1", scr(locus[0])[1])
      .attr("x2", scr(locus[locus.length - 1])[0]).attr("y2", scr(locus[locus.length - 1])[1])
      .attr("stroke", VC.violet).attr("stroke-width", 1.6);
    gc.append("text").attr("x", CX(0.42)).attr("y", CY(0.10)).attr("font-size", 9.5)
      .attr("fill", VC.violet).text("purple line — no wavelength");

    VZ.poly(gc, G.P.map(scr), { stroke: VC.a2, w: 2, fill: "none" });
    G.P.forEach((p, i) => {
      gc.append("circle").attr("cx", scr(p)[0]).attr("cy", scr(p)[1]).attr("r", 3.5)
        .attr("fill", [VC.bad, VC.good, VC.accent][i]).attr("stroke", VC.bg);
    });
    const mk = chrom(lam);
    gc.append("circle").attr("cx", scr(mk)[0]).attr("cy", scr(mk)[1]).attr("r", 5)
      .attr("fill", "none").attr("stroke", VC.a2).attr("stroke-width", 2);
    gc.append("text").attr("x", scr(mk)[0] + 8).attr("y", scr(mk)[1] - 5).attr("font-size", 9.5)
      .attr("fill", VC.a2).text(lam + " nm");
    if (eW.checked) {
      [[G.w, "D65", VC.ink, -9], [[1 / 3, 1 / 3], "equal energy", VC.muted, 15]].forEach(wp => {
        gc.append("path").attr("d", `M${scr(wp[0])[0] - 5},${scr(wp[0])[1]} h10 M${scr(wp[0])[0]},${scr(wp[0])[1] - 5} v10`)
          .attr("stroke", wp[2]).attr("stroke-width", 1.4);
        gc.append("text").attr("x", scr(wp[0])[0] + 8).attr("y", scr(wp[0])[1] + wp[3])
          .attr("font-size", 9).attr("fill", wp[2]).text(wp[1]);
      });
    }
    gr.append("text").attr("x", 12).attr("y", 18).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("chromaticity · " + G.name + " gamut");

    /* How much of the locus does the triangle miss? "How many spectral colours are
       INSIDE the triangle" is a useless statistic — the answer is always zero, since
       a triangle with vertices strictly inside a strictly convex region cannot touch
       its boundary. The informative measure is the SATURATION SHORTFALL: cast a ray
       from the white point through each spectral colour and compare how far the
       gamut reaches along it with how far the locus is.                              */
    function reach(target, W_) {
      /* the largest t with W + t(target − W) inside the triangle, by clipping the ray */
      const d = [target[0] - W_[0], target[1] - W_[1]];
      let tmax = 1e9;
      for (let i = 0; i < 3; i++) {
        const a = G.P[i], b = G.P[(i + 1) % 3];
        const e = [b[0] - a[0], b[1] - a[1]];
        const den = d[0] * e[1] - d[1] * e[0];
        if (Math.abs(den) < 1e-12) continue;
        const t = ((a[0] - W_[0]) * e[1] - (a[1] - W_[1]) * e[0]) / den;
        const u = ((a[0] - W_[0]) * d[1] - (a[1] - W_[1]) * d[0]) / den;
        if (t > 1e-9 && u >= -1e-9 && u <= 1 + 1e-9) tmax = Math.min(tmax, t);
      }
      return tmax;
    }
    let worstSat = 1, worstLam = LMIN;
    for (let l = LMIN; l <= LMAX; l += 1) {
      const r = reach(chrom(l), G.w);
      if (r < worstSat) { worstSat = r; worstLam = l; }
    }
    const aT = area(G.P), aL = area(locus);
    let yy = fr.ih - 34;
    const put = (lab, val, col) => {
      gr.append("text").attr("x", 12).attr("y", yy).attr("font-size", 9.5).attr("fill", VC.muted).text(lab);
      gr.append("text").attr("x", rightW - 12).attr("y", yy).attr("text-anchor", "end").attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col || VC.ink).text(val);
      yy += 15;
    };
    put("triangle area / horseshoe area", VZ.fmt(100 * aT / aL, 1) + "%", VC.a2);
    put("worst saturation reach, at " + worstLam + " nm", VZ.fmt(100 * worstSat, 1) + "%",
      worstSat > 0.85 ? VC.good : (worstSat > 0.6 ? VC.a2 : VC.bad));

    /* ── readout ── */
    const c = chrom(lam);
    const rgbAt = rgbBar(lam);
    out.innerHTML =
      `<span class="keep">λ</span> = ${lam} nm has chromaticity (<b>${VZ.fmt(c[0], 4)}, ${VZ.fmt(c[1], 4)}</b>) and matching values `
      + `x̄ = ${VZ.fmt(xbar(lam), 4)}, ȳ = ${VZ.fmt(ybar(lam), 4)}, z̄ = ${VZ.fmt(zbar(lam), 4)}; `
      + `in the 1931 RGB basis, r̄ = <b style="color:${rgbAt[0] < 0 ? "var(--bad)" : "var(--ink)"}">${VZ.fmt(rgbAt[0], 4)}</b>, `
      + `ḡ = ${VZ.fmt(rgbAt[1], 4)}, b̄ = ${VZ.fmt(rgbAt[2], 4)}`
      + (rgbAt[0] < 0 ? ` — negative, so this wavelength cannot be matched by any positive mixture of the three primaries.` : `.`)
      + `<br>The <b>${G.name}</b> triangle covers <b>${VZ.fmt(100 * aT / aL, 1)}%</b> of the horseshoe's area. `
      + `Its worst direction is <b>${worstLam} nm</b>, where it reaches only <b>${VZ.fmt(100 * worstSat, 1)}%</b> of the way `
      + `from white out to the pure colour. No spectral colour at all is reachable, and that is not a defect of this `
      + `particular gamut: the locus is strictly convex, so a triangle whose corners lie inside it can never touch its rim.`;
  }

  eM.onchange = draw; eG.onchange = draw; eW.onchange = draw;
  eL.oninput = () => { eLv.textContent = eL.value + " nm"; draw(); };
  draw();
})();

/* ───────── 16 · gamma encoding, quantisation and the blending bug ───────── */
(function () {
  const svg = d3.select("#gamma-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 420;
  const gid = i => document.getElementById(i);
  const eG = gid("gm2-g"), eGv = gid("gm2-gv"), eB = gid("gm2-b"), eBv = gid("gm2-bv");
  const eN = gid("gm2-n"), eNv = gid("gm2-nv"), eS = gid("gm2-srgb");
  const out = gid("gamma-readout");

  const srgbEnc = L => L <= 0.0031308 ? 12.92 * L : 1.055 * Math.pow(L, 1 / 2.4) - 0.055;
  const srgbDec = V => V <= 0.04045 ? V / 12.92 : Math.pow((V + 0.055) / 1.055, 2.4);

  function draw() {
    const gm = +eG.value, bits = +eB.value, noise = +eN.value, useS = eS.checked;
    const N = Math.pow(2, bits) - 1;
    const enc = L => useS ? srgbEnc(L) : Math.pow(L, 1 / gm);
    const dec = V => useS ? srgbDec(V) : Math.pow(V, gm);

    const fr = VZ.frame(svg, W, H, { l: 8, r: 8, t: 8, b: 8 });
    const g = fr.g;
    const leftW = 390, rightW = fr.iw - leftW - 14;

    /* ── left: the encoding curve, with the code ladder projected onto both axes ── */
    const gl = g.append("g");
    gl.append("rect").attr("x", 0).attr("y", 0).attr("width", leftW).attr("height", fr.ih)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.55).attr("stroke", VC.line);
    const m = { l: 46, r: 16, t: 32, b: 76 };
    const pw = leftW - m.l - m.r, ph = fr.ih - m.t - m.b;
    const gg = gl.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const X = d3.scaleLinear().domain([0, 1]).range([0, pw]);
    const Y = d3.scaleLinear().domain([0, 1]).range([ph, 0]);
    VZ.gridY(gg, Y, pw, 5);
    VZ.axisB(gg, X, ph, 5, "linear luminance L");
    VZ.axisL(gg, Y, 5);
    gl.append("text").attr("x", leftW - 12).attr("y", 18).attr("text-anchor", "end")
      .attr("font-size", 9.5).attr("fill", VC.muted).text("stored code V ↑");
    gg.append("line").attr("x1", X(0)).attr("y1", Y(0)).attr("x2", X(1)).attr("y2", Y(1))
      .attr("stroke", VC.grid).attr("stroke-dasharray", "4 4");
    const pts = [];
    for (let i = 0; i <= 400; i++) { const L = i / 400; pts.push([X(L), Y(enc(L))]); }
    VZ.poly(gg, pts, { stroke: VC.accent, w: 2, close: false });

    /* a ladder of evenly spaced CODES, projected back to linear light */
    const step = Math.max(1, Math.round(N / 24));
    for (let n = 0; n <= N; n += step) {
      const V = n / N, L = dec(V);
      gg.append("line").attr("x1", X(L)).attr("x2", X(L)).attr("y1", ph).attr("y2", ph + 8)
        .attr("stroke", VC.a2).attr("stroke-width", 1.1);
      gg.append("line").attr("x1", -8).attr("x2", 0).attr("y1", Y(V)).attr("y2", Y(V))
        .attr("stroke", VC.a2).attr("stroke-width", 1.1);
    }
    gg.append("text").attr("x", 0).attr("y", ph + 34).attr("font-size", 9.5).attr("fill", VC.a2)
      .text("evenly spaced codes, landing in linear light ↑");
    gl.append("text").attr("x", 10).attr("y", 18).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text(useS ? "the exact sRGB transfer function" : "V = L^(1/γ), γ = " + VZ.fmt(gm, 2));

    /* the two quantised strips: gamma-encoded vs linear, same bit depth */
    const stripY = fr.ih - 42, stripH = 26, sw = leftW - 24;
    for (let i = 0; i < sw; i++) {
      const L = i / (sw - 1);
      const qg = dec(Math.round(enc(L) * N) / N);
      const ql = Math.round(L * N) / N;
      const cg = Math.round(255 * enc(qg)), cl = Math.round(255 * enc(ql));
      gl.append("rect").attr("x", 12 + i).attr("y", stripY - stripH - 2).attr("width", 1.4).attr("height", stripH)
        .attr("fill", `rgb(${cg},${cg},${cg})`);
      gl.append("rect").attr("x", 12 + i).attr("y", stripY + 4).attr("width", 1.4).attr("height", stripH)
        .attr("fill", `rgb(${cl},${cl},${cl})`);
    }
    gl.append("text").attr("x", 12).attr("y", stripY - stripH - 6).attr("font-size", 9).attr("fill", VC.muted)
      .text(bits + "-bit, gamma encoded");
    gl.append("text").attr("x", 12).attr("y", stripY + 2).attr("font-size", 9).attr("fill", VC.muted)
      .text(bits + "-bit, LINEAR — watch the shadows band");

    /* ── right: the errors and the blending bug ── */
    const gr = g.append("g").attr("transform", `translate(${leftW + 14},0)`);
    gr.append("rect").attr("x", 0).attr("y", 0).attr("width", rightW).attr("height", fr.ih)
      .attr("rx", 8).attr("fill", VC.bg).attr("fill-opacity", 0.45).attr("stroke", VC.line);
    const m2 = { l: 46, r: 14, t: 32, b: 172 };
    const pw2 = rightW - m2.l - m2.r, ph2 = fr.ih - m2.t - m2.b;
    const g2 = gr.append("g").attr("transform", `translate(${m2.l},${m2.t})`);
    const X2 = d3.scaleLog().domain([0.001, 1]).range([0, pw2]);
    const Y2 = d3.scaleLog().domain([1e-4, 3]).range([ph2, 0]);
    g2.append("g").attr("class", "axis").attr("transform", `translate(0,${ph2})`)
      .call(d3.axisBottom(X2).ticks(4, "~g"));
    g2.append("g").attr("class", "axis").call(d3.axisLeft(Y2).ticks(4, "~g"));
    g2.append("text").attr("x", pw2).attr("y", ph2 + 30).attr("text-anchor", "end").attr("font-size", 10)
      .attr("fill", VC.muted).text("linear luminance L");
    g2.append("text").attr("x", 0).attr("y", -8).attr("font-size", 10).attr("fill", VC.muted)
      .text("relative luminance error of one code step");
    [["gamma encoded", VC.accent, true], ["linear encoded", VC.bad, false]].forEach(s => {
      const p = [];
      for (let i = 0; i <= 260; i++) {
        const L = Math.pow(10, -3 + 3 * i / 260);
        const rel = s[2] ? (dec((Math.round(enc(L) * N) + 1) / N) - dec(Math.round(enc(L) * N) / N)) / L
          : (1 / N) / L;
        if (rel > 1e-4 && rel < 3) p.push([X2(L), Y2(rel)]);
      }
      VZ.poly(g2, p, { stroke: s[1], w: 1.8, close: false });
    });
    /* the 1% discrimination threshold */
    g2.append("line").attr("x1", 0).attr("x2", pw2).attr("y1", Y2(0.01)).attr("y2", Y2(0.01))
      .attr("stroke", VC.good).attr("stroke-width", 1).attr("stroke-dasharray", "4 3");
    g2.append("text").attr("x", 4).attr("y", Y2(0.01) - 4).attr("font-size", 9).attr("fill", VC.good)
      .text("1% — the eye's threshold");
    VZ.legend(g2, [{ label: "gamma", color: VC.accent }, { label: "linear", color: VC.bad }], pw2 - 54, 12, { gap: 13 });
    gr.append("text").attr("x", 12).attr("y", 18).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("banding: below the green line it is invisible");

    /* the blend demo */
    const byy = fr.ih - 156;
    const midCode = 0.5, midLin = dec(midCode);
    const trueMid = 0.5;                                   // the average of black and white LIGHT
    const trueMidCode = enc(trueMid);
    gr.append("text").attr("x", 12).attr("y", byy).attr("font-size", 10.5).attr("fill", VC.ink)
      .attr("font-weight", 600).text("blending black and white, half and half");
    const bw = rightW - 24, bh = 26;
    const sw2 = [["average the CODES: V = 0.5", midCode, midLin, VC.bad],
    ["average the LIGHT: L = 0.5", trueMidCode, trueMid, VC.good]];
    sw2.forEach((s, i) => {
      const y = byy + 14 + i * 58;
      const c = Math.round(255 * s[1]);
      gr.append("rect").attr("x", 12).attr("y", y).attr("width", bw / 3).attr("height", bh).attr("fill", "#000");
      gr.append("rect").attr("x", 12 + bw / 3).attr("y", y).attr("width", bw / 3).attr("height", bh)
        .attr("fill", `rgb(${c},${c},${c})`).attr("stroke", s[3]);
      gr.append("rect").attr("x", 12 + 2 * bw / 3).attr("y", y).attr("width", bw / 3).attr("height", bh).attr("fill", "#fff");
      gr.append("text").attr("x", 12).attr("y", y + bh + 13).attr("font-size", 9.5).attr("fill", s[3])
        .text(s[0] + " → code " + Math.round(255 * s[1]) + ", light " + VZ.fmt(s[2], 4));
    });

    /* ── readout ── */
    const noiseEff = (function () {
      /* a noise of the given size added to the CODE, seen after decoding, at two levels */
      const at = L => {
        const V = enc(L);
        return Math.abs(dec(VZ.clamp(V + noise, 0, 1)) - L) / Math.max(1e-9, L);
      };
      return [at(0.02), at(0.8)];
    })();
    out.innerHTML =
      `<span class="keep">γ</span> = ${VZ.fmt(gm, 2)}${useS ? " (overridden by the exact sRGB curve)" : ""}, ${bits} bits = ${N + 1} codes.<br>`
      + `Half-way in <b>code</b> is L = <b>${VZ.fmt(midLin, 4)}</b> — only ${VZ.fmt(100 * midLin, 1)}% of the light, not 50%. `
      + `Half-way in <b>light</b> is code <b>${Math.round(255 * trueMidCode)}</b>, not 128. `
      + `Every resize, blur or blend done on raw code values makes exactly this error at every pixel.<br>`
      + `A transmission error of ${VZ.fmt(noise, 3)} in the stored code costs <b>${VZ.fmt(100 * noiseEff[0], 1)}%</b> of the `
      + `luminance at L = 0.02 but <b>${VZ.fmt(100 * noiseEff[1], 1)}%</b> at L = 0.8 — the encoding pushes the damage `
      + `into the highlights, where the eye cannot see it. That accident is why gamma outlived the hardware that needed it.`;
  }

  eG.oninput = () => { eGv.textContent = (+eG.value).toFixed(2); draw(); };
  eB.oninput = () => { eBv.textContent = eB.value; draw(); };
  eN.oninput = () => { eNv.textContent = (+eN.value).toFixed(3); draw(); };
  eS.onchange = draw;
  draw();
})();

/* ───────── 17 · the Bayer mosaic and demosaicing ───────── */
(function () {
  const svg = d3.select("#bayer-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 400;
  const gid = i => document.getElementById(i);
  const eSc = gid("by-scene"), eMe = gid("by-meth");
  const eZ = gid("by-z"), eZv = gid("by-zv"), eG = gid("by-grid");
  const out = gid("bayer-readout");

  /* the RGGB tile: (row even, col even) = R; (odd, odd) = B; the rest G */
  const chan = (r, c) => (r % 2 === 0) ? ((c % 2 === 0) ? 0 : 1) : ((c % 2 === 0) ? 1 : 2);

  let N = 40;                                  // photosites across; set by the slider

  function scene(kind) {
    const I = [];
    for (let r = 0; r < N; r++) {
      I.push([]);
      for (let c = 0; c < N; c++) {
        const x = c / (N - 1), y = r / (N - 1);
        let v;
        if (kind === "edge") {
          const t = (x + y > 1.0) ? 0.92 : 0.10;         // a purely LUMINANCE edge: grey to grey
          v = [t, t, t];
        } else if (kind === "bars") {
          const t = (c % 4 < 2) ? 0.9 : 0.12;
          v = [t, t, t];
        } else if (kind === "ring") {
          const dx = x - 0.5, dy = y - 0.5, rr = Math.hypot(dx, dy);
          const t = 0.5 + 0.45 * Math.cos(160 * rr * rr);
          v = [t, t, t];
        } else {
          v = [0.25 + 0.6 * x, 0.35 + 0.4 * y, 0.8 - 0.5 * x];
        }
        I[r].push(v);
      }
    }
    return I;
  }

  const at = (M, r, c) => M[VZ.clamp(r, 0, N - 1)][VZ.clamp(c, 0, N - 1)];

  function demosaic(mos, method) {
    /* mos[r][c] is the single measured value at that photosite */
    const G = [], out_ = [];
    const m = (r, c) => mos[VZ.clamp(r, 0, N - 1)][VZ.clamp(c, 0, N - 1)];
    const isG = (r, c) => chan(r, c) === 1;
    /* green first — it is the densely sampled channel */
    for (let r = 0; r < N; r++) {
      G.push([]);
      for (let c = 0; c < N; c++) {
        if (isG(r, c)) { G[r].push(m(r, c)); continue; }
        const n = m(r - 1, c), s = m(r + 1, c), w = m(r, c - 1), e = m(r, c + 1);
        if (method === "grad") {
          /* interpolate ALONG the smaller gradient, never across the larger one */
          const dv = Math.abs(n - s) + Math.abs(2 * m(r, c) - m(r - 2, c) - m(r + 2, c));
          const dh = Math.abs(w - e) + Math.abs(2 * m(r, c) - m(r, c - 2) - m(r, c + 2));
          G[r].push(dv < dh ? (n + s) / 2 : (dh < dv ? (w + e) / 2 : (n + s + w + e) / 4));
        } else {
          G[r].push((n + s + w + e) / 4);
        }
      }
    }
    /* then red and blue */
    for (let r = 0; r < N; r++) {
      out_.push([]);
      for (let c = 0; c < N; c++) {
        const val = [0, 0, 0];
        val[1] = G[r][c];
        [0, 2].forEach(ch => {
          if (chan(r, c) === ch) { val[ch] = m(r, c); return; }
          /* gather the same-colour neighbours */
          const cand = [];
          for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
            const rr = r + dr, cc = c + dc;
            if (dr === 0 && dc === 0) continue;
            const R2 = VZ.clamp(rr, 0, N - 1), C2 = VZ.clamp(cc, 0, N - 1);
            if (chan(R2, C2) !== ch) continue;
            cand.push(method === "bilinear" ? m(R2, C2) : m(R2, C2) - G[R2][C2]);
          }
          if (!cand.length) { val[ch] = G[r][c]; return; }
          const mean = cand.reduce((a, b) => a + b, 0) / cand.length;
          /* constant-hue and gradient methods interpolate the COLOUR DIFFERENCE and add G back */
          val[ch] = method === "bilinear" ? mean : mean + G[r][c];
        });
        out_[r].push(val.map(v => VZ.clamp(v, 0, 1)));
      }
    }
    return out_;
  }

  function draw() {
    const kind = eSc.value, method = eMe.value;
    N = 2 * Math.round(+eZ.value / 2);         // even, so the RGGB tile is not cut in half
    const orig = scene(kind);
    const mos = [];
    for (let r = 0; r < N; r++) { mos.push([]); for (let c = 0; c < N; c++) mos[r].push(orig[r][c][chan(r, c)]); }
    const rec = demosaic(mos, method);

    /* errors, measured */
    let se = 0, maxSat = 0, satSq = 0, nSat = 0;
    for (let r = 2; r < N - 2; r++) for (let c = 2; c < N - 2; c++) {
      for (let k = 0; k < 3; k++) se += Math.pow(rec[r][c][k] - orig[r][c][k], 2);
      /* False colour = saturation INTRODUCED where the scene had none. The maximum of
         this is a fragile statistic — one pixel right on the edge is as bad for every
         method — so the RMS over the frame is reported alongside it, and that is what
         actually separates the methods.                                              */
      const o = orig[r][c], q = rec[r][c];
      const oSat = Math.max(o[0], o[1], o[2]) - Math.min(o[0], o[1], o[2]);
      const pSat = Math.max(q[0], q[1], q[2]) - Math.min(q[0], q[1], q[2]);
      const d = Math.max(0, pSat - oSat);
      maxSat = Math.max(maxSat, d);
      satSq += d * d; nSat++;
    }
    const rmse = Math.sqrt(se / (3 * (N - 4) * (N - 4)));
    const rmsSat = Math.sqrt(satSq / Math.max(1, nSat));

    const fr = VZ.frame(svg, W, H, { l: 8, r: 8, t: 8, b: 8 });
    const g = fr.g;
    /* the four panels must FIT: derive their size from the frame, not from a slider */
    const gap = 10, side = Math.min((fr.iw - 5 * gap) / 4, fr.ih - 74);
    const z = side / N;
    const top = 34;

    function panel(k, title, get) {
      const x0 = gap + k * (side + gap);
      const gg = g.append("g").attr("transform", `translate(${x0},${top})`);
      gg.append("text").attr("x", 0).attr("y", -18).attr("font-size", 10.5).attr("fill", VC.ink)
        .attr("font-weight", 600).text(title);
      for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        const v = get(r, c);
        gg.append("rect").attr("x", c * z).attr("y", r * z).attr("width", z + 0.4).attr("height", z + 0.4)
          .attr("fill", d3.rgb(255 * v[0], 255 * v[1], 255 * v[2]).toString());
      }
      if (eG.checked && k === 1) {
        for (let i = 0; i <= N; i += 2) {
          gg.append("line").attr("x1", 0).attr("x2", side).attr("y1", i * z).attr("y2", i * z)
            .attr("stroke", VC.line).attr("stroke-width", 0.4);
          gg.append("line").attr("y1", 0).attr("y2", side).attr("x1", i * z).attr("x2", i * z)
            .attr("stroke", VC.line).attr("stroke-width", 0.4);
        }
      }
      gg.append("rect").attr("x", 0).attr("y", 0).attr("width", side).attr("height", side)
        .attr("fill", "none").attr("stroke", VC.line);
      return gg;
    }

    /* gamma-encode for display, since these are linear-ish values */
    const enc = v => v.map(x => Math.pow(VZ.clamp(x, 0, 1), 1 / 2.2));
    panel(0, "the scene", (r, c) => enc(orig[r][c]));
    panel(1, "what the sensor records", (r, c) => {
      const v = [0, 0, 0]; v[chan(r, c)] = mos[r][c]; return enc(v);
    });
    panel(2, "demosaiced · " + eMe.options[eMe.selectedIndex].text.split(" (")[0], (r, c) => enc(rec[r][c]));
    panel(3, "error × 6", (r, c) => {
      const d = [0, 1, 2].map(k => Math.abs(rec[r][c][k] - orig[r][c][k]) * 6);
      return d.map(v => VZ.clamp(v, 0, 1));
    });

    /* readouts */
    const by = top + side + 26;
    g.append("text").attr("x", gap).attr("y", by).attr("font-size", 10).attr("fill", VC.muted)
      .text("2 × 2 tile: R G / G B — 50% green, 25% red, 25% blue. One measurement per photosite.");
    g.append("text").attr("x", gap).attr("y", by + 16).attr("font-size", 10.5)
      .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", rmse < 0.02 ? VC.good : (rmse < 0.06 ? VC.a2 : VC.bad))
      .text("RMSE " + VZ.fmt(rmse, 5) + "   ·   false colour introduced: RMS " + VZ.fmt(rmsSat, 5) + ", worst " + VZ.fmt(maxSat, 4));

    out.innerHTML =
      `<b>${eSc.options[eSc.selectedIndex].text}</b>, reconstructed by <b>${eMe.options[eMe.selectedIndex].text}</b>. `
      + `RMSE against the true scene = <b>${VZ.fmt(rmse, 5)}</b>; the colour invented where the scene was `
      + `neutral averages <b>${VZ.fmt(rmsSat, 5)}</b> and peaks at <b>${VZ.fmt(maxSat, 4)}</b> — the peak is one `
      + `pixel on the edge and barely moves between methods; the average is what separates them.<br>`
      + (kind === "edge" || kind === "bars"
        ? `The scene here is <b>purely a luminance pattern</b> — every true pixel is grey. Any colour in the third `
        + `panel is entirely manufactured by the interpolation, and any alternating light–dark pattern along the `
        + `edge is the zipper artefact. `
        : ``)
      + (method === "bilinear"
        ? `Bilinear interpolates R and B directly, so an edge splits unevenly between the channels and the result fringes.`
        : method === "hue"
          ? `Interpolating R−G and B−G instead exploits the fact that colour differences are smooth across a luminance `
          + `edge even when the channels are not — most of the false colour disappears for no extra cost.`
          : `The gradient-corrected method estimates the edge direction first and interpolates along it. On an `
          + `axis-aligned pattern that is exact; on a 45° edge the two gradients tie and it falls back to the `
          + `four-neighbour average, so it matches the constant-hue result; and on the zone plate, where the `
          + `frequencies run past Nyquist, the direction estimate is unreliable and it does no better than bilinear.`);
  }

  eSc.onchange = draw; eMe.onchange = draw; eG.onchange = draw;
  eZ.oninput = () => { eZv.textContent = 2 * Math.round(+eZ.value / 2); draw(); };
  eZv.textContent = 2 * Math.round(+eZ.value / 2);
  draw();
})();

/* ───────── 18 · exposure, gain, and the three noise floors ───────── */
(function () {
  const svg = d3.select("#noise-svg");
  if (svg.empty() || typeof VZ === "undefined") return;

  const W = 760, H = 430;
  const gid = i => document.getElementById(i);
  const eE = gid("ns-e"), eEv = gid("ns-ev"), eR = gid("ns-r"), eRv = gid("ns-rv");
  const eG = gid("ns-g"), eGv = gid("ns-gv"), eB = gid("ns-b"), eBv = gid("ns-bv");
  const eW = gid("ns-w"), eWv = gid("ns-wv");
  const out = gid("noise-readout");

  const PATCH = 46;                                  // pixels across one simulated patch

  function draw() {
    const ev = +eE.value, sread = +eR.value, gainStop = +eG.value;
    const bits = +eB.value, well = +eW.value;
    const gain = Math.pow(2, gainStop), iso = 100 * gain;
    const levels = [0.002, 0.008, 0.03, 0.12, 0.45, 1.0];    // scene reflectances
    const expo = Math.pow(2, ev);
    /* electrons collected, capped by the well */
    const elec = levels.map(L => Math.min(well, L * well * expo));
    /* the ADC spans the well AFTER gain, so gain reduces the electrons per code */
    const codes = Math.pow(2, bits) - 1;
    const ePerCode = well / gain / codes;
    const sQuant = ePerCode / Math.sqrt(12);

    const rng = VZ.rng(20260908);

    const fr = VZ.frame(svg, W, H, { l: 8, r: 8, t: 8, b: 8 });
    const g = fr.g;
    const leftW = 356, rightW = fr.iw - leftW - 14;

    /* ── left: simulated patches ── */
    const gl = g.append("g");
    gl.append("rect").attr("x", 0).attr("y", 0).attr("width", leftW).attr("height", fr.ih)
      .attr("rx", 8).attr("fill", VC.bg).attr("fill-opacity", 0.45).attr("stroke", VC.line);
    gl.append("text").attr("x", 10).attr("y", 18).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("six patches, simulated photon by photon");
    const cols = 3, rows = 2, pw = (leftW - 40) / cols, phh = (fr.ih - 70) / rows;
    const sz = Math.min(pw, phh) - 10;
    const measured = [];
    levels.forEach((L, k) => {
      const S = elec[k];
      const gx = 20 + (k % cols) * pw + (pw - sz) / 2, gy = 42 + Math.floor(k / cols) * phh;
      const vals = [];
      const nsub = Math.max(8, Math.round(sz / 3));
      const cell = sz / nsub;
      for (let i = 0; i < nsub; i++) for (let j = 0; j < nsub; j++) {
        const shot = VZ.poisson(S, rng);
        const read = sread * VZ.randn(rng);
        const raw = VZ.clamp((shot + read) * gain, 0, well);
        const q = Math.round(raw / (well / codes)) * (well / codes);
        const frac = VZ.clamp(q / well, 0, 1);
        vals.push(frac / Math.max(1e-9, gain));
        const disp = Math.round(255 * Math.pow(VZ.clamp(frac, 0, 1), 1 / 2.2));
        gl.append("rect").attr("x", gx + i * cell).attr("y", gy + j * cell)
          .attr("width", cell + 0.5).attr("height", cell + 0.5)
          .attr("fill", `rgb(${disp},${disp},${disp})`);
      }
      const mu = vals.reduce((a, b) => a + b, 0) / vals.length;
      const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mu) * (b - mu), 0) / (vals.length - 1));
      measured.push({ S: S, snr: sd > 0 ? mu / sd : Infinity });
      gl.append("rect").attr("x", gx).attr("y", gy).attr("width", sz).attr("height", sz)
        .attr("fill", "none").attr("stroke", VC.line);
      gl.append("text").attr("x", gx).attr("y", gy + sz + 12).attr("font-size", 9)
        .attr("fill", VC.muted).text(VZ.fmt(S, 0) + " e⁻ · SNR " + (S > 0 ? VZ.fmt(S / Math.sqrt(S + sread * sread + sQuant * sQuant), 1) : "0"));
    });

    /* ── right: the noise budget ── */
    const gr = g.append("g").attr("transform", `translate(${leftW + 14},0)`);
    gr.append("rect").attr("x", 0).attr("y", 0).attr("width", rightW).attr("height", fr.ih)
      .attr("rx", 8).attr("fill", VC.panel2).attr("fill-opacity", 0.6).attr("stroke", VC.line);
    const m = { l: 50, r: 16, t: 32, b: 152 };
    const pw2 = rightW - m.l - m.r, ph2 = fr.ih - m.t - m.b;
    const gg = gr.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const X = d3.scaleLog().domain([1, well]).range([0, pw2]);
    const Y = d3.scaleLog().domain([0.2, Math.max(300, Math.sqrt(well) * 2)]).range([ph2, 0]);
    gg.append("g").attr("class", "axis").attr("transform", `translate(0,${ph2})`)
      .call(d3.axisBottom(X).ticks(4, "~s"));
    gg.append("g").attr("class", "axis").call(d3.axisLeft(Y).ticks(4, "~s"));
    gg.append("text").attr("x", pw2).attr("y", ph2 + 30).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", VC.muted).text("signal S (electrons)");
    gg.append("text").attr("x", pw2).attr("y", -10).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", VC.muted).text("noise σ (electrons) ↑");

    const curves = [
      { n: "shot √S", c: VC.accent, f: S => Math.sqrt(S) },
      { n: "read", c: VC.bad, f: () => sread },
      { n: "quantisation", c: VC.violet, f: () => sQuant },
      { n: "total", c: VC.good, f: S => Math.sqrt(S + sread * sread + sQuant * sQuant) }
    ];
    curves.forEach(cv => {
      const pts = [];
      for (let i = 0; i <= 200; i++) {
        const S = Math.pow(10, Math.log10(1) + (Math.log10(well) - Math.log10(1)) * i / 200);
        pts.push([X(S), Y(VZ.clamp(cv.f(S), 0.2, 1e6))]);
      }
      VZ.poly(gg, pts, { stroke: cv.c, w: cv.n === "total" ? 2.2 : 1.4, close: false, dash: cv.n === "total" ? null : "4 3" });
    });
    /* the crossover where shot noise overtakes the floor */
    const floor2 = sread * sread + sQuant * sQuant;
    if (floor2 > 1 && floor2 < well) {
      gg.append("line").attr("x1", X(floor2)).attr("x2", X(floor2)).attr("y1", 0).attr("y2", ph2)
        .attr("stroke", VC.a2).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
      gg.append("text").attr("x", X(floor2) + 4).attr("y", 12).attr("font-size", 9)
        .attr("fill", VC.a2).text("S = σ²floor = " + VZ.fmt(floor2, 1) + " e⁻");
    }
    VZ.legend(gg, curves.map(c => ({ label: c.n, color: c.c })), pw2 - 78, 12, { gap: 13 });
    gr.append("text").attr("x", 12).attr("y", 18).attr("font-size", 11).attr("fill", VC.ink)
      .attr("font-weight", 600).text("σ²total = S + σ²read + q²/12");

    const dr = well / Math.sqrt(floor2);
    let yy = fr.ih - 112;
    const put = (lab, val, col) => {
      gr.append("text").attr("x", 12).attr("y", yy).attr("font-size", 9.5).attr("fill", VC.muted).text(lab);
      gr.append("text").attr("x", rightW - 12).attr("y", yy).attr("text-anchor", "end").attr("font-size", 10)
        .attr("font-family", "SF Mono, Menlo, monospace").attr("fill", col || VC.ink).text(val);
      yy += 16;
    };
    put("gain", "×" + VZ.fmt(gain, 0) + "  (ISO " + iso + ")", VC.a2);
    put("electrons per ADC code", VZ.fmt(ePerCode, 3) + " e⁻");
    put("quantisation σ = q/√12", VZ.fmt(sQuant, 3) + " e⁻", sQuant > sread ? VC.bad : VC.good);
    put("read noise σ", VZ.fmt(sread, 2) + " e⁻");
    put("shot overtakes the floor at", VZ.fmt(floor2, 1) + " e⁻", VC.accent);
    put("dynamic range", VZ.fmt(dr, 0) + " : 1 = " + VZ.fmt(20 * Math.log10(dr), 1) + " dB = "
      + VZ.fmt(Math.log2(dr), 2) + " stops", VC.good);
    put("SNR at full well", VZ.fmt(well / Math.sqrt(well + floor2), 1)
      + " = " + VZ.fmt(20 * Math.log10(well / Math.sqrt(well + floor2)), 1) + " dB");

    /* ── readout ── */
    const dominatesQ = sQuant > sread;
    out.innerHTML =
      `exposure ${ev >= 0 ? "+" : ""}${VZ.fmt(ev, 2)} EV (×${VZ.fmt(expo, 3)} light), gain ×${VZ.fmt(gain, 0)} = ISO ${iso}, `
      + `${bits}-bit ADC over a ${well} e⁻ well.<br>`
      + `One code is worth <b>${VZ.fmt(ePerCode, 3)} e⁻</b>, so quantisation contributes <b>${VZ.fmt(sQuant, 3)} e⁻</b> — `
      + (dominatesQ
        ? `<b style="color:var(--bad)">more than the ${VZ.fmt(sread, 2)} e⁻ read noise</b>, so the converter is now the limiting component. Add bits, or add gain.`
        : `comfortably below the ${VZ.fmt(sread, 2)} e⁻ read noise, so the converter is not the bottleneck.`)
      + `<br>Shot noise overtakes everything else above <b>${VZ.fmt(floor2, 1)} e⁻</b>. Above that point the grain you see `
      + `is the <i>light itself</i> and no sensor design removes it — only more photons do, and doubling the exposure `
      + `buys exactly √2 = 1.414× in SNR, half a stop of quality per stop of light.<br>`
      + `Note that raising the gain does <b>not</b> change the SNR of the collected photons: it multiplies signal and `
      + `shot noise by the same factor. All it does is lift both clear of the quantiser.`;
  }

  eE.oninput = () => { eEv.textContent = (+eE.value).toFixed(1) + " EV"; draw(); };
  eR.oninput = () => { eRv.textContent = (+eR.value).toFixed(1) + " e⁻"; draw(); };
  eG.oninput = () => { eGv.textContent = "ISO " + (100 * Math.pow(2, +eG.value)); draw(); };
  eB.oninput = () => { eBv.textContent = eB.value; draw(); };
  eW.oninput = () => { eWv.textContent = eW.value + " e⁻"; draw(); };
  draw();
})();
