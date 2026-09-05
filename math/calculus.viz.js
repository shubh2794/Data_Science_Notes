/* calculus.viz.js — extracted from calculus.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ── Calculus page · inline D3 visualizations ──────────────────────────────
   Palette C (A, B, good, bad, ink, muted, line) comes from ../notes.js.      */
const P = (typeof C !== "undefined") ? C
        : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
const fmt = (v, d) => (Math.abs(v) < 1e-12 ? 0 : v).toFixed(d === undefined ? 3 : d);

function arrowDefs(svg, specs){
  const defs = svg.append("defs");
  specs.forEach(([id, col]) => defs.append("marker")
    .attr("id", id).attr("viewBox", "0 0 10 10").attr("refX", 8).attr("refY", 5)
    .attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto-start-reverse")
    .append("path").attr("d", "M0,0L10,5L0,10").attr("fill", col));
}
function axisBox(svg, m, W, H, xl, yl){
  svg.append("line").attr("x1", m.l).attr("y1", H - m.b).attr("x2", W - m.r).attr("y2", H - m.b)
     .attr("stroke", P.line).attr("stroke-width", 1);
  svg.append("line").attr("x1", m.l).attr("y1", m.t).attr("x2", m.l).attr("y2", H - m.b)
     .attr("stroke", P.line).attr("stroke-width", 1);
  svg.append("text").attr("x", W - m.r).attr("y", H - m.b + 22).attr("text-anchor", "end")
     .attr("fill", P.muted).attr("font-size", 11).text(xl);
  svg.append("text").attr("x", m.l - 6).attr("y", m.t + 2).attr("text-anchor", "end")
     .attr("fill", P.muted).attr("font-size", 11).text(yl);
}

/* ═══════════ §03 · secant → tangent ═══════════════════════════════════════ */
(function(){
  const svg = d3.select("#tan-svg"); if (svg.empty()) return;
  const W = 660, H = 380, m = {l:46, r:24, t:24, b:40};
  const f  = x => 0.22*x*x*x - 1.1*x + 3.2;
  const df = x => 0.66*x*x - 1.1;
  const xs = d3.scaleLinear().domain([-3.2, 3.2]).range([m.l, W - m.r]);
  const ys = d3.scaleLinear().domain([-1.2, 7.6]).range([H - m.b, m.t]);
  arrowDefs(svg, [["tanA", P.good]]);
  axisBox(svg, m, W, H, "x", "f(x)");
  for (let g = -3; g <= 3; g++)
    svg.append("line").attr("x1", xs(g)).attr("x2", xs(g)).attr("y1", m.t).attr("y2", H - m.b)
       .attr("stroke", P.line).attr("stroke-opacity", .45);
  const pts = []; for (let x = -3.2; x <= 3.2001; x += 0.02) pts.push([xs(x), ys(f(x))]);
  svg.append("path").attr("d", d3.line()(pts)).attr("fill", "none")
     .attr("stroke", P.A).attr("stroke-width", 2.5);

  const tri  = svg.append("path").attr("fill", P.good).attr("fill-opacity", .12)
                  .attr("stroke", P.good).attr("stroke-opacity", .45).attr("stroke-dasharray", "3 3");
  const tang = svg.append("line").attr("stroke", P.B).attr("stroke-width", 2.5);
  const sec  = svg.append("line").attr("stroke", P.good).attr("stroke-width", 2.5);
  const p2   = svg.append("circle").attr("r", 5).attr("fill", P.good).attr("fill-opacity", .85)
                  .attr("stroke", "#0f1117").attr("stroke-width", 1.5);
  const p1   = svg.append("circle").attr("r", 8).attr("fill", P.A)
                  .attr("stroke", "#0f1117").attr("stroke-width", 2)
                  .attr("class", "dragpt").style("cursor", "grab");
  const lbl  = svg.append("text").attr("font-size", 11).attr("fill", P.good);
  const out  = document.getElementById("tan-readout");
  const hIn  = document.getElementById("tan-h");

  let x0 = -1.55, timer = null;
  const hOf = () => Math.pow(10, -3 + (+hIn.value / 100) * (Math.log10(2) + 3));

  function draw(){
    const h = hOf(), y0 = f(x0), x1 = Math.min(3.2, x0 + h), y1 = f(x1);
    const sl = (f(x0 + h) - y0) / h, tr = df(x0), err = Math.abs(sl - tr);
    p1.attr("cx", xs(x0)).attr("cy", ys(y0));
    p2.attr("cx", xs(x1)).attr("cy", ys(y1));
    const ext = (x, s) => { const L = 2.6;
      return [[xs(x0 - L), ys(y0 - L*s)], [xs(x0 + L), ys(y0 + L*s)]]; };
    const t = ext(x0, tr), c = ext(x0, sl);
    tang.attr("x1", t[0][0]).attr("y1", t[0][1]).attr("x2", t[1][0]).attr("y2", t[1][1]);
    sec .attr("x1", c[0][0]).attr("y1", c[0][1]).attr("x2", c[1][0]).attr("y2", c[1][1]);
    tri.attr("d", `M${xs(x0)},${ys(y0)} L${xs(x1)},${ys(y0)} L${xs(x1)},${ys(y1)} Z`);
    lbl.attr("x", m.l + 12).attr("y", m.t + 16)
       .text(`h = ${fmt(h, 4)}   Δf = ${fmt(y1 - y0, 4)}   Δf/h = ${fmt(sl, 4)}`);
    out.innerHTML = `x₀ = <b>${fmt(x0, 2)}</b> · h = <b>${fmt(h, 5)}</b> · `
      + `secant slope [f(x₀+h) − f(x₀)]/h = <b>${fmt(sl, 5)}</b> · `
      + `tangent slope f′(x₀) = <b>${fmt(tr, 5)}</b> · `
      + `|error| = <b>${err.toExponential(2)}</b> ≈ ${fmt(err / h, 2)}·h  (first-order, O(h))`;
  }
  p1.call(d3.drag()
    .on("start", function(){ d3.select(this).style("cursor", "grabbing"); })
    .on("drag", ev => { x0 = Math.max(-3.1, Math.min(2.6, xs.invert(ev.x))); draw(); })
    .on("end", function(){ d3.select(this).style("cursor", "grab"); }));
  hIn.addEventListener("input", draw);
  document.getElementById("tan-anim").addEventListener("click", function(){
    if (timer){ clearInterval(timer); timer = null; this.textContent = "▶ shrink h"; return; }
    this.textContent = "■ stop";
    const btn = this;
    timer = setInterval(() => {
      const v = +hIn.value - 2;
      if (v <= 0){ hIn.value = 0; draw(); clearInterval(timer); timer = null; btn.textContent = "▶ shrink h"; return; }
      hIn.value = v; draw();
    }, 60);
  });
  draw();
})();

/* ═══════════ §05 · Taylor polynomial builder ═════════════════════════════ */
(function(){
  const svg = d3.select("#tay-svg"); if (svg.empty()) return;
  const W = 660, H = 400, m = {l:48, r:24, t:22, b:40};

  const FN = {
    sin : {label:"sin x",     f: x => Math.sin(x),        dom:[-7.5, 7.5], rng:[-2.4, 2.4],  aRng:[-3, 3],
           d:(k,a) => [Math.sin(a), Math.cos(a), -Math.sin(a), -Math.cos(a)][k % 4], R: () => Infinity},
    exp : {label:"eˣ",        f: x => Math.exp(x),        dom:[-3, 3],     rng:[-1.5, 11],   aRng:[-2, 2],
           d:(k,a) => Math.exp(a), R: () => Infinity},
    log : {label:"ln(1+x)",   f: x => Math.log(1 + x),    dom:[-0.92, 3.4],rng:[-3.4, 2.2],  aRng:[-0.5, 2],
           d:(k,a) => k === 0 ? Math.log(1 + a)
                              : (k % 2 ? 1 : -1) * fact(k - 1) / Math.pow(1 + a, k),
           R: a => 1 + a},
    geo : {label:"1/(1−x)",   f: x => 1 / (1 - x),        dom:[-3.4, 0.92],rng:[-1.5, 6.5],  aRng:[-2, 0.5],
           d:(k,a) => fact(k) / Math.pow(1 - a, k + 1), R: a => 1 - a},
    sqrt: {label:"√(1+x)",    f: x => Math.sqrt(1 + x),   dom:[-0.98, 3.4],rng:[-1.2, 2.6],  aRng:[-0.5, 2],
           d:(k,a) => { let c = 1; for (let i = 0; i < k; i++) c *= (0.5 - i);
                        return c * Math.pow(1 + a, 0.5 - k); }, R: a => 1 + a}
  };
  function fact(n){ let r = 1; for (let i = 2; i <= n; i++) r *= i; return r; }

  const gAx = svg.append("g"), gPlot = svg.append("g"), gHud = svg.append("g");
  const selFn = document.getElementById("tay-fn");
  const aIn   = document.getElementById("tay-a");
  const out   = document.getElementById("tay-readout");
  let order = 1;

  function coeffs(F, a, n){
    const c = []; for (let k = 0; k <= n; k++) c.push(F.d(k, a) / fact(k)); return c;
  }
  function draw(){
    const F = FN[selFn.value];
    const a = F.aRng[0] + (+aIn.value / 100) * (F.aRng[1] - F.aRng[0]);
    const xs = d3.scaleLinear().domain(F.dom).range([m.l, W - m.r]);
    const ys = d3.scaleLinear().domain(F.rng).range([H - m.b, m.t]);
    gAx.selectAll("*").remove(); gPlot.selectAll("*").remove(); gHud.selectAll("*").remove();

    gAx.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", ys(0)).attr("y2", ys(0))
       .attr("stroke", P.line);
    gAx.append("line").attr("x1", xs(0)).attr("x2", xs(0)).attr("y1", m.t).attr("y2", H - m.b)
       .attr("stroke", P.line);
    gAx.append("text").attr("x", W - m.r).attr("y", H - m.b + 22).attr("text-anchor", "end")
       .attr("fill", P.muted).attr("font-size", 11).text("x");

    const c = coeffs(F, a, order);
    const T = x => { let s = 0, p = 1; for (let k = 0; k <= order; k++){ s += c[k] * p; p *= (x - a); } return s; };
    const N = 460, step = (F.dom[1] - F.dom[0]) / N;
    const fp = [], tp = [], band = [];
    let maxErr = 0;
    for (let i = 0; i <= N; i++){
      const x = F.dom[0] + i * step, yf = F.f(x), yt = T(x);
      const okF = isFinite(yf), okT = isFinite(yt);
      fp.push({x: xs(x), y: ys(yf), ok: okF && yf > F.rng[0] - 1 && yf < F.rng[1] + 1});
      tp.push({x: xs(x), y: ys(yt), ok: okT && yt > F.rng[0] - 1 && yt < F.rng[1] + 1});
      if (okF && okT && Math.abs(x - a) < 2.2) maxErr = Math.max(maxErr, Math.abs(yf - yt));
      if (fp[i].ok && tp[i].ok) band.push([x, yf, yt]);
    }
    const ln = d3.line().defined(d => d.ok).x(d => d.x).y(d => d.y);
    band.forEach((b, i) => { if (i % 4) return;              // remainder |f - Tn| as hatch lines
      if (Math.abs(b[1] - b[2]) < 1e-4) return;
      gPlot.append("line").attr("x1", xs(b[0])).attr("x2", xs(b[0]))
           .attr("y1", ys(b[1])).attr("y2", ys(b[2]))
           .attr("stroke", P.bad).attr("stroke-opacity", .3).attr("stroke-width", 1.8);
    });
    gPlot.append("path").attr("d", ln(fp)).attr("fill", "none").attr("stroke", P.A).attr("stroke-width", 2.6);
    gPlot.append("path").attr("d", ln(tp)).attr("fill", "none").attr("stroke", P.B).attr("stroke-width", 2.4)
         .attr("stroke-dasharray", "6 3");

    const R = F.R(a);
    if (isFinite(R)){
      [a - R, a + R].forEach(b => { if (b > F.dom[0] && b < F.dom[1])
        gHud.append("line").attr("x1", xs(b)).attr("x2", xs(b)).attr("y1", m.t).attr("y2", H - m.b)
            .attr("stroke", P.bad).attr("stroke-opacity", .5).attr("stroke-dasharray", "4 4"); });
    }
    gHud.append("line").attr("x1", xs(a)).attr("x2", xs(a)).attr("y1", m.t).attr("y2", H - m.b)
        .attr("stroke", P.good).attr("stroke-opacity", .5);
    gHud.append("circle").attr("cx", xs(a)).attr("cy", ys(F.f(a))).attr("r", 5.5)
        .attr("fill", P.good).attr("stroke", "#0f1117").attr("stroke-width", 2);
    gHud.append("text").attr("x", xs(a) + 8).attr("y", m.t + 12).attr("font-size", 11)
        .attr("fill", P.good).text("centre a = " + fmt(a, 2));

    const terms = [];
    for (let k = 0; k <= order && k <= 6; k++){
      const v = c[k];
      if (Math.abs(v) < 1e-12) continue;
      terms.push((v > 0 && terms.length ? "+ " : v < 0 ? "− " : "")
        + fmt(Math.abs(v), 4) + (k === 0 ? "" : k === 1 ? "·(x−a)" : "·(x−a)" + supOf(k)));
    }
    out.innerHTML = `<b>${FN[selFn.value].label}</b> · order n = <b>${order}</b> · centre a = <b>${fmt(a, 2)}</b>`
      + ` · radius of convergence R = <b>${isFinite(R) ? fmt(R, 2) : "∞"}</b>`
      + ` · max |f − Tₙ| within 2.2 of a = <b>${maxErr > 1e6 ? "diverges" : maxErr.toExponential(2)}</b>`
      + `<br>Tₙ(x) = ${terms.join(" ")}${order > 6 ? " + …" : ""}`;
  }
  function supOf(k){ const s = "⁰¹²³⁴⁵⁶⁷⁸⁹"; return String(k).split("").map(d => s[+d]).join(""); }

  selFn.addEventListener("change", () => { order = 1; draw(); });
  aIn.addEventListener("input", draw);
  document.getElementById("tay-plus").addEventListener("click", () => { order = Math.min(24, order + 1); draw(); });
  document.getElementById("tay-minus").addEventListener("click", () => { order = Math.max(0, order - 1); draw(); });
  document.getElementById("tay-reset").addEventListener("click", () => { order = 1; aIn.value = 50; draw(); });
  draw();
})();

/* ═══════════ §08 · gradient field & directional derivative ═══════════════ */
(function(){
  const svg = d3.select("#gr-svg"); if (svg.empty()) return;
  const W = 660, H = 430, m = {l:40, r:170, t:20, b:34};
  const PW = W - m.l - m.r, PH = H - m.t - m.b;

  const SURF = {
    bowl  : {f:(x,y) => 0.5*x*x + 2*y*y,                      name:"0.5x² + 2y²"},
    tilt  : {f:(x,y) => x*x + 3*y*y + 1.6*x*y,                name:"x² + 3y² + 1.6xy"},
    saddle: {f:(x,y) => x*x - y*y,                            name:"x² − y²"},
    wave  : {f:(x,y) => Math.sin(x)*Math.cos(y) + 0.15*(x*x + y*y), name:"sin x·cos y + 0.15(x²+y²)"}
  };
  const D = 3.2;
  const xs = d3.scaleLinear().domain([-D, D]).range([m.l, m.l + PW]);
  const ys = d3.scaleLinear().domain([-D, D]).range([m.t + PH, m.t]);
  const NX = 96, NY = 96;

  arrowDefs(svg, [["grG", P.bad], ["grU", P.good]]);
  const gGrid = svg.append("g"), gAx = svg.append("g"), gArt = svg.append("g"), gPanel = svg.append("g");
  const sel = document.getElementById("gr-fn");
  const cur = document.getElementById("gr-curv");
  const out = document.getElementById("gr-readout");

  let px = -1.6, py = 1.1, ang = -0.6;

  function grad(f, x, y){ const h = 1e-5;
    return [(f(x+h,y) - f(x-h,y)) / (2*h), (f(x,y+h) - f(x,y-h)) / (2*h)]; }
  function hess(f, x, y){ const h = 1e-3;
    const fxx = (f(x+h,y) - 2*f(x,y) + f(x-h,y)) / (h*h);
    const fyy = (f(x,y+h) - 2*f(x,y) + f(x,y-h)) / (h*h);
    const fxy = (f(x+h,y+h) - f(x+h,y-h) - f(x-h,y+h) + f(x-h,y-h)) / (4*h*h);
    return [fxx, fxy, fyy]; }
  function eig2(a, b, c){                    // [[a,b],[b,c]]
    const tr = a + c, dt = a*c - b*b, s = Math.sqrt(Math.max(0, tr*tr/4 - dt));
    const l1 = tr/2 + s, l2 = tr/2 - s;
    const v = l => { let vx = b, vy = l - a;
      if (Math.abs(vx) + Math.abs(vy) < 1e-9){ vx = 1; vy = 0; }
      const n = Math.hypot(vx, vy); return [vx/n, vy/n]; };
    return [[l1, v(l1)], [l2, v(l2)]];
  }

  function drawField(){
    const f = SURF[sel.value].f;
    gGrid.selectAll("*").remove(); gAx.selectAll("*").remove();
    const vals = new Array(NX * NY);
    let lo = Infinity, hi = -Infinity;
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++){
      const x = -D + 2*D*i/(NX-1), y = D - 2*D*j/(NY-1);
      const v = f(x, y); vals[j*NX + i] = v; lo = Math.min(lo, v); hi = Math.max(hi, v);
    }
    const col = d3.scaleLinear().domain([lo, (lo+hi)/2, hi])
                  .range(["#101a2e", "#22314d", "#3d4d70"]);
    const thr = d3.range(10).map(k => lo + (hi - lo) * Math.pow((k + 0.5)/10, 1.25));
    const g = gGrid.append("g")
      .attr("transform", `translate(${m.l},${m.t}) scale(${PW/(NX-1)},${PH/(NY-1)})`);
    const path = d3.geoPath();
    d3.contours().size([NX, NY]).thresholds(thr)(vals).forEach(ct => {
      g.append("path").attr("d", path(ct)).attr("fill", col(ct.value))
       .attr("stroke", "#4a5a80").attr("stroke-opacity", .55)
       .attr("stroke-width", (NX-1)/PW).attr("vector-effect", "non-scaling-stroke");
    });
    gAx.append("rect").attr("x", m.l).attr("y", m.t).attr("width", PW).attr("height", PH)
       .attr("fill", "none").attr("stroke", P.line);
    gAx.append("text").attr("x", m.l + PW).attr("y", m.t + PH + 22).attr("text-anchor", "end")
       .attr("fill", P.muted).attr("font-size", 11).text("x");
    gAx.append("text").attr("x", m.l - 6).attr("y", m.t + 10).attr("text-anchor", "end")
       .attr("fill", P.muted).attr("font-size", 11).text("y");
  }

  function draw(){
    const S = SURF[sel.value], f = S.f;
    gArt.selectAll("*").remove(); gPanel.selectAll("*").remove();
    const g = grad(f, px, py), gn = Math.hypot(g[0], g[1]);
    const u = [Math.cos(ang), Math.sin(ang)];
    const Du = g[0]*u[0] + g[1]*u[1];
    const phi = Math.acos(Math.max(-1, Math.min(1, gn > 1e-9 ? Du/gn : 0)));

    // level set through the point
    const lv = f(px, py);
    const vals = new Array(NX * NY);
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++){
      vals[j*NX + i] = f(-D + 2*D*i/(NX-1), D - 2*D*j/(NY-1));
    }
    const gL = gArt.append("g")
      .attr("transform", `translate(${m.l},${m.t}) scale(${PW/(NX-1)},${PH/(NY-1)})`);
    d3.contours().size([NX, NY]).thresholds([lv])(vals).forEach(ct => {
      gL.append("path").attr("d", d3.geoPath()(ct)).attr("fill", "none")
        .attr("stroke", P.B).attr("stroke-width", 2).attr("vector-effect", "non-scaling-stroke");
    });

    const SC = 46;                                    // px per unit for arrows
    const gl = Math.min(110, gn * SC / 2 + 18);
    const gdir = gn > 1e-9 ? [g[0]/gn, g[1]/gn] : [0, 0];
    gArt.append("line").attr("x1", xs(px)).attr("y1", ys(py))
        .attr("x2", xs(px) + gdir[0]*gl).attr("y2", ys(py) - gdir[1]*gl)
        .attr("stroke", P.bad).attr("stroke-width", 3).attr("marker-end", "url(#grG)");
    gArt.append("line").attr("x1", xs(px)).attr("y1", ys(py))
        .attr("x2", xs(px) - gdir[0]*gl*0.55).attr("y2", ys(py) + gdir[1]*gl*0.55)
        .attr("stroke", P.bad).attr("stroke-width", 1.5).attr("stroke-dasharray", "4 3")
        .attr("stroke-opacity", .6);
    gArt.append("line").attr("x1", xs(px)).attr("y1", ys(py))
        .attr("x2", xs(px) + u[0]*70).attr("y2", ys(py) - u[1]*70)
        .attr("stroke", P.good).attr("stroke-width", 3).attr("marker-end", "url(#grU)");

    if (cur.checked){
      const h = hess(f, px, py), ev = eig2(h[0], h[1], h[2]);
      ev.forEach(([lam, v]) => {
        const L = 22 + 40 / (1 + Math.exp(-Math.abs(lam)/2));
        const c = lam >= 0 ? P.A : P.bad;
        gArt.append("line").attr("x1", xs(px) - v[0]*L).attr("y1", ys(py) + v[1]*L)
            .attr("x2", xs(px) + v[0]*L).attr("y2", ys(py) - v[1]*L)
            .attr("stroke", c).attr("stroke-width", 2.5).attr("stroke-opacity", .95)
            .attr("stroke-dasharray", "2 3");
      });
    }
    gArt.append("circle").attr("cx", xs(px)).attr("cy", ys(py)).attr("r", 7)
        .attr("fill", P.ink).attr("stroke", "#0f1117").attr("stroke-width", 2)
        .attr("class", "dragpt").style("cursor", "grab")
        .call(d3.drag().on("drag", ev => {
          px = Math.max(-D, Math.min(D, xs.invert(ev.x)));
          py = Math.max(-D, Math.min(D, ys.invert(ev.y))); draw();
        }));
    gArt.append("circle").attr("cx", xs(px) + u[0]*70).attr("cy", ys(py) - u[1]*70).attr("r", 6.5)
        .attr("fill", P.good).attr("stroke", "#0f1117").attr("stroke-width", 2)
        .attr("class", "dragpt").style("cursor", "grab")
        .call(d3.drag().on("drag", ev => {
          ang = Math.atan2(ys.invert(ev.y) - py, xs.invert(ev.x) - px); draw();
        }));

    // side panel: D_u φ as a signed bar against ±‖∇φ‖
    const bx = m.l + PW + 26, bw = 118, by = m.t + 40, bh = 150, mid = by + bh/2;
    gPanel.append("text").attr("x", bx).attr("y", by - 20).attr("font-size", 11).attr("fill", P.muted)
          .text("D_u φ = ∇φ·u");
    gPanel.append("rect").attr("x", bx).attr("y", by).attr("width", bw).attr("height", bh)
          .attr("fill", "#11141b").attr("stroke", P.line).attr("rx", 6);
    gPanel.append("line").attr("x1", bx).attr("x2", bx + bw).attr("y1", mid).attr("y2", mid)
          .attr("stroke", P.line);
    const sc = gn > 1e-9 ? (bh/2 - 10) / gn : 0;
    gPanel.append("rect").attr("x", bx + bw/2 - 16)
          .attr("y", Du >= 0 ? mid - Du*sc : mid)
          .attr("width", 32).attr("height", Math.max(1, Math.abs(Du)*sc))
          .attr("fill", Du >= 0 ? P.good : P.bad);
    [["+‖∇φ‖", mid - (bh/2 - 10)], ["0", mid], ["−‖∇φ‖", mid + (bh/2 - 10)]].forEach(([t, yy]) =>
      gPanel.append("text").attr("x", bx + bw + 6).attr("y", yy + 4).attr("font-size", 9.5)
            .attr("fill", P.muted).text(t));
    gPanel.append("text").attr("x", bx).attr("y", by + bh + 22).attr("font-size", 11)
          .attr("fill", P.ink).text("= " + fmt(Du, 3));
    gPanel.append("text").attr("x", bx).attr("y", by + bh + 40).attr("font-size", 11)
          .attr("fill", P.muted).text("ϑ = " + fmt(phi*180/Math.PI, 1) + "°");

    let extra = "";
    if (cur.checked){
      const h = hess(f, px, py), ev = eig2(h[0], h[1], h[2]);
      const kind = ev[0][0] > 1e-6 && ev[1][0] > 1e-6 ? "positive definite → locally a bowl"
                 : ev[0][0] < -1e-6 && ev[1][0] < -1e-6 ? "negative definite → locally a dome"
                 : ev[0][0]*ev[1][0] < -1e-6 ? "indefinite → SADDLE" : "singular / semi-definite";
      extra = ` · H = [[${fmt(h[0],2)}, ${fmt(h[1],2)}], [${fmt(h[1],2)}, ${fmt(h[2],2)}]]`
            + ` · λ = ${fmt(ev[0][0],2)}, ${fmt(ev[1][0],2)} — <b>${kind}</b>`;
    }
    out.innerHTML = `φ(x,y) = ${S.name} · point (<b>${fmt(px,2)}</b>, <b>${fmt(py,2)}</b>) · `
      + `φ = <b>${fmt(f(px,py),3)}</b> · ∇φ = (<b>${fmt(g[0],3)}</b>, <b>${fmt(g[1],3)}</b>), `
      + `‖∇φ‖ = <b>${fmt(gn,3)}</b> · u = (${fmt(u[0],2)}, ${fmt(u[1],2)}) · `
      + `D_u φ = ‖∇φ‖cos ϑ = <b>${fmt(Du,3)}</b>${extra}`;
  }
  sel.addEventListener("change", () => { drawField(); draw(); });
  cur.addEventListener("change", draw);
  drawField(); draw();
})();

/* ═══════════ §13 · forward vs reverse mode AD ════════════════════════════ */
(function(){
  const svg = d3.select("#ad-svg"); if (svg.empty()) return;
  const W = 680, H = 430;
  const NODE = [
    {id:"vm1", x: 70,  y: 90,  t:"v₋₁ = x₁",  short:"x₁"},
    {id:"v0",  x: 70,  y: 320, t:"v₀ = x₂",   short:"x₂"},
    {id:"v1",  x: 250, y: 60,  t:"v₁ = ln v₋₁", short:"ln"},
    {id:"v2",  x: 250, y: 190, t:"v₂ = v₋₁·v₀", short:"×"},
    {id:"v3",  x: 250, y: 340, t:"v₃ = sin v₀", short:"sin"},
    {id:"v4",  x: 435, y: 120, t:"v₄ = v₁ + v₂", short:"+"},
    {id:"v5",  x: 600, y: 220, t:"v₅ = v₄ − v₃", short:"−  = y"}
  ];
  const EDGE = [["vm1","v1"],["vm1","v2"],["v0","v2"],["v0","v3"],["v1","v4"],["v2","v4"],["v4","v5"],["v3","v5"]];
  const NI = {}; NODE.forEach((n, i) => NI[n.id] = i);

  arrowDefs(svg, [["adE", "#4a5470"], ["adH", P.B]]);
  const gE = svg.append("g"), gN = svg.append("g");
  const out  = document.getElementById("ad-readout");
  const mSel = document.getElementById("ad-mode");
  const x1In = document.getElementById("ad-x1"), x2In = document.getElementById("ad-x2");

  const nodeSel = {}, valSel = {}, derSel = {};
  EDGE.forEach(([a, b]) => {
    const A = NODE[NI[a]], B = NODE[NI[b]];
    const dx = B.x - A.x, dy = B.y - A.y, L = Math.hypot(dx, dy);
    gE.append("line").attr("x1", A.x + 46*dx/L).attr("y1", A.y + 22*dy/L)
      .attr("x2", B.x - 50*dx/L).attr("y2", B.y - 24*dy/L)
      .attr("stroke", "#4a5470").attr("stroke-width", 1.6).attr("marker-end", "url(#adE)");
  });
  NODE.forEach(n => {
    const g = gN.append("g");
    nodeSel[n.id] = g.append("rect").attr("x", n.x - 46).attr("y", n.y - 21)
      .attr("width", 92).attr("height", 42).attr("rx", 9)
      .attr("fill", "#1e222d").attr("stroke", P.line).attr("stroke-width", 1.5);
    g.append("text").attr("x", n.x).attr("y", n.y - 3).attr("text-anchor", "middle")
      .attr("font-size", 11.5).attr("fill", P.ink).text(n.t);
    valSel[n.id] = g.append("text").attr("x", n.x).attr("y", n.y + 13).attr("text-anchor", "middle")
      .attr("font-size", 10.5).attr("fill", P.muted).text("");
    derSel[n.id] = g.append("text").attr("x", n.x).attr("y", n.y + 36).attr("text-anchor", "middle")
      .attr("font-size", 11).attr("font-weight", 600).attr("fill", P.B).text("");
  });
  svg.append("text").attr("x", W/2).attr("y", H - 12).attr("text-anchor", "middle")
     .attr("font-size", 11.5).attr("fill", P.muted)
     .text("y = ln(x₁) + x₁·x₂ − sin(x₂)");

  let frames = [], k = 0, timer = null;

  function build(){
    const x1 = +x1In.value / 100, x2 = +x2In.value / 100;
    const v = {};
    v.vm1 = x1; v.v0 = x2;
    v.v1 = Math.log(x1); v.v2 = x1 * x2; v.v3 = Math.sin(x2);
    v.v4 = v.v1 + v.v2; v.v5 = v.v4 - v.v3;
    const mode = mSel.value;
    const F = [];
    const push = (id, der, txt) => F.push({id, der, txt, v});

    if (mode === "rev"){
      F.push({id:null, der:null, txt:"forward pass first — evaluate every vᵢ (values shown in grey).", v});
      push("v5", "v̄₅ = 1", "seed the OUTPUT: ȳ = ∂y/∂y = 1.");
      push("v4", "v̄₄ = v̄₅·(∂v₅/∂v₄) = 1", "v₅ = v₄ − v₃, so ∂v₅/∂v₄ = +1.");
      push("v3", "v̄₃ = v̄₅·(−1) = −1", "v₅ = v₄ − v₃, so ∂v₅/∂v₃ = −1.");
      push("v1", "v̄₁ = v̄₄·1 = 1", "v₄ = v₁ + v₂.");
      push("v2", "v̄₂ = v̄₄·1 = 1", "v₄ = v₁ + v₂ — both parents get the adjoint unchanged.");
      push("v0", "v̄₀ = v̄₂·v₋₁ + v̄₃·cos v₀ = " + fmt(x1 - Math.cos(x2), 4),
           "x₂ feeds TWO nodes (v₂ and v₃) — its adjoint is the SUM over outgoing edges.");
      push("vm1", "v̄₋₁ = v̄₁/v₋₁ + v̄₂·v₀ = " + fmt(1/x1 + x2, 4),
           "x₁ also feeds two nodes. Done: the FULL gradient in one reverse sweep.");
    } else {
      const seed1 = mode === "fwd1" ? 1 : 0, seed2 = mode === "fwd1" ? 0 : 1;
      const which = mode === "fwd1" ? "x₁" : "x₂";
      const d = {};
      d.vm1 = seed1; d.v0 = seed2;
      d.v1 = d.vm1 / x1; d.v2 = d.vm1 * x2 + x1 * d.v0; d.v3 = Math.cos(x2) * d.v0;
      d.v4 = d.v1 + d.v2; d.v5 = d.v4 - d.v3;
      push("vm1", "v̇₋₁ = " + seed1, "seed the INPUT " + which + ": set its tangent to 1, the other to 0.");
      push("v0",  "v̇₀ = " + seed2, "the other input's tangent is 0 for this sweep.");
      push("v1",  "v̇₁ = v̇₋₁/v₋₁ = " + fmt(d.v1, 4), "d(ln u) = du/u.");
      push("v2",  "v̇₂ = v̇₋₁·v₀ + v₋₁·v̇₀ = " + fmt(d.v2, 4), "product rule, carried alongside the value.");
      push("v3",  "v̇₃ = cos(v₀)·v̇₀ = " + fmt(d.v3, 4), "chain rule on sin.");
      push("v4",  "v̇₄ = v̇₁ + v̇₂ = " + fmt(d.v4, 4), "tangents add.");
      push("v5",  "ẏ = v̇₄ − v̇₃ = " + fmt(d.v5, 4),
           "done — this sweep produced ∂y/∂" + which + " ONLY. A second sweep is needed for the other.");
    }
    frames = F; k = 0; render();
  }

  function render(){
    const x1 = +x1In.value / 100, x2 = +x2In.value / 100;
    const v = {vm1:x1, v0:x2, v1:Math.log(x1), v2:x1*x2, v3:Math.sin(x2)};
    v.v4 = v.v1 + v.v2; v.v5 = v.v4 - v.v3;
    NODE.forEach(n => {
      valSel[n.id].text("= " + fmt(v[n.id], 3));
      nodeSel[n.id].attr("stroke", P.line).attr("stroke-width", 1.5).attr("fill", "#1e222d");
      derSel[n.id].text("");
    });
    for (let i = 0; i < k; i++){
      const f = frames[i];
      if (!f.id) continue;
      derSel[f.id].text(f.der);
      nodeSel[f.id].attr("stroke", P.B).attr("stroke-width", 1.6).attr("fill", "#23283a");
    }
    const curF = frames[k - 1];
    if (curF && curF.id)
      nodeSel[curF.id].attr("stroke", P.good).attr("stroke-width", 3).attr("fill", "#1b2a24");

    const mode = mSel.value;
    const sweeps = mode === "rev"
      ? "reverse mode: <b>1 sweep</b> gives ∂y/∂x₁ AND ∂y/∂x₂ (1 output)"
      : "forward mode: <b>2 sweeps</b> needed — one per input";
    const truth = `exact: ∂y/∂x₁ = 1/x₁ + x₂ = <b>${fmt(1/x1 + x2, 4)}</b>, `
                + `∂y/∂x₂ = x₁ − cos x₂ = <b>${fmt(x1 - Math.cos(x2), 4)}</b>`;
    out.innerHTML = `x₁ = <b>${fmt(x1,2)}</b>, x₂ = <b>${fmt(x2,2)}</b>, y = <b>${fmt(v.v5,4)}</b>`
      + ` · step <b>${k}</b>/${frames.length}` + (curF ? ` — ${curF.txt}` : "")
      + `<br>${sweeps} · ${truth}`;
  }

  document.getElementById("ad-step").addEventListener("click", () => {
    if (k < frames.length){ k++; render(); }
  });
  document.getElementById("ad-run").addEventListener("click", () => {
    if (timer){ clearInterval(timer); timer = null; return; }
    timer = setInterval(() => {
      if (k >= frames.length){ clearInterval(timer); timer = null; return; }
      k++; render();
    }, 700);
  });
  document.getElementById("ad-reset").addEventListener("click", () => {
    if (timer){ clearInterval(timer); timer = null; } k = 0; render();
  });
  mSel.addEventListener("change", build);
  x1In.addEventListener("input", build);
  x2In.addEventListener("input", build);
  build();
})();
