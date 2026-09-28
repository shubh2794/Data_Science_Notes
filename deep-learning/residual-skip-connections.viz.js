/* residual-skip-connections.viz.js — the visualizations on
   deep-learning/residual-skip-connections.html.
   Loaded after ../data.js → ../notes.js → dl-viz.js (DC palette, DL.rng,
   DL.randn, DL.fmt). Each block is an IIFE that exits quietly if its
   container is not on the page.

     1  #gr-svg   ‖∂L/∂h_l‖ vs layer, plain vs residual — a real backward pass
     2  #en-svg   the ensemble-of-paths view: C(n,k) and C(n,k)·βᵏ
     3  #bk-svg   block builder: shapes, weights, MACs, shortcut type
     4  #hw-svg   highway carry product ∏(1 − T) vs depth
     5  #sd-svg   stochastic depth: survival probabilities and a sampled pass
     6  #ls-svg   2-D filter-normalised loss slice, plain vs residual          */

/* ---------- small shared helpers (page-local) ---------- */
const RS = (function () {
  function axisStyle(g) {
    g.selectAll("path,line").attr("stroke", DC.line);
    g.selectAll("text").attr("fill", DC.muted).attr("font-size", 10);
    return g;
  }
  function label(svg, x, y, txt, o) {
    o = o || {};
    return svg.append("text").attr("x", x).attr("y", y)
      .attr("fill", o.fill || DC.muted).attr("font-size", o.size || 10)
      .attr("text-anchor", o.anchor || "start").attr("font-weight", o.weight || null)
      .text(txt);
  }
  function fmtInt(x) { return Math.round(x).toLocaleString("en-US"); }
  function fmtBig(x) {
    if (x >= 1e9) return (x / 1e9).toFixed(2) + "G";
    if (x >= 1e6) return (x / 1e6).toFixed(1) + "M";
    if (x >= 1e3) return (x / 1e3).toFixed(1) + "k";
    return String(Math.round(x));
  }
  function bind(id, out, fmt) {
    const el = document.getElementById(id), o = document.getElementById(out);
    const f = () => { if (o) o.textContent = fmt ? fmt(+el.value) : el.value; };
    f(); return f;
  }
  return { axisStyle: axisStyle, label: label, fmtInt: fmtInt, fmtBig: fmtBig, bind: bind };
})();

/* ══════════ 1 · gradient norm vs depth ══════════ */
(function () {
  const svg = d3.select("#gr-svg"); if (svg.empty()) return;
  const W = 680, H = 300, m = { l: 54, r: 150, t: 16, b: 40 };
  const n = 48;
  let seed = 7;
  const phi = {
    relu: { f: x => (x > 0 ? x : 0), d: x => (x > 0 ? 1 : 0), c: 2 },
    tanh: { f: x => Math.tanh(x), d: x => { const t = Math.tanh(x); return 1 - t * t; }, c: 1 }
  };
  function randMat(r, sd) {
    const M = [];
    for (let i = 0; i < n; i++) { const row = new Float64Array(n); for (let j = 0; j < n; j++) row[j] = DL.randn(r) * sd; M.push(row); }
    return M;
  }
  function mv(M, v) { const o = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; const row = M[i]; for (let j = 0; j < n; j++) s += row[j] * v[j]; o[i] = s; } return o; }
  function mtv(M, v) { const o = new Float64Array(n); for (let i = 0; i < n; i++) { const vi = v[i], row = M[i]; if (vi === 0) continue; for (let j = 0; j < n; j++) o[j] += row[j] * vi; } return o; }
  function norm(v) { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s); }
  function map(v, f) { const o = new Float64Array(v.length); for (let i = 0; i < v.length; i++) o[i] = f(v[i]); return o; }
  function mul(a, b) { const o = new Float64Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] * b[i]; return o; }

  function run() {
    const L = +d3.select("#gr-L").property("value");
    const act = d3.select("#gr-act").property("value");
    const g = +d3.select("#gr-g").property("value");
    const wire = d3.select("#gr-wire").property("value");
    const aSel = d3.select("#gr-a").property("value");
    const a = aSel === "isqrt" ? 1 / Math.sqrt(L) : +aSel;
    const P = phi[act], sd = g * Math.sqrt(P.c / n);

    const r = DL.rng(seed);
    const x0 = new Float64Array(n); for (let i = 0; i < n; i++) x0[i] = DL.randn(r);
    const rv = new Float64Array(n); for (let i = 0; i < n; i++) rv[i] = DL.randn(r);
    const rn = norm(rv); for (let i = 0; i < n; i++) rv[i] /= rn;
    const Wp = [], W1 = [], W2 = [];
    for (let l = 0; l < L; l++) { Wp.push(randMat(r, sd)); W1.push(randMat(r, sd)); W2.push(randMat(r, sd)); }

    /* plain */
    const hP = [x0], zP = [];
    for (let l = 0; l < L; l++) { const z = mv(Wp[l], hP[l]); zP.push(z); hP.push(map(z, P.f)); }
    const gP = new Array(L + 1); let d = rv; gP[L] = norm(d);
    for (let l = L - 1; l >= 0; l--) { d = mtv(Wp[l], mul(map(zP[l], P.d), d)); gP[l] = norm(d); }

    /* residual */
    const hR = [x0], st = [];
    for (let l = 0; l < L; l++) {
      const h = hR[l];
      if (wire === "pre") {
        const u = map(h, P.f), z1 = mv(W1[l], u), v = map(z1, P.f), f = mv(W2[l], v);
        const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = h[i] + a * f[i];
        st.push({ h: h, z1: z1 }); hR.push(o);
      } else {
        const z1 = mv(W1[l], h), v = map(z1, P.f), f = mv(W2[l], v);
        const s = new Float64Array(n); for (let i = 0; i < n; i++) s[i] = h[i] + a * f[i];
        st.push({ h: h, z1: z1, s: s }); hR.push(map(s, P.f));
      }
    }
    const gR = new Array(L + 1); d = rv; gR[L] = norm(d);
    for (let l = L - 1; l >= 0; l--) {
      const S = st[l];
      if (wire === "pre") {
        let t = mtv(W2[l], d); t = mul(map(S.z1, P.d), t); t = mtv(W1[l], t); t = mul(map(S.h, P.d), t);
        const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = d[i] + a * t[i]; d = o;
      } else {
        const ds = mul(map(S.s, P.d), d);
        let t = mtv(W2[l], ds); t = mul(map(S.z1, P.d), t); t = mtv(W1[l], t);
        const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = ds[i] + a * t[i]; d = o;
      }
      gR[l] = norm(d);
    }
    const lg = v => (v > 0 ? Math.log10(v) : -30);
    const yP = gP.map(lg), yR = gR.map(lg);
    draw(L, yP, yR, norm(hP[L]) / norm(x0), norm(hR[L]) / norm(x0), a, wire, act, g);
  }

  function draw(L, yP, yR, fP, fR, a, wire, act, g) {
    svg.selectAll("*").remove();
    const all = yP.concat(yR).filter(v => v > -29);
    let lo = Math.min(-1, d3.min(all)), hi = Math.max(1, d3.max(all));
    lo = Math.max(lo, -30); hi = Math.min(hi, 30);
    const x = d3.scaleLinear().domain([0, L]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([lo, hi]).nice().range([H - m.b, m.t]);
    RS.axisStyle(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(8)));
    RS.axisStyle(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6)));
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(0)).attr("y2", y(0))
      .attr("stroke", DC.muted).attr("stroke-dasharray", "3,4").attr("opacity", .6);
    RS.label(svg, (m.l + W - m.r) / 2, H - 8, "layer index l  (0 = input, L = output where the gradient starts)", { anchor: "middle" });
    svg.append("text").attr("transform", `translate(14,${(H - m.b + m.t) / 2}) rotate(-90)`).attr("text-anchor", "middle")
      .attr("fill", DC.muted).attr("font-size", 10).text("log₁₀ ‖∂L/∂hₗ‖");
    const ln = d3.line().x((d, i) => x(i)).y(d => y(Math.max(lo, d)));
    svg.append("path").datum(yP).attr("d", ln).attr("fill", "none").attr("stroke", DC.bad).attr("stroke-width", 2);
    svg.append("path").datum(yR).attr("d", ln).attr("fill", "none").attr("stroke", DC.good).attr("stroke-width", 2);
    const lx = W - m.r + 12;
    svg.append("rect").attr("x", lx).attr("y", 30).attr("width", 14).attr("height", 3).attr("fill", DC.bad);
    RS.label(svg, lx + 20, 34, "plain stack", { fill: DC.ink });
    svg.append("rect").attr("x", lx).attr("y", 52).attr("width", 14).attr("height", 3).attr("fill", DC.good);
    RS.label(svg, lx + 20, 56, "residual stack", { fill: DC.ink });
    RS.label(svg, lx, 84, `${act}, g = ${g.toFixed(2)}`);
    RS.label(svg, lx, 100, `${wire === "pre" ? "x + a·F(x)" : "φ(x + a·F(x))"}`);
    RS.label(svg, lx, 116, `a = ${a.toFixed(3)}`);
    RS.label(svg, lx, 140, "forward ‖h_L‖/‖x₀‖:");
    RS.label(svg, lx, 156, `plain ${fP.toExponential(1)}`, { fill: DC.bad });
    RS.label(svg, lx, 172, `resid ${fR.toExponential(1)}`, { fill: DC.good });
    const rP = Math.pow(10, yP[0]), rR = Math.pow(10, yR[0]);
    const verdict = v => (v < 0.01 ? "vanished" : v > 100 ? "exploded" : "preserved");
    d3.select("#gr-read").html(
      `gradient reaching the input (relative to the output, ‖∂L/∂x_L‖ = 1): ` +
      `<b style="color:${DC.bad}">plain ${rP.toExponential(2)}</b> (${verdict(rP)}) · ` +
      `<b style="color:${DC.good}">residual ${rR.toExponential(2)}</b> (${verdict(rR)}). ` +
      (a === 0 ? "With a = 0 every block is exactly the identity at init: the gradient is carried unchanged (post-activation still passes through φ′)." :
        rR > 100 ? "a = 1 with no normalisation: each block roughly doubles the variance, so the residual gradient GROWS toward the input — the reason for BatchNorm, 1/√L scaling or zero-init." :
          wire === "post" && act === "tanh" ? "The post-addition tanh multiplies the highway by tanh′ < 1 at every block, so even the residual gradient decays — the reason pre-activation moved it off the main path." :
            "The identity term keeps the residual gradient within a narrow band; the plain product drifts geometrically."));
  }

  ["#gr-L", "#gr-g"].forEach(id => d3.select(id).on("input", () => { upd(); run(); }));
  ["#gr-act", "#gr-wire", "#gr-a"].forEach(id => d3.select(id).on("change", run));
  d3.select("#gr-seed").on("click", () => { seed = (seed * 7919 + 13) % 100003; run(); });
  const u1 = RS.bind("gr-L", "gr-Lv"), u2 = RS.bind("gr-g", "gr-gv", v => v.toFixed(2));
  function upd() { u1(); u2(); }
  run();
})();

/* ══════════ 2 · ensemble of paths ══════════ */
(function () {
  const svg = d3.select("#en-svg"); if (svg.empty()) return;
  const W = 680, H = 290, m = { l: 48, r: 16, t: 20, b: 40 };
  function binom(n) {             // C(n,k) as doubles, row of Pascal's triangle
    const row = [1];
    for (let i = 1; i <= n; i++) { for (let k = i; k >= 1; k--) row[k] = (row[k] || 0) + row[k - 1]; }
    return row;
  }
  function draw() {
    const n = +d3.select("#en-n").property("value");
    const b = +d3.select("#en-b").property("value");
    const del = Math.min(+d3.select("#en-del").property("value"), n - 1);
    const C = binom(n), tot = Math.pow(2, n);
    const share = C.map(c => c / tot);
    const q = b / (1 + b);
    const grad = C.map((c, k) => c * Math.pow(q, k) * Math.pow(1 - q, n - k));
    svg.selectAll("*").remove();
    const x = d3.scaleBand().domain(d3.range(n + 1)).range([m.l, W - m.r]).padding(0.12);
    const ymax = Math.max(d3.max(share), d3.max(grad)) * 1.08;
    const y = d3.scaleLinear().domain([0, ymax]).range([H - m.b, m.t]);
    /* 90% central band of the gradient distribution */
    let cum = 0, k5 = 0, k95 = n;
    for (let k = 0; k <= n; k++) { cum += grad[k]; if (cum < 0.05) k5 = k + 1; if (cum >= 0.95) { k95 = k; break; } }
    svg.append("rect").attr("x", x(k5)).attr("y", m.t).attr("width", x(k95) + x.bandwidth() - x(k5)).attr("height", H - m.b - m.t)
      .attr("fill", DC.a2).attr("opacity", 0.08);
    svg.selectAll("rect.bar").data(share).enter().append("rect").attr("class", "bar")
      .attr("x", (d, k) => x(k)).attr("width", x.bandwidth()).attr("y", d => y(d)).attr("height", d => y(0) - y(d))
      .attr("fill", DC.accent).attr("opacity", 0.55);
    const ln = d3.line().x((d, k) => x(k) + x.bandwidth() / 2).y(d => y(d));
    svg.append("path").datum(grad).attr("d", ln).attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 2.2);
    const every = n > 40 ? 10 : n > 20 ? 5 : n > 10 ? 2 : 1;
    RS.axisStyle(svg.append("g").attr("transform", `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).tickValues(d3.range(0, n + 1, every))));
    RS.axisStyle(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%"))));
    RS.label(svg, (m.l + W - m.r) / 2, H - 6, "path length k = number of residual branches traversed", { anchor: "middle" });
    RS.label(svg, W - m.r - 4, m.t + 10, "■ share of paths  C(n,k)/2ⁿ", { anchor: "end", fill: DC.accent });
    RS.label(svg, W - m.r - 4, m.t + 26, "— share of gradient  ∝ C(n,k)·βᵏ", { anchor: "end", fill: DC.a2 });
    RS.label(svg, W - m.r - 4, m.t + 42, `shaded: 90% of gradient, k = ${k5}…${k95}`, { anchor: "end" });
    const mean = n * q, sd = Math.sqrt(n * q * (1 - q));
    const lostPaths = 1 - Math.pow(0.5, del), lostGrad = 1 - Math.pow(1 - q, del);
    d3.select("#en-read").html(
      `<b>${n}</b> blocks → <b>2<sup>${n}</sup> ≈ ${tot.toExponential(2)}</b> paths, mean length <b>${(n / 2).toFixed(1)}</b>. ` +
      `Gradient-weighted length: mean <b>${mean.toFixed(1)}</b>, sd ${sd.toFixed(1)} — effective depth is ${(100 * mean / n).toFixed(0)}% of nominal. ` +
      (del > 0 ? `Deleting <b>${del}</b> block${del > 1 ? "s" : ""} removes ${(100 * lostPaths).toFixed(1)}% of paths but only <b>${(100 * lostGrad).toFixed(1)}%</b> of the gradient-weighted mass (β/(1+β) = ${(100 * q).toFixed(1)}% per block); a plain net would lose its only path.`
        : "Try deleting blocks: a residual net degrades gracefully.")
    );
  }
  const u = [RS.bind("en-n", "en-nv"), RS.bind("en-b", "en-bv", v => v.toFixed(2)), RS.bind("en-del", "en-delv")];
  ["#en-n", "#en-b", "#en-del"].forEach(id => d3.select(id).on("input", () => { u.forEach(f => f()); draw(); }));
  draw();
})();

/* ══════════ 3 · block builder ══════════ */
(function () {
  const svg = d3.select("#bk-svg"); if (svg.empty()) return;
  const W = 680, H = 380;
  const PRESETS = {
    r34: { type: "basic", cin: 64, w: 128, s: 2, g: 1, h: 56 },
    r50: { type: "bottleneck", cin: 64, w: 64, s: 1, g: 1, h: 56 },
    r50b: { type: "bottleneck", cin: 256, w: 64, s: 1, g: 1, h: 56 },
    rx50: { type: "resnext", cin: 256, w: 128, s: 1, g: 32, h: 56 },
    d121: { type: "dense", cin: 128, w: 32, s: 1, g: 1, h: 56 }
  };
  const val = id => d3.select(id).property("value");
  function conv(k, cin, cout, s, hin, groups, name) {
    groups = groups || 1;
    const hout = Math.ceil(hin / s);
    const params = k * k * (cin / groups) * cout;
    return { k: k, cin: cin, cout: cout, s: s, hin: hin, hout: hout, groups: groups, params: params, bn: 2 * cout, macs: params * hout * hout, name: name };
  }
  function build() {
    const type = val("#bk-type"), cin = +val("#bk-cin"), w = +val("#bk-w"), s = +val("#bk-s"), h = +val("#bk-h");
    let g = +val("#bk-g");
    let layers = [], out, short = null, merge = "add", note = "";
    if (type === "basic") {
      layers = [conv(3, cin, w, s, h, 1, "3×3 conv"), conv(3, w, w, 1, Math.ceil(h / s), 1, "3×3 conv")];
      out = w;
    } else if (type === "bottleneck") {
      layers = [conv(1, cin, w, 1, h, 1, "1×1 reduce"), conv(3, w, w, s, h, 1, "3×3 conv"), conv(1, w, 4 * w, 1, Math.ceil(h / s), 1, "1×1 expand")];
      out = 4 * w;
    } else if (type === "resnext") {
      g = Math.min(g, w); while (w % g) g--;
      layers = [conv(1, cin, w, 1, h, 1, "1×1 reduce"), conv(3, w, w, s, h, g, `3×3 grouped (g=${g})`), conv(1, w, 2 * w, 1, Math.ceil(h / s), 1, "1×1 expand")];
      out = 2 * w; note = `${g}×${w / g}d`;
    } else {
      const k = w;
      layers = [conv(1, cin, 4 * k, 1, h, 1, "BN-ReLU-1×1"), conv(3, 4 * k, k, 1, h, 1, "BN-ReLU-3×3")];
      layers[0].bn = 2 * cin; layers[1].bn = 2 * 4 * k;
      out = cin + k; merge = "concat"; note = `growth k = ${k}`;
    }
    const sEff = type === "dense" ? 1 : s;
    const hout = Math.ceil(h / sEff);
    if (merge === "add" && (cin !== out || sEff !== 1)) short = conv(1, cin, out, sEff, h, 1, "1×1 projection");
    return { type: type, cin: cin, h: h, hout: hout, layers: layers, out: out, short: short, merge: merge, note: note, s: sEff };
  }
  function draw() {
    const B = build();
    svg.selectAll("*").remove();
    const cx = 230, bw = 250, bh = 36;
    const rows = B.layers.length;
    const gap = rows === 3 ? 62 : 80;
    const yIn = 14, y0 = yIn + 58;
    const shapeTxt = (hh, c) => `${hh}×${hh}×${c}`;
    // input
    svg.append("rect").attr("x", cx - bw / 2).attr("y", yIn).attr("width", bw).attr("height", 28).attr("rx", 6)
      .attr("fill", DC.panel2).attr("stroke", DC.muted);
    RS.label(svg, cx, yIn + 18, `input x  ·  ${shapeTxt(B.h, B.cin)}`, { anchor: "middle", fill: DC.ink, size: 11.5 });
    let prevY = yIn + 28;
    B.layers.forEach((L, i) => {
      const y = y0 + i * gap;
      svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", prevY).attr("y2", y).attr("stroke", DC.muted);
      svg.append("rect").attr("x", cx - bw / 2).attr("y", y).attr("width", bw).attr("height", bh).attr("rx", 6)
        .attr("fill", "rgba(91,156,255,.12)").attr("stroke", DC.accent);
      RS.label(svg, cx, y + 15, `${L.name}${L.s > 1 ? ", stride 2" : ""}`, { anchor: "middle", fill: DC.ink, size: 11.5 });
      RS.label(svg, cx, y + 29, `${L.cin} → ${L.cout} ch   ·   out ${shapeTxt(L.hout, L.cout)}`, { anchor: "middle", size: 10 });
      RS.label(svg, cx - bw / 2 - 8, y + 15, RS.fmtInt(L.params) + " w", { anchor: "end", fill: DC.ink, size: 10.5 });
      RS.label(svg, cx - bw / 2 - 8, y + 29, RS.fmtBig(L.macs) + " MACs", { anchor: "end", size: 10 });
      prevY = y + bh;
    });
    const yPlus = y0 + (rows - 1) * gap + bh + 30;
    svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", prevY).attr("y2", yPlus - 13).attr("stroke", DC.muted);
    svg.append("circle").attr("cx", cx).attr("cy", yPlus).attr("r", 13).attr("fill", DC.panel2).attr("stroke", DC.a2);
    RS.label(svg, cx, yPlus + 4, B.merge === "add" ? "+" : "‖", { anchor: "middle", fill: DC.a2, size: 15, weight: 700 });
    RS.label(svg, cx - 20, yPlus + 4, B.merge === "add" ? "add" : "concat", { anchor: "end", fill: DC.a2 });
    // shortcut path on the right
    const sx = cx + bw / 2 + 70;
    const shortCol = B.short ? DC.bad : DC.good;
    svg.append("path").attr("d", `M${cx + bw / 2},${yIn + 14} H${sx} V${yPlus} H${cx + 13}`)
      .attr("fill", "none").attr("stroke", shortCol).attr("stroke-width", 2).attr("stroke-dasharray", B.short ? "6,4" : null);
    const midY = (yIn + yPlus) / 2;
    if (B.merge === "concat") {
      RS.label(svg, sx + 10, midY - 6, "x passed through unchanged", { fill: DC.good, size: 11 });
      RS.label(svg, sx + 10, midY + 10, "(concatenated, not added)", { size: 10 });
    } else if (B.short) {
      svg.append("rect").attr("x", sx - 58).attr("y", midY - 20).attr("width", 116).attr("height", 40).attr("rx", 6)
        .attr("fill", DC.panel2).attr("stroke", DC.bad);
      RS.label(svg, sx, midY - 5, `1×1 proj${B.s > 1 ? ", s=2" : ""}`, { anchor: "middle", fill: DC.ink, size: 11 });
      RS.label(svg, sx, midY + 11, `${RS.fmtInt(B.short.params)} w`, { anchor: "middle", size: 10 });
      RS.label(svg, sx + 64, midY + 4, `${B.cin}→${B.out} ch`, { size: 10 });
    } else {
      RS.label(svg, sx + 10, midY, "identity shortcut (0 weights)", { fill: DC.good, size: 11 });
    }
    // output
    const yOut = yPlus + 26;
    svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", yPlus + 13).attr("y2", yOut).attr("stroke", DC.muted);
    svg.append("rect").attr("x", cx - bw / 2).attr("y", yOut).attr("width", bw).attr("height", 28).attr("rx", 6)
      .attr("fill", DC.panel2).attr("stroke", DC.muted);
    RS.label(svg, cx, yOut + 18, `output · ${shapeTxt(B.hout, B.out)}`, { anchor: "middle", fill: DC.ink, size: 11.5 });

    const wBranch = d3.sum(B.layers, L => L.params), bn = d3.sum(B.layers, L => L.bn) + (B.short ? B.short.bn : 0);
    const wShort = B.short ? B.short.params : 0;
    const macs = d3.sum(B.layers, L => L.macs) + (B.short ? B.short.macs : 0);
    const basicEq = 2 * 9 * B.out * B.out;
    d3.select("#bk-read").html(
      `branch weights <b>${RS.fmtInt(wBranch)}</b> · shortcut <b>${RS.fmtInt(wShort)}</b>${B.short ? " (projection — shapes differ)" : " (identity)"} · BN params ${RS.fmtInt(bn)} · ` +
      `total <b>${RS.fmtInt(wBranch + wShort + bn)}</b> · <b>${RS.fmtBig(macs)}</b> MACs at ${B.h}×${B.h}` +
      (B.note ? ` · ${B.note}` : "") +
      (B.type !== "basic" && B.type !== "dense" ? `. A basic block at the same ${B.out}-channel width would need ${RS.fmtInt(basicEq)} weights (${(basicEq / wBranch).toFixed(1)}×).` : ".") +
      (B.type === "dense" ? ` Width grows ${B.cin} → ${B.out}; a transition layer later compresses it.` : "")
    );
  }
  ["#bk-type", "#bk-cin", "#bk-w", "#bk-s", "#bk-g", "#bk-h"].forEach(id => d3.select(id).on("change", draw));
  d3.selectAll("#viz-blk button[data-preset]").on("click", function () {
    const p = PRESETS[this.getAttribute("data-preset")];
    d3.select("#bk-type").property("value", p.type); d3.select("#bk-cin").property("value", String(p.cin));
    d3.select("#bk-w").property("value", String(p.w)); d3.select("#bk-s").property("value", String(p.s));
    d3.select("#bk-g").property("value", String(p.g)); d3.select("#bk-h").property("value", String(p.h));
    draw();
  });
  draw();
})();

/* ══════════ 4 · highway carry product ══════════ */
(function () {
  const svg = d3.select("#hw-svg"); if (svg.empty()) return;
  const W = 680, H = 280, m = { l: 54, r: 110, t: 16, b: 40 }, U = 64;
  const sig = z => 1 / (1 + Math.exp(-z));
  function curve(b, s, L, seed) {
    const r = DL.rng(seed), c = new Float64Array(U).fill(1), out = [0];
    for (let l = 1; l <= L; l++) {
      let mean = 0;
      for (let j = 0; j < U; j++) { c[j] *= 1 - sig(b + s * DL.randn(r)); mean += c[j]; }
      out.push(Math.log10(Math.max(mean / U, 1e-30)));
    }
    return out;
  }
  function draw() {
    const b = +d3.select("#hw-b").property("value"), s = +d3.select("#hw-s").property("value"), L = +d3.select("#hw-L").property("value");
    svg.selectAll("*").remove();
    const refs = [-4, -3, -1, 0].filter(v => Math.abs(v - b) > 0.05).map(v => ({ b: v, y: curve(v, s, L, 11) }));
    const main = curve(b, s, L, 11);
    const lo = Math.max(-12, Math.min(-2, d3.min(main), d3.min(refs, r => d3.min(r.y))));
    const x = d3.scaleLinear().domain([0, L]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([lo, 0.5]).range([H - m.b, m.t]);
    RS.axisStyle(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(8)));
    RS.axisStyle(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6)));
    svg.append("text").attr("transform", `translate(14,${(H - m.b + m.t) / 2}) rotate(-90)`).attr("text-anchor", "middle")
      .attr("fill", DC.muted).attr("font-size", 10).text("log₁₀ mean ∏(1 − T)");
    RS.label(svg, (m.l + W - m.r) / 2, H - 6, "number of highway layers the gradient passes back through", { anchor: "middle" });
    const ln = d3.line().x((d, i) => x(i)).y(d => y(Math.max(lo, d)));
    svg.append("line").attr("x1", x(0)).attr("x2", x(L)).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", DC.good).attr("stroke-width", 2);
    RS.label(svg, W - m.r + 6, y(0) + 4, "residual (= 1)", { fill: DC.good });
    refs.forEach(rf => {
      svg.append("path").datum(rf.y).attr("d", ln).attr("fill", "none").attr("stroke", DC.muted).attr("stroke-width", 1).attr("opacity", .5);
      const last = rf.y[rf.y.length - 1];
      if (last > lo + 0.3) RS.label(svg, W - m.r + 6, y(last) + 4, `b = ${rf.b}`, { size: 9 });
    });
    svg.append("path").datum(main).attr("d", ln).attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 2.4);
    const lastM = main[main.length - 1];
    RS.label(svg, W - m.r + 6, y(Math.max(lo, lastM)) - 6, `b = ${b.toFixed(1)}`, { fill: DC.a2 });
    const Tm = sig(b), carry = 1 - Tm;
    const L1 = carry < 1 ? Math.log(0.01) / Math.log(carry) : Infinity;
    d3.select("#hw-read").html(
      `mean gate σ(b_T) = <b>${Tm.toFixed(3)}</b>, carry factor per layer 1 − T ≈ <b>${carry.toFixed(3)}</b>; ` +
      `after ${L} layers the carry path delivers <b>${Math.pow(10, lastM).toExponential(2)}</b> of the gradient ` +
      `(with s = 0 it would fall below 1% after ${isFinite(L1) ? Math.round(L1) : "∞"} layers). ` +
      `A residual shortcut delivers exactly 1 at every depth — it is the highway with the carry gate welded open.`);
  }
  const u = [RS.bind("hw-b", "hw-bv", v => v.toFixed(1).replace("-", "−")), RS.bind("hw-s", "hw-sv", v => v.toFixed(1)), RS.bind("hw-L", "hw-Lv")];
  ["#hw-b", "#hw-s", "#hw-L"].forEach(id => d3.select(id).on("input", () => { u.forEach(f => f()); draw(); }));
  draw();
})();

/* ══════════ 5 · stochastic depth ══════════ */
(function () {
  const svg = d3.select("#sd-svg"); if (svg.empty()) return;
  const W = 680, H = 240, m = { l: 44, r: 16, t: 28, b: 40 };
  let mask = null, sampleSeed = 1;
  function probs() {
    const L = +d3.select("#sd-L").property("value"), pL = +d3.select("#sd-p").property("value"), rule = d3.select("#sd-rule").property("value");
    return d3.range(1, L + 1).map(l => rule === "linear" ? 1 - (l / L) * (1 - pL) : pL);
  }
  function draw() {
    const p = probs(), L = p.length;
    if (mask && mask.length !== L) mask = null;
    svg.selectAll("*").remove();
    const x = d3.scaleBand().domain(d3.range(L)).range([m.l, W - m.r]).padding(0.15);
    const y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
    RS.axisStyle(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)));
    const every = L > 50 ? 10 : L > 20 ? 5 : 1;
    RS.axisStyle(svg.append("g").attr("transform", `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).tickValues(d3.range(0, L, every)).tickFormat(i => i + 1)));
    svg.selectAll("rect.p").data(p).enter().append("rect").attr("class", "p")
      .attr("x", (d, i) => x(i)).attr("width", x.bandwidth()).attr("y", d => y(d)).attr("height", d => y(0) - y(d))
      .attr("fill", (d, i) => mask ? (mask[i] ? DC.good : DC.bad) : DC.accent)
      .attr("opacity", (d, i) => mask ? (mask[i] ? 0.85 : 0.35) : 0.7);
    RS.label(svg, (m.l + W - m.r) / 2, H - 6, "residual block l (input side → output side)", { anchor: "middle" });
    RS.label(svg, m.l, 12, "survival probability p_l", { size: 10 });
    const E = d3.sum(p);
    let extra = "";
    if (mask) {
      const kept = d3.sum(mask);
      extra = ` · this sampled pass keeps <b style="color:${DC.good}">${kept}</b> blocks and skips <b style="color:${DC.bad}">${L - kept}</b> (dropped blocks are pure identity for this minibatch)`;
    }
    d3.select("#sd-read").html(`expected active blocks <b>${E.toFixed(2)}</b> of ${L} → about <b>${(100 * (1 - E / L)).toFixed(0)}%</b> less branch compute per step` +
      ` · at test time every block runs, branch scaled by p_l${extra}`);
  }
  function sample() {
    const p = probs(), r = DL.rng(sampleSeed++ * 2654435761);
    mask = p.map(pp => r() < pp ? 1 : 0); draw();
  }
  const u = [RS.bind("sd-L", "sd-Lv"), RS.bind("sd-p", "sd-pv", v => v.toFixed(2))];
  ["#sd-L", "#sd-p"].forEach(id => d3.select(id).on("input", () => { u.forEach(f => f()); mask = null; draw(); }));
  d3.select("#sd-rule").on("change", () => { mask = null; draw(); });
  d3.select("#sd-sample").on("click", sample);
  draw();
})();

/* ══════════ 6 · loss-landscape slice ══════════ */
(function () {
  const svg = d3.select("#ls-svg"); if (svg.empty()) return;
  const W = 680, H = 330, G = 25, N = 32, DIN = 4, WD = 10;
  let dirSeed = 3;
  const acts = {
    relu: { f: x => (x > 0 ? x : 0), c: 2 },
    tanh: { f: x => Math.tanh(x), c: 1 }
  };
  function randM(r, rows, cols, sd) { const M = []; for (let i = 0; i < rows; i++) { const a = new Float64Array(cols); for (let j = 0; j < cols; j++) a[j] = DL.randn(r) * sd; M.push(a); } return M; }
  function filterNorm(D, T) {        // per row (neuron): scale direction row to the weight row's norm
    return D.map((row, i) => {
      let dn = 0, tn = 0; for (let j = 0; j < row.length; j++) { dn += row[j] * row[j]; tn += T[i][j] * T[i][j]; }
      const s = Math.sqrt(tn) / (Math.sqrt(dn) || 1); return row.map(v => v * s);
    });
  }
  function combo(T, D1, D2, a, b) { return T.map((row, i) => row.map((v, j) => v + a * D1[i][j] + b * D2[i][j])); }
  function forward(mats, X, act, resid, D) {
    const f = acts[act].f, sc = 1 / Math.sqrt(D);
    let loss = 0;
    for (let s = 0; s < X.x.length; s++) {
      const x = X.x[s];
      let h = new Float64Array(WD);
      const W0 = mats[0];
      for (let i = 0; i < WD; i++) { let z = 0; for (let j = 0; j < DIN; j++) z += W0[i][j] * x[j]; h[i] = f(z); }
      for (let l = 1; l <= D; l++) {
        const Wl = mats[l], o = new Float64Array(WD);
        if (resid) {
          const u = h.map(f);
          for (let i = 0; i < WD; i++) { let z = 0; for (let j = 0; j < WD; j++) z += Wl[i][j] * u[j]; o[i] = h[i] + sc * z; }
        } else {
          for (let i = 0; i < WD; i++) { let z = 0; for (let j = 0; j < WD; j++) z += Wl[i][j] * h[j]; o[i] = f(z); }
        }
        h = o;
      }
      const Wo = mats[D + 1][0]; let yh = 0; for (let j = 0; j < WD; j++) yh += Wo[j] * h[j];
      const e = yh - X.y[s]; loss += e * e;
    }
    return loss / X.x.length;
  }
  function surface(resid, D, act, rad) {
    const r = DL.rng(101), c = acts[act].c;
    const T = [randM(r, WD, DIN, Math.sqrt(c / DIN))];
    for (let l = 1; l <= D; l++) T.push(randM(r, WD, WD, Math.sqrt(c / WD)));
    T.push(randM(r, 1, WD, Math.sqrt(1 / WD)));
    const rd = DL.rng(dirSeed * 977 + (resid ? 0 : 0));
    const D1 = T.map(M => filterNorm(randM(rd, M.length, M[0].length, 1), M));
    const D2 = T.map(M => filterNorm(randM(rd, M.length, M[0].length, 1), M));
    const rx = DL.rng(55), X = { x: [], y: [] };
    for (let s = 0; s < N; s++) { const x = []; for (let j = 0; j < DIN; j++) x.push(DL.randn(rx)); X.x.push(x); X.y.push(Math.tanh(1.2 * x[0] - 0.8 * x[1] + 0.5 * x[2] * x[3])); }
    const Z = [];
    for (let i = 0; i < G; i++) {
      const row = [];
      for (let j = 0; j < G; j++) {
        const a = -rad + 2 * rad * j / (G - 1), b = rad - 2 * rad * i / (G - 1);
        const mats = T.map((M, k) => combo(M, D1[k], D2[k], a, b));
        const L = forward(mats, X, act, resid, D);
        row.push(Math.log10(Math.max(L, 1e-12)));
      }
      Z.push(row);
    }
    return Z;
  }
  function rough(Z) {
    let s = 0, c = 0;
    for (let i = 1; i < G - 1; i++) for (let j = 1; j < G - 1; j++) {
      s += Math.abs(Z[i][j - 1] - 2 * Z[i][j] + Z[i][j + 1]) + Math.abs(Z[i - 1][j] - 2 * Z[i][j] + Z[i + 1][j]); c += 2;
    }
    return s / c;
  }
  function panel(Z, x0, title, col) {
    const size = 250, cell = size / G, y0 = 40;
    const flat = [].concat.apply([], Z).filter(isFinite);
    const lo = d3.min(flat), hi = d3.max(flat);
    const color = d3.scaleSequential(d3.interpolateViridis).domain([lo, hi]);
    const g = svg.append("g");
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      g.append("rect").attr("x", x0 + j * cell).attr("y", y0 + i * cell).attr("width", cell + 0.4).attr("height", cell + 0.4)
        .attr("fill", isFinite(Z[i][j]) ? color(Z[i][j]) : DC.bad);
    }
    svg.append("rect").attr("x", x0).attr("y", y0).attr("width", size).attr("height", size).attr("fill", "none").attr("stroke", DC.line);
    svg.append("circle").attr("cx", x0 + size / 2).attr("cy", y0 + size / 2).attr("r", 3.5).attr("fill", "none").attr("stroke", DC.ink);
    RS.label(svg, x0 + size / 2, 28, title, { anchor: "middle", fill: col, size: 12, weight: 600 });
    RS.label(svg, x0 + size / 2, y0 + size + 16, `log₁₀ loss ${lo.toFixed(2)} … ${hi.toFixed(2)}  (range ${(hi - lo).toFixed(1)} decades)`, { anchor: "middle", size: 10 });
    return { lo: lo, hi: hi };
  }
  function draw() {
    const D = +d3.select("#ls-D").property("value"), act = d3.select("#ls-act").property("value"), rad = +d3.select("#ls-r").property("value");
    svg.selectAll("*").remove();
    const Zp = surface(false, D, act, rad), Zr = surface(true, D, act, rad);
    const a = panel(Zp, 50, `plain, ${D} hidden layers`, DC.bad);
    const b = panel(Zr, 380, `residual, ${D} blocks (branch ×1/√${D})`, DC.good);
    RS.label(svg, W / 2, H - 4, "α (horizontal) and β (vertical) ∈ [−r, r] along two filter-normalised random directions · ○ = the initial weights", { anchor: "middle", size: 10 });
    const rp = rough(Zp), rr = rough(Zr);
    d3.select("#ls-read").html(
      `roughness (mean |second difference| of log₁₀ loss): <b style="color:${DC.bad}">plain ${rp.toFixed(3)}</b> vs <b style="color:${DC.good}">residual ${rr.toFixed(3)}</b>` +
      ` · dynamic range ${(a.hi - a.lo).toFixed(1)} vs ${(b.hi - b.lo).toFixed(1)} decades. ` +
      (D <= 3 ? "At this depth the two are similar — the difference is a depth effect." : "Increase the depth: the plain slice sharpens into steep walls while the residual one stays a gentle bowl."));
  }
  const u = [RS.bind("ls-D", "ls-Dv"), RS.bind("ls-r", "ls-rv", v => v.toFixed(1))];
  ["#ls-D", "#ls-r"].forEach(id => { d3.select(id).on("input", () => u.forEach(f => f())); d3.select(id).on("change", draw); });
  d3.select("#ls-act").on("change", draw);
  d3.select("#ls-seed").on("click", () => { dirSeed++; draw(); });
  draw();
})();
