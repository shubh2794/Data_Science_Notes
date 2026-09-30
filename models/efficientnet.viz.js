/* efficientnet.viz.js — figures for models/efficientnet.html.
   Loaded after data.js / notes.js (palette C). Every cost a figure displays is computed here
   from the block specification of the reference implementation; accuracies are the paper's
   reported numbers (or, where marked, values read off its plots). Each figure is wrapped in
   fig(), so one failure cannot blank the figures after it. */

/* ───────────────────────── palette & helpers ───────────────────────── */
const PAL = Object.assign(
  { A:"#5b9cff", B:"#ffb454", good:"#4ade80", bad:"#f87171", ink:"#e6e9ef", muted:"#9aa3b2", line:"#2a2f3a" },
  (typeof C !== "undefined" ? C : {}));
const XC = { pu:"#c084fc", te:"#2dd4bf", pk:"#f472b6", panel:"#15181f", grid:"#232834", dim:"#5d6675" };
const AXC = { d: PAL.A, w: PAL.B, r: XC.te, c: PAL.good };   // depth, width, resolution, compound

function fig(name, fn){
  try { fn(); }
  catch (e) {
    if (typeof console !== "undefined") console.error("efficientnet figure '" + name + "' failed:", e);
    const r = document.getElementById(name + "-read");
    if (r) r.textContent = "figure failed to render: " + (e && e.message);
  }
}
const $v = id => { const el = document.getElementById(id); return el ? el.value : null; };
function bind(ids, fn){
  ids.forEach(id => {
    const el = document.getElementById(id); if (!el) return;
    const ev = (el.tagName === "SELECT") ? "change" : "input";
    el.addEventListener(ev, fn);
  });
}
function frame(sel, W, H, m){
  const svg = d3.select(sel); svg.selectAll("*").remove();
  const g = svg.append("g").attr("transform", "translate(" + m.l + "," + m.t + ")");
  return { svg: svg, g: g, w: W - m.l - m.r, h: H - m.t - m.b };
}
function axes(f, x, y, xl, yl, xt, yt){
  const ax = d3.axisBottom(x); if (xt) xt(ax);
  const ay = d3.axisLeft(y);   if (yt) yt(ay);
  f.g.append("g").attr("class", "axis").attr("transform", "translate(0," + f.h + ")").call(ax);
  f.g.append("g").attr("class", "axis").call(ay);
  if (xl) f.g.append("text").attr("x", f.w / 2).attr("y", f.h + 32).attr("text-anchor", "middle")
    .attr("fill", PAL.muted).attr("font-size", 11).text(xl);
  if (yl) f.g.append("text").attr("transform", "rotate(-90)").attr("x", -f.h / 2).attr("y", -40)
    .attr("text-anchor", "middle").attr("fill", PAL.muted).attr("font-size", 11).text(yl);
}
function gridlines(f, y, n){
  f.g.append("g").selectAll("line").data(y.ticks(n || 5)).enter().append("line")
    .attr("x1", 0).attr("x2", f.w).attr("y1", d => y(d)).attr("y2", d => y(d))
    .attr("stroke", XC.grid).attr("stroke-width", 1);
}
function txt(g, x, y, s, o){
  o = o || {};
  return g.append("text").attr("x", x).attr("y", y).attr("fill", o.fill || PAL.muted)
    .attr("font-size", o.size || 10.5).attr("text-anchor", o.anchor || "start")
    .attr("font-family", o.mono ? "SF Mono,Menlo,monospace" : null)
    .attr("font-weight", o.bold ? 700 : null).text(s);
}
function legend(g, x, y, items){
  items.forEach((it, i) => {
    g.append("rect").attr("x", x).attr("y", y + i * 15 - 8).attr("width", 10).attr("height", 10).attr("rx", 2).attr("fill", it[1]);
    txt(g, x + 15, y + i * 15, it[0], { fill: PAL.ink, size: 10.5 });
  });
}
const f2 = x => (+x).toFixed(2), f1 = x => (+x).toFixed(1), f3 = x => (+x).toFixed(3);

/* ───────────────────────── shared model: the reference spec, counted ─────────────────────────
   Block strings of efficientnet_builder.py (r=repeats k=kernel s=stride e=expansion i=in o=out),
   round_filters / round_repeats of efficientnet_model.py, SE squeeze = floor(0.25 · block input). */
const EN = (function(){
  const BLOCKS = [
    {r:1,k:3,s:1,e:1,i:32, o:16 }, {r:2,k:3,s:2,e:6,i:16, o:24 }, {r:2,k:5,s:2,e:6,i:24, o:40 },
    {r:3,k:3,s:2,e:6,i:40, o:80 }, {r:3,k:5,s:1,e:6,i:80, o:112}, {r:4,k:5,s:2,e:6,i:112,o:192},
    {r:1,k:3,s:1,e:6,i:192,o:320}
  ];
  const RELEASED = [   // [name, width, depth, resolution, dropout] — the released params dict
    ["B0",1.0,1.0,224,0.2],["B1",1.0,1.1,240,0.2],["B2",1.1,1.2,260,0.3],["B3",1.2,1.4,300,0.3],
    ["B4",1.4,1.8,380,0.4],["B5",1.6,2.2,456,0.4],["B6",1.8,2.6,528,0.5],["B7",2.0,3.1,600,0.5]
  ];
  const DIV = 8;
  function roundFilters(f, w){
    const x = f * w;
    let nf = Math.max(DIV, Math.floor((x + DIV / 2) / DIV) * DIV);
    const bumped = nf < 0.9 * x;
    if (bumped) nf += DIV;
    return { raw: x, out: nf, bumped: bumped };
  }
  const rf = (f, w) => roundFilters(f, w).out;
  const rr = (r, d) => Math.ceil(d * r - 1e-9);      // guard float noise such as 1.1·2 = 2.2000000000000002
  const same = (h, s) => Math.ceil(h / s);
  const seW = cin => Math.max(1, Math.floor(cin * 0.25));

  function mbconv(cin, cout, e, k, hin, s, useSE){
    if (useSE === undefined) useSE = true;
    const hout = same(hin, s), E = cin * e, S = seW(cin);
    const t = { cin: cin, cout: cout, e: e, k: k, s: s, hin: hin, hout: hout, exp: E, cse: useSE ? S : 0 };
    t.expandP = e !== 1 ? cin * E : 0;          t.expandM = e !== 1 ? hin * hin * cin * E : 0;
    t.dwP = k * k * E;                          t.dwM = hout * hout * k * k * E;
    t.seP = useSE ? (E * S + S + S * E + E) : 0; t.seM = useSE ? (2 * E * S + hout * hout * E) : 0;
    t.projP = E * cout;                         t.projM = hout * hout * E * cout;
    t.bnP = 2 * ((e !== 1 ? E : 0) + E + cout);
    t.P = t.expandP + t.dwP + t.seP + t.projP + t.bnP;
    t.M = t.expandM + t.dwM + t.seM + t.projM;
    t.skip = (s === 1 && cin === cout);
    // activation elements kept for backward (rough): conv+BN+act per stage, SE-scaled map, projection conv+BN
    t.act = (e !== 1 ? 3 * hin * hin * E : 0) + 3 * hout * hout * E + hout * hout * E + 2 * hout * hout * cout;
    return t;
  }

  function build(w, d, res){
    const stages = [];
    let h = same(res, 2);
    const stem = rf(32, w);
    stages.push({ name: "stem", op: "Conv3×3, s2", hin: res, hout: h, cin: 3, cout: stem, L: 1,
      P: 27 * stem + 2 * stem, M: h * h * 27 * stem, act: 3 * h * h * stem });
    const list = [];
    BLOCKS.forEach((b, si) => {
      const cin0 = rf(b.i, w), cout = rf(b.o, w), L = rr(b.r, d);
      const st = { name: "stage " + (si + 2), op: "MBConv" + b.e + ", k" + b.k + "×" + b.k, hin: h, cin: cin0, cout: cout,
        L: L, P: 0, M: 0, act: 0, blocks: [], parts: { expand: 0, dw: 0, se: 0, proj: 0 } };
      for (let j = 0; j < L; j++){
        const t = mbconv(j === 0 ? cin0 : cout, cout, b.e, b.k, h, j === 0 ? b.s : 1);
        st.blocks.push(t); list.push(t);
        st.P += t.P; st.M += t.M; st.act += t.act;
        st.parts.expand += t.expandM; st.parts.dw += t.dwM; st.parts.se += t.seM; st.parts.proj += t.projM;
        h = t.hout;
      }
      st.hout = h; stages.push(st);
    });
    const cl = rf(320, w), head = rf(1280, w);
    stages.push({ name: "head", op: "Conv1×1 & Pool & FC", hin: h, hout: 1, cin: cl, cout: head, L: 1,
      convP: cl * head + 2 * head, fcP: head * 1000 + 1000, convM: h * h * cl * head, fcM: head * 1000,
      P: cl * head + 2 * head + head * 1000 + 1000, M: h * h * cl * head + head * 1000, act: 3 * h * h * head });
    const sum = k => stages.reduce((a, s) => a + s[k], 0);
    return { w: w, d: d, res: res, stages: stages, blocks: list, P: sum("P"), M: sum("M"), act: sum("act"), head: head };
  }

  /* ResNet-50 (torchvision v1.5 layout), counted the same way, for comparisons */
  function resnet50(res){
    let P = 0, M = 0, act = 0, h = same(res, 2);
    P += 49 * 3 * 64 + 128; M += h * h * 49 * 3 * 64; act += 3 * h * h * 64;
    h = same(h, 2); act += h * h * 64;
    let cin = 64;
    [[3,64,1],[4,128,2],[6,256,2],[3,512,2]].forEach(c => {
      for (let i = 0; i < c[0]; i++){
        const w = c[1], s = i === 0 ? c[2] : 1, out = 4 * w, h2 = same(h, s);
        P += cin * w + 2 * w;     M += h * h * cin * w;     act += 3 * h * h * w;
        P += 9 * w * w + 2 * w;   M += h2 * h2 * 9 * w * w; act += 3 * h2 * h2 * w;
        P += w * out + 2 * out;   M += h2 * h2 * w * out;   act += 3 * h2 * h2 * out;
        if (i === 0){ P += cin * out + 2 * out; M += h2 * h2 * cin * out; act += 2 * h2 * h2 * out; }
        cin = out; h = h2;
      }
    });
    P += 2048 * 1000 + 1000; M += 2048 * 1000;
    return { P: P, M: M, act: act };
  }

  const released = RELEASED.map(r => ({ name: r[0], w: r[1], d: r[2], res: r[3], drop: r[4], net: build(r[1], r[2], r[3]) }));
  const fmtM = x => x >= 1e9 ? (x / 1e9).toFixed(2) + " B" : (x / 1e6).toFixed(1) + " M";
  const fmtP = x => x >= 1e6 ? (x / 1e6).toFixed(2) + " M" : x >= 1e3 ? (x / 1e3).toFixed(1) + " k" : String(x);
  const fmtI = x => String(Math.round(x)).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
  return { BLOCKS, RELEASED, roundFilters, rf, rr, same, seW, mbconv, build, resnet50, released, fmtM, fmtP, fmtI,
    ALPHA: 1.2, BETA: 1.1, GAMMA: 1.15 };
})();
const M0 = EN.released[0].net.M;

/* ───────────────────────── 05 · FLOPs multiplier: ideal law vs exact recount ───────────────────────── */
fig("flops", function(){
  const W = 640, H = 300, m = { l: 52, r: 120, t: 16, b: 42 };
  const grid = d3.range(1, 3.0001, 0.25);
  const exact = {
    d: grid.map(v => ({ v: v, y: EN.build(1, v, 224).M / M0 })),
    w: grid.map(v => ({ v: v, y: EN.build(v, 1, 224).M / M0 })),
    r: grid.map(v => ({ v: v, y: EN.build(1, 1, Math.round(224 * v)).M / M0 }))
  };
  function draw(){
    const mm = +$v("flops-m"), log = $v("flops-scale") === "log";
    const f = frame("#flops-svg", W, H, m);
    const x = d3.scaleLinear().domain([1, 3]).range([0, f.w]);
    const y = log ? d3.scaleLog().domain([1, 10]).range([f.h, 0]) : d3.scaleLinear().domain([0, 10]).range([f.h, 0]);
    gridlines(f, y, 5);
    axes(f, x, y, "multiplier applied to one axis", "MACs ÷ B0 MACs", ax => ax.ticks(8), ay => ay.ticks(5, log ? "~g" : null));
    const xs = d3.range(1, 3.0001, 0.02);
    const law = [["d", v => v, "depth (ideal d)"], ["w", v => v * v, "width (ideal w²)"], ["r", v => v * v, "resolution (ideal r²)"]];
    law.forEach(L => {
      f.g.append("path").datum(xs).attr("fill", "none").attr("stroke", AXC[L[0]]).attr("stroke-width", 1.6)
        .attr("stroke-dasharray", L[0] === "r" ? "5,3" : null).attr("opacity", 0.8)
        .attr("d", d3.line().x(v => x(v)).y(v => y(Math.max(1, L[1](v)))));
      f.g.selectAll(null).data(exact[L[0]]).enter().append("circle").attr("cx", p => x(p.v)).attr("cy", p => y(Math.max(1, p.y)))
        .attr("r", 3.2).attr("fill", AXC[L[0]]).attr("stroke", "#0f1117");
    });
    f.g.append("line").attr("x1", x(mm)).attr("x2", x(mm)).attr("y1", 0).attr("y2", f.h).attr("stroke", PAL.ink).attr("stroke-opacity", 0.35).attr("stroke-dasharray", "3,3");
    legend(f.g, f.w + 14, 10, [["depth d", AXC.d], ["width w", AXC.w], ["resolution r", AXC.r]]);
    txt(f.g, f.w + 14, 70, "lines: ideal law", { size: 10 });
    txt(f.g, f.w + 14, 84, "dots: exact recount", { size: 10 });
    const nd = EN.build(1, mm, 224), nw = EN.build(mm, 1, 224), nr = EN.build(1, 1, Math.round(224 * mm));
    d3.select("#flops-read").html(
      "multiplier <b>" + f2(mm) + "</b> · depth: ideal ×" + f2(mm) + ", exact <b>×" + f2(nd.M / M0) + "</b> (" + nd.blocks.length + " blocks)" +
      " · width: ideal ×" + f2(mm * mm) + ", exact <b>×" + f2(nw.M / M0) + "</b>" +
      " · resolution: ideal ×" + f2(mm * mm) + ", exact <b>×" + f2(nr.M / M0) + "</b> (" + Math.round(224 * mm) + " px)");
  }
  bind(["flops-m", "flops-scale"], draw); draw();
});

/* ───────────────────────── 06 · single-axis saturation (Figure 3 read-offs) ───────────────────────── */
fig("sat", function(){
  const W = 640, H = 300, m = { l: 52, r: 110, t: 16, b: 42 };
  // accuracy and x position read off the paper's Figure 3 (approximate); x recounted from the spec below
  const S = {
    w: { lab: "width", vals: [1.0, 1.4, 1.8, 2.6, 3.8, 5.0], acc: [76.1, 77.5, 78.5, 79.5, 80.0, 80.3], px: [0.5, 0.8, 1.15, 2.25, 4.9, 8.25] },
    d: { lab: "depth", vals: [1, 2, 3, 4, 6, 8],             acc: [76.1, 78.5, 78.8, 79.0, 80.0, 80.0], px: [0.5, 0.95, 1.4, 1.85, 2.75, 3.65] },
    r: { lab: "resolution", vals: [1.0, 1.3, 1.5, 1.7, 1.9, 2.2, 2.5], acc: [76.1, 77.7, 78.1, 78.8, 79.1, 79.5, 79.7], px: [0.5, 0.95, 1.15, 1.45, 1.95, 2.55, 3.1] }
  };
  S.w.rx = S.w.vals.map(v => EN.build(v, 1, 224).M / 1e9);
  S.d.rx = S.d.vals.map(v => EN.build(1, v, 224).M / 1e9);
  S.r.rx = S.r.vals.map(v => EN.build(1, 1, Math.round(224 * v)).M / 1e9);
  function draw(){
    const show = $v("sat-show"), log = $v("sat-x") === "log", src = $v("sat-src") || "recount";
    const keys = show === "all" ? ["w", "d", "r"] : [show];
    const f = frame("#sat-svg", W, H, m);
    const x = log ? d3.scaleLog().base(2).domain([0.3, 10]).range([0, f.w]) : d3.scaleLinear().domain([0, 9]).range([0, f.w]);
    const y = d3.scaleLinear().domain([75, 81]).range([f.h, 0]);
    gridlines(f, y, 6);
    axes(f, x, y, (src === "paper" ? "x as plotted in the paper (read off)" : "MACs recounted from the spec") + ", billions" + (log ? " (log₂)" : ""), "ImageNet top-1 (%)",
      ax => ax.ticks(log ? 6 : 9, log ? "~g" : null));
    let msg = [];
    keys.forEach(k => {
      const s = S[k], xs = src === "paper" ? s.px : s.rx;
      const pts = s.vals.map((v, i) => ({ v: v, x: xs[i], a: s.acc[i] }));
      f.g.append("path").datum(pts).attr("fill", "none").attr("stroke", AXC[k]).attr("stroke-width", 1.8)
        .attr("d", d3.line().x(p => x(p.x)).y(p => y(p.a)));
      const gp = f.g.selectAll(null).data(pts).enter().append("g");
      gp.append("circle").attr("cx", p => x(p.x)).attr("cy", p => y(p.a)).attr("r", 3.5).attr("fill", AXC[k]).attr("stroke", "#0f1117");
      gp.append("text").attr("x", p => x(p.x) + 5).attr("y", p => y(p.a) + 13).attr("font-size", 9).attr("fill", AXC[k]).text(p => k + "=" + p.v);
      const n = pts.length, a = pts[n - 2], b = pts[n - 1], a0 = pts[0], b0 = pts[1];
      const late = (b.a - a.a) / Math.log2(b.x / a.x), early = (b0.a - a0.a) / Math.log2(b0.x / a0.x);
      msg.push(s.lab + ": first step <b>" + f2(early) + "</b> pts/doubling, last step <b>" + f2(late) + "</b>");
    });
    legend(f.g, f.w + 14, 10, [["width", AXC.w], ["depth", AXC.d], ["resolution", AXC.r]].filter(it => keys.indexOf(it[0][0]) >= 0));
    txt(f.g, f.w + 14, 70, "accuracy: read", { size: 10 }); txt(f.g, f.w + 14, 83, "off Figure 3", { size: 10 });
    d3.select("#sat-read").html(msg.join(" · ") + " (x = " + (src === "paper" ? "paper's plotted positions" : "recounted MACs") + ")");
  }
  bind(["sat-show", "sat-x", "sat-src"], draw); draw();
});

/* ───────────────────────── 09 · five scaling modes (paper Figure 2) ───────────────────────── */
fig("modes", function(){
  const W = 640, H = 300;
  function draw(){
    const phi = +$v("modes-phi"), match = $v("modes-match") === "same";
    const svg = d3.select("#modes-svg"); svg.selectAll("*").remove();
    const F = Math.pow(EN.ALPHA * EN.BETA * EN.BETA * EN.GAMMA * EN.GAMMA, phi);
    const modes = [
      { k: "baseline", d: 1, w: 1, r: 1, col: PAL.muted },
      { k: "width", d: 1, w: match ? Math.sqrt(F) : Math.pow(EN.BETA, phi), r: 1, col: AXC.w },
      { k: "depth", d: match ? F : Math.pow(EN.ALPHA, phi), w: 1, r: 1, col: AXC.d },
      { k: "resolution", d: 1, w: 1, r: match ? Math.sqrt(F) : Math.pow(EN.GAMMA, phi), col: AXC.r },
      { k: "compound", d: Math.pow(EN.ALPHA, phi), w: Math.pow(EN.BETA, phi), r: Math.pow(EN.GAMMA, phi), col: AXC.c }
    ];
    const pw = W / 5, base = 250;
    modes.forEach((md, i) => {
      const g = svg.append("g").attr("transform", "translate(" + (i * pw) + ",0)");
      const cx = pw / 2;
      txt(g, cx, 16, md.k, { anchor: "middle", fill: md.col, size: 11.5, bold: true });
      // input image square, side ∝ resolution
      const side = Math.min(60, 26 * md.r);
      g.append("rect").attr("x", cx - side / 2).attr("y", base - side + 26).attr("width", side).attr("height", side)
        .attr("fill", "rgba(45,212,191,.10)").attr("stroke", md.r > 1.0001 ? AXC.r : PAL.line);
      // layers: count ∝ depth, bar width ∝ width
      const n = Math.max(1, Math.round(4 * md.d)), avail = 150, gap = 2;
      const bh = Math.max(1.2, Math.min(12, (avail - gap * (n - 1)) / n));
      const bw = Math.min(pw - 14, 46 * md.w);
      for (let j = 0; j < n; j++){
        const yy = base - side + 20 - (j + 1) * (bh + gap);
        g.append("rect").attr("x", cx - bw / 2).attr("y", yy).attr("width", bw).attr("height", bh).attr("rx", 1.5)
          .attr("fill", md.col).attr("fill-opacity", 0.55);
      }
      const net = EN.build(md.w, md.d, Math.round(224 * md.r));
      md.exact = net.M / M0; md.ideal = md.d * md.w * md.w * md.r * md.r;
      txt(g, cx, 292, "×" + f2(md.ideal) + " ideal", { anchor: "middle", size: 9.5, mono: true });
    });
    d3.select("#modes-read").html("φ = <b>" + f2(phi) + "</b> · compound: d = " + f2(modes[4].d) + ", w = " + f2(modes[4].w) + ", r = " + f2(modes[4].r) +
      " (" + Math.round(224 * modes[4].r) + " px) · exact MACs vs B0 — " +
      modes.slice(1).map(md => md.k + " <b>×" + f2(md.exact) + "</b>").join(", "));
  }
  bind(["modes-phi", "modes-match"], draw); draw();
});

/* ───────────────────────── 10 · constraint explorer ───────────────────────── */
fig("grid", function(){
  const W = 640, H = 300, m = { l: 52, r: 170, t: 16, b: 42 };
  function draw(){
    const a = +$v("grid-a"), b = +$v("grid-b"), g = +$v("grid-g");
    const f = frame("#grid-svg", W, H, m);
    const x = d3.scaleLinear().domain([1, 1.4]).range([0, f.w]), y = d3.scaleLinear().domain([1, 1.4]).range([f.h, 0]);
    axes(f, x, y, "β (width base)", "γ (resolution base)", ax => ax.ticks(8), ay => ay.ticks(8));
    // band 1.9..2.1 as shaded area between two curves
    const bs = d3.range(1, 1.4001, 0.005);
    const gOf = (bb, c) => Math.sqrt(c / (a * bb * bb));
    const area = d3.area().x(bb => x(bb)).y0(bb => y(Math.max(1, Math.min(1.4, gOf(bb, 1.9))))).y1(bb => y(Math.max(1, Math.min(1.4, gOf(bb, 2.1)))));
    f.g.append("path").datum(bs).attr("d", area).attr("fill", PAL.good).attr("fill-opacity", 0.10);
    f.g.append("path").datum(bs.filter(bb => gOf(bb, 2) >= 1 && gOf(bb, 2) <= 1.4)).attr("fill", "none").attr("stroke", PAL.good).attr("stroke-width", 1.6)
      .attr("d", d3.line().x(bb => x(bb)).y(bb => y(gOf(bb, 2))));
    let inBand = 0;
    const pts = [];
    for (let bi = 0; bi <= 8; bi++) for (let gi = 0; gi <= 8; gi++){
      const bb = 1 + 0.05 * bi, gg = 1 + 0.05 * gi, p = a * bb * bb * gg * gg, ok = Math.abs(p - 2) / 2 <= 0.05;
      if (ok) inBand++;
      pts.push({ b: bb, g: gg, ok: ok });
    }
    f.g.selectAll(null).data(pts).enter().append("circle").attr("cx", p => x(p.b)).attr("cy", p => y(p.g)).attr("r", 2.6)
      .attr("fill", p => p.ok ? PAL.good : XC.dim).attr("fill-opacity", p => p.ok ? 0.9 : 0.5);
    if (Math.abs(a - 1.2) < 1e-9){
      f.g.append("path").attr("d", d3.symbol().type(d3.symbolStar).size(110)).attr("transform", "translate(" + x(1.1) + "," + y(1.15) + ")").attr("fill", PAL.B);
    }
    f.g.append("circle").attr("cx", x(b)).attr("cy", y(g)).attr("r", 6).attr("fill", "none").attr("stroke", PAL.ink).attr("stroke-width", 2);
    const prod = a * b * b * g * g, T = Math.log(prod);
    const sh = T > 0 ? [Math.log(a) / T, 2 * Math.log(b) / T, 2 * Math.log(g) / T] : [0, 0, 0];
    const gx = f.w + 16;
    txt(f.g, gx, 12, "α·β²·γ² = " + prod.toFixed(4), { fill: PAL.ink, mono: true, size: 11 });
    txt(f.g, gx, 30, "deviation from 2: " + ((prod / 2 - 1) * 100).toFixed(1) + "%", { mono: true });
    txt(f.g, gx, 48, "φ = 7 budget: ×" + Math.pow(prod, 7).toFixed(1), { mono: true });
    txt(f.g, gx, 72, "share of log-FLOPs:", {});
    [["depth", sh[0], AXC.d], ["width", sh[1], AXC.w], ["resolution", sh[2], AXC.r]].forEach((s, i) => {
      f.g.append("rect").attr("x", gx).attr("y", 80 + i * 20).attr("width", Math.max(0, 140 * s[1])).attr("height", 12).attr("fill", s[2]).attr("fill-opacity", 0.8);
      txt(f.g, gx + 3, 90 + i * 20, s[0] + " " + (100 * s[1]).toFixed(1) + "%", { fill: "#0f1117", size: 9.5, bold: true });
    });
    txt(f.g, gx, 158, "★ paper's (1.1, 1.15) at α = 1.2", { size: 9.5, fill: PAL.B });
    txt(f.g, gx, 174, "● grid points within ±5% of 2", { size: 9.5, fill: PAL.good });
    txt(f.g, gx, 190, "band: 1.9 ≤ product ≤ 2.1", { size: 9.5 });
    d3.select("#grid-read").html("α = " + f2(a) + ", β = " + f2(b) + ", γ = " + f2(g) + " → α·β²·γ² = <b>" + prod.toFixed(4) + "</b> · " +
      inBand + " of 81 grid points (β, γ ∈ 1.00…1.40, step 0.05) satisfy the constraint to ±5% at this α");
  }
  bind(["grid-a", "grid-b", "grid-g"], draw); draw();
});

/* ───────────────────────── 11 · φ ray vs released models ───────────────────────── */
fig("phi", function(){
  const W = 640, H = 300, m = { l: 52, r: 120, t: 16, b: 42 };
  const rel = EN.released.map((r, i) => ({ name: r.name, i: i, d: r.d, w: r.w, r: r.res / 224,
    phiF: Math.log(r.net.M / M0) / Math.log(EN.ALPHA * EN.BETA * EN.BETA * EN.GAMMA * EN.GAMMA) }));
  function draw(){
    const place = $v("phi-place"), ph = +$v("phi-phi");
    const f = frame("#phi-svg", W, H, m);
    const x = d3.scaleLinear().domain([0, 7.5]).range([0, f.w]), y = d3.scaleLinear().domain([1, 3.8]).range([f.h, 0]);
    gridlines(f, y, 6);
    axes(f, x, y, "compound coefficient φ", "multiplier relative to B0", ax => ax.ticks(8), ay => ay.ticks(6));
    const xs = d3.range(0, 7.5001, 0.05);
    [["d", EN.ALPHA], ["w", EN.BETA], ["r", EN.GAMMA]].forEach(L => {
      f.g.append("path").datum(xs).attr("fill", "none").attr("stroke", AXC[L[0]]).attr("stroke-width", 1.6).attr("opacity", 0.85)
        .attr("d", d3.line().x(v => x(v)).y(v => y(Math.pow(L[1], v))));
      rel.forEach(p => {
        const px = place === "flops" ? p.phiF : p.i;
        f.g.append("circle").attr("cx", x(px)).attr("cy", y(p[L[0]])).attr("r", 4).attr("fill", AXC[L[0]]).attr("stroke", "#0f1117");
      });
    });
    rel.forEach(p => {
      const px = place === "flops" ? p.phiF : p.i;
      txt(f.g, x(px), y(p.d) - 8, p.name, { anchor: "middle", size: 9, fill: PAL.ink });
    });
    f.g.append("line").attr("x1", x(ph)).attr("x2", x(ph)).attr("y1", 0).attr("y2", f.h).attr("stroke", PAL.ink).attr("stroke-opacity", 0.35).attr("stroke-dasharray", "3,3");
    legend(f.g, f.w + 14, 10, [["depth 1.2ᵠ", AXC.d], ["width 1.1ᵠ", AXC.w], ["res 1.15ᵠ", AXC.r]]);
    const near = rel.reduce((a, p) => Math.abs((place === "flops" ? p.phiF : p.i) - ph) < Math.abs((place === "flops" ? a.phiF : a.i) - ph) ? p : a, rel[0]);
    const net = EN.build(Math.pow(EN.BETA, ph), Math.pow(EN.ALPHA, ph), Math.round(224 * Math.pow(EN.GAMMA, ph)));
    d3.select("#phi-read").html("formula at φ = <b>" + f1(ph) + "</b>: d = " + f2(Math.pow(EN.ALPHA, ph)) + ", w = " + f2(Math.pow(EN.BETA, ph)) +
      ", res = " + Math.round(224 * Math.pow(EN.GAMMA, ph)) + " px → built: " + EN.fmtM(net.M) + " MACs (×" + f2(net.M / M0) + " B0, ideal ×" +
      f2(Math.pow(1.92027, ph)) + ") · nearest released: <b>" + near.name + "</b> (d " + near.d + ", w " + near.w + ", " + Math.round(near.r * 224) + " px, FLOP-implied φ " + f2(near.phiF) + ")");
  }
  bind(["phi-place", "phi-phi"], draw); draw();
});

/* ───────────────────────── 13 · round_filters / round_repeats ───────────────────────── */
fig("round", function(){
  const W = 640, H = 300, m = { l: 52, r: 16, t: 20, b: 58 };
  const base = [["stem", 32, null]].concat(EN.BLOCKS.map((b, i) => ["st" + (i + 2), b.o, b.r])).concat([["head", 1280, null]]);
  function draw(){
    const w = +$v("round-w"), d = +$v("round-d");
    const f = frame("#round-svg", W, H, m);
    const rows = base.map(b => { const R = EN.roundFilters(b[1], w); return { k: b[0], c: b[1], raw: R.raw, out: R.out, bump: R.bumped, rep: b[2], rep2: b[2] ? EN.rr(b[2], d) : null }; });
    const x = d3.scaleBand().domain(rows.map(r => r.k)).range([0, f.w]).padding(0.25);
    const y = d3.scaleLog().domain([6, 4000]).range([f.h, 0]);
    gridlines(f, y, 4);
    axes(f, x, y, null, "channels (log)", null, ay => ay.ticks(4, "~g"));
    rows.forEach(r => {
      const bx = x(r.k), bw = x.bandwidth();
      f.g.append("rect").attr("x", bx).attr("y", y(r.out)).attr("width", bw / 2).attr("height", f.h - y(r.out)).attr("fill", r.bump ? PAL.bad : PAL.A).attr("fill-opacity", 0.8);
      f.g.append("rect").attr("x", bx + bw / 2).attr("y", y(Math.max(6, r.raw))).attr("width", bw / 2).attr("height", f.h - y(Math.max(6, r.raw))).attr("fill", "none").attr("stroke", PAL.B).attr("stroke-dasharray", "3,2");
      txt(f.g, bx + bw / 2, y(Math.max(r.out, r.raw)) - 4, String(r.out), { anchor: "middle", size: 9.5, fill: PAL.ink, mono: true });
      if (r.rep) txt(f.g, bx + bw / 2, f.h + 26, r.rep + "→" + r.rep2, { anchor: "middle", size: 9.5, fill: PAL.ink, mono: true });
    });
    txt(f.g, -44, f.h + 26, "repeats", { size: 9.5 });
    legend(f.g, 10, -6, []);
    txt(f.g, 4, 4, "■ built (blue; red = 10% guard fired)   ┆ requested w·C (dashed)", { size: 9.5, fill: PAL.ink });
    const net = EN.build(w, d, 224);
    const bumps = rows.filter(r => r.bump).map(r => r.k);
    const blocks = rows.filter(r => r.rep).reduce((a, r) => a + r.rep2, 0);
    d3.select("#round-read").html("w = " + f2(w) + ": built widths " + rows.map(r => r.out).join(", ") +
      (bumps.length ? " · guard fired at <b>" + bumps.join(", ") + "</b>" : " · guard never fired") +
      " · d = " + f1(d) + ": <b>" + blocks + "</b> blocks (" + f2(blocks / 16) + "× B0, nominal " + f1(d) + "×) · at 224 px: " +
      EN.fmtP(net.P) + " params, " + EN.fmtM(net.M) + " MACs");
  }
  bind(["round-w", "round-d"], draw); draw();
});

/* ───────────────────────── 14 · NAS reward explorer ───────────────────────── */
fig("nas", function(){
  const W = 640, H = 300, m = { l: 52, r: 16, t: 16, b: 42 }, T = 400;
  function draw(){
    const w = +$v("nas-w"), F = +$v("nas-f"), A = +$v("nas-a");
    const f = frame("#nas-svg", W, H, m);
    const x = d3.scaleLog().domain([100, 1600]).range([0, f.w]), y = d3.scaleLinear().domain([68, 84]).range([f.h, 0]);
    gridlines(f, y, 8);
    axes(f, x, y, "FLOPs (M, log) — target T = 400 M", "top-1 (%)", ax => ax.tickValues([100, 200, 400, 800, 1600]).tickFormat(d3.format("~g")));
    const R = (acc, fl) => acc / 100 * Math.pow(fl / T, w);
    const rA = R(76, T), rC = R(A, F);
    const fs = d3.range(100, 1601, 10);
    [rA - 0.02, rA - 0.01, rA, rA + 0.01, rA + 0.02].forEach((lev, i) => {
      const pts = fs.map(fl => ({ fl: fl, a: 100 * lev / Math.pow(fl / T, w) })).filter(p => p.a >= 68 && p.a <= 84);
      f.g.append("path").datum(pts).attr("fill", "none").attr("stroke", i === 2 ? PAL.B : XC.dim).attr("stroke-width", i === 2 ? 1.8 : 1)
        .attr("stroke-dasharray", i === 2 ? null : "4,3").attr("d", d3.line().x(p => x(p.fl)).y(p => y(p.a)));
    });
    f.g.append("line").attr("x1", x(T)).attr("x2", x(T)).attr("y1", 0).attr("y2", f.h).attr("stroke", PAL.muted).attr("stroke-opacity", 0.4);
    f.g.append("circle").attr("cx", x(T)).attr("cy", y(76)).attr("r", 5).attr("fill", PAL.B);
    txt(f.g, x(T) + 7, y(76) + 14, "A: 76.0% @ 400M", { fill: PAL.B, size: 10 });
    const win = rC >= rA;
    f.g.append("circle").attr("cx", x(F)).attr("cy", y(A)).attr("r", 6).attr("fill", win ? PAL.good : PAL.bad).attr("stroke", "#0f1117");
    txt(f.g, x(F) + 8, y(A) - 6, "candidate", { fill: win ? PAL.good : PAL.bad, size: 10 });
    txt(f.g, 6, 12, "orange: iso-reward through A · grey: ±0.01 reward steps", { size: 9.5 });
    const breakEven = 76 * Math.pow(F / T, -w);
    d3.select("#nas-read").html("w = " + f2(w) + " · reward(A) = " + rA.toFixed(4) + " · reward(candidate: " + f1(A) + "% @ " + F + " M) = <b>" + rC.toFixed(4) + "</b> → " +
      (win ? "<b>beats</b>" : "loses to") + " A · to tie A at " + F + " M it needs " + f2(breakEven) + "% · doubling FLOPs must buy ×" + f3(Math.pow(2, -w)) + " accuracy");
  }
  bind(["nas-w", "nas-f", "nas-a"], draw); draw();
});

/* ───────────────────────── 15 · architecture diagram, counted live ───────────────────────── */
fig("arch", function(){
  const W = 640, H = 300;
  let hover = -1;
  function draw(){
    const mi = +$v("arch-model"), key = $v("arch-color");
    const R = EN.released[mi], net = R.net, st = net.stages;
    const svg = d3.select("#arch-svg"); svg.selectAll("*").remove();
    const tot = net[key], maxShare = d3.max(st, s => s[key] / tot);
    const x0 = 14, avail = W - 28, gap = 4;
    const wts = st.map(s => 1 + Math.sqrt(s.L));
    const sw = wts.reduce((a, b) => a + b, 0);
    let x = x0;
    const col = d3.scaleSequential(d3.interpolateRgb("#1e2a44", key === "M" ? PAL.A : key === "P" ? PAL.B : XC.pu)).domain([0, maxShare]);
    st.forEach((s, i) => {
      const bw = (avail - gap * (st.length - 1)) * wts[i] / sw;
      const ch = s.cout, bh = 30 + 26 * Math.log2(ch / 8);
      const y = 200 - bh;
      const on = hover === i;
      svg.append("rect").attr("x", x).attr("y", y).attr("width", bw).attr("height", bh).attr("rx", 5)
        .attr("fill", col(s[key] / tot)).attr("stroke", on ? PAL.ink : PAL.line).attr("stroke-width", on ? 2 : 1);
      txt(svg, x + bw / 2, 216, s.name.replace("stage ", "s"), { anchor: "middle", size: 9.5, fill: on ? PAL.ink : PAL.muted });
      txt(svg, x + bw / 2, 230, "×" + s.L, { anchor: "middle", size: 9.5, mono: true, fill: PAL.muted });
      txt(svg, x + bw / 2, 244, String(ch), { anchor: "middle", size: 9.5, mono: true, fill: PAL.ink });
      txt(svg, x + bw / 2, 258, s.hin + "²", { anchor: "middle", size: 9, mono: true, fill: XC.dim });
      txt(svg, x + bw / 2, y - 5, (100 * s[key] / tot).toFixed(0) + "%", { anchor: "middle", size: 9.5, fill: PAL.ink });
      svg.append("rect").attr("x", x).attr("y", 10).attr("width", bw).attr("height", 250).attr("fill", "transparent").style("cursor", "pointer")
        .on("mouseenter", () => { hover = i; draw(); }).on("mouseleave", () => { hover = -1; draw(); });
      x += bw + gap;
    });
    txt(svg, 14, 16, R.name + " · " + R.res + " px · rows under each stage: name, repeats, output channels, input size", { size: 10 });
    txt(svg, 14, 286, "totals: " + EN.fmtI(net.P) + " params · " + EN.fmtM(net.M) + " MACs · " + net.blocks.length + " MBConv blocks · ~" +
      (net.act * 4 / Math.pow(2, 20)).toFixed(0) + " MiB fp32 activations per image (rough)", { size: 10, fill: PAL.ink });
    let msg;
    if (hover < 0){
      msg = R.name + ": <b>" + EN.fmtI(net.P) + "</b> parameters, <b>" + EN.fmtI(net.M) + "</b> MACs (paper: " +
        ["5.3M/0.39B","7.8M/0.70B","9.2M/1.0B","12M/1.8B","19M/4.2B","30M/9.9B","43M/19B","66M/37B"][mi] + ") — hover a stage";
    } else {
      const s = st[hover];
      msg = "<b>" + s.name + "</b> " + s.op + " · " + s.hin + "→" + s.hout + " px · " + s.cin + "→" + s.cout + " ch · ×" + s.L +
        " · params <b>" + EN.fmtI(s.P) + "</b> (" + (100 * s.P / net.P).toFixed(1) + "%) · MACs <b>" + EN.fmtI(s.M) + "</b> (" + (100 * s.M / net.M).toFixed(1) + "%)";
    }
    d3.select("#arch-read").html(msg);
  }
  bind(["arch-model", "arch-color"], () => { hover = -1; draw(); }); draw();
});

/* ───────────────────────── 17 · one MBConv block, live ───────────────────────── */
fig("mb", function(){
  const W = 640, H = 300;
  function draw(){
    const cin = +$v("mb-cin"), cout = +$v("mb-cout"), e = +$v("mb-e"), k = +$v("mb-k"), s = +$v("mb-s"), h = +$v("mb-h"), se = $v("mb-se") === "1";
    const t = EN.mbconv(cin, cout, e, k, h, s, se);
    const svg = d3.select("#mb-svg"); svg.selectAll("*").remove();
    const maxC = Math.max(t.exp, cin, cout), sc = c => 20 + 150 * Math.sqrt(c / maxC);
    const cols = [
      { lab: "input", c: cin, hw: h, col: PAL.muted },
      { lab: e === 1 ? "(no expand)" : "expand 1×1", c: t.exp, hw: h, col: PAL.B, skip: e === 1 },
      { lab: "depthwise " + k + "×" + k + (s > 1 ? " s" + s : ""), c: t.exp, hw: t.hout, col: XC.te },
      { lab: se ? "SE gate" : "(no SE)", c: t.exp, hw: t.hout, col: XC.pu, skip: !se },
      { lab: "project 1×1 (linear)", c: cout, hw: t.hout, col: PAL.A }
    ];
    const cx = i => 60 + i * 120, mid = 130;
    cols.forEach((c, i) => {
      const bh = sc(c.c);
      if (i > 0) svg.append("line").attr("x1", cx(i - 1) + 22).attr("x2", cx(i) - 22).attr("y1", mid).attr("y2", mid).attr("stroke", PAL.line).attr("stroke-width", 2);
      svg.append("rect").attr("x", cx(i) - 20).attr("y", mid - bh / 2).attr("width", 40).attr("height", bh).attr("rx", 5)
        .attr("fill", c.col).attr("fill-opacity", c.skip ? 0.08 : 0.35).attr("stroke", c.col).attr("stroke-dasharray", c.skip ? "4,3" : null);
      txt(svg, cx(i), mid + bh / 2 + 16, c.c + " ch", { anchor: "middle", size: 10, fill: PAL.ink, mono: true });
      txt(svg, cx(i), mid + bh / 2 + 30, c.hw + "×" + c.hw, { anchor: "middle", size: 9.5, mono: true });
      txt(svg, cx(i), 18, c.lab, { anchor: "middle", size: 10, fill: c.col });
    });
    if (se){
      svg.append("rect").attr("x", cx(3) - 16).attr("y", 30).attr("width", 32).attr("height", 14).attr("rx", 3).attr("fill", XC.pu).attr("fill-opacity", 0.5);
      txt(svg, cx(3), 41, "S=" + t.cse, { anchor: "middle", size: 9.5, fill: PAL.ink, mono: true });
    }
    if (t.skip){
      svg.append("path").attr("d", "M" + cx(0) + ",250 C" + cx(0) + ",285 " + cx(4) + ",285 " + cx(4) + ",250").attr("fill", "none").attr("stroke", PAL.good).attr("stroke-width", 1.8).attr("stroke-dasharray", "5,3");
      txt(svg, (cx(0) + cx(4)) / 2, 296, "identity skip + drop-path (stride 1 and C_in = C_out)", { anchor: "middle", size: 10, fill: PAL.good });
    } else {
      txt(svg, (cx(0) + cx(4)) / 2, 292, "no skip: " + (s !== 1 ? "stride " + s : "") + (s !== 1 && cin !== cout ? " and " : "") + (cin !== cout ? "C_in ≠ C_out" : ""), { anchor: "middle", size: 10, fill: PAL.bad });
    }
    const pct = v => (100 * v / t.M).toFixed(1) + "%";
    d3.select("#mb-read").html("params <b>" + EN.fmtI(t.P) + "</b> = expand " + EN.fmtI(t.expandP) + " + dw " + EN.fmtI(t.dwP) + " + SE " + EN.fmtI(t.seP) + " + project " + EN.fmtI(t.projP) + " + BN " + EN.fmtI(t.bnP) +
      " · MACs <b>" + EN.fmtI(t.M) + "</b> = expand " + pct(t.expandM) + ", dw " + pct(t.dwM) + ", SE " + pct(t.seM) + ", project " + pct(t.projM));
  }
  bind(["mb-cin", "mb-cout", "mb-e", "mb-k", "mb-s", "mb-h", "mb-se"], draw); draw();
});

/* ───────────────────────── 19 · squeeze-and-excitation gating ───────────────────────── */
fig("se", function(){
  const W = 640, H = 300, Cn = 8;
  const scenes = [
    [0.9, 0.2, 0.7, 0.1, 0.5, 0.05, 0.3, 0.6],
    [0.1, 0.8, 0.05, 0.7, 0.1, 0.6, 0.05, 0.1],
    [0.3, 0.3, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2]
  ];
  // fixed deterministic weights (a small LCG), so the demo is reproducible
  function weights(S){
    let s = 12345; const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648 * 2 - 1; };
    const W1 = d3.range(S).map(() => d3.range(Cn).map(() => 1.6 * rnd()));
    const W2 = d3.range(Cn).map(() => d3.range(S).map(() => 1.8 * rnd()));
    return { W1: W1, W2: W2 };
  }
  const sig = v => 1 / (1 + Math.exp(-v)), swish = v => v * sig(v);
  function draw(){
    const sc = +$v("se-scene"), boost = +$v("se-boost"), S = +$v("se-s");
    const z = scenes[sc].slice(); z[0] = Math.max(0, z[0] + boost);
    const Wt = weights(S);
    const u = Wt.W1.map(row => swish(row.reduce((a, w, i) => a + w * z[i], 0)));
    const gate = Wt.W2.map(row => sig(row.reduce((a, w, j) => a + w * u[j], 0)));
    const out = z.map((v, i) => v * gate[i]);
    const svg = d3.select("#se-svg"); svg.selectAll("*").remove();
    const panels = [["pooled mean zᶜ", z, PAL.A, 3], ["gate gᶜ = σ(·)", gate, XC.pu, 1], ["rescaled gᶜ·zᶜ", out, PAL.good, 3]];
    panels.forEach((p, pi) => {
      const g = svg.append("g").attr("transform", "translate(" + (20 + pi * 210) + ",30)");
      txt(g, 90, -10, p[0], { anchor: "middle", fill: p[2], size: 11 });
      const y = d3.scaleLinear().domain([0, p[3]]).range([200, 0]);
      g.append("line").attr("x1", 0).attr("x2", 184).attr("y1", 200).attr("y2", 200).attr("stroke", PAL.line);
      p[1].forEach((v, i) => {
        g.append("rect").attr("x", i * 23).attr("y", y(Math.min(v, p[3]))).attr("width", 18).attr("height", 200 - y(Math.min(v, p[3]))).attr("fill", p[2]).attr("fill-opacity", i === 0 ? 0.95 : 0.6);
        txt(g, i * 23 + 9, 214, "c" + (i + 1), { anchor: "middle", size: 9 });
        txt(g, i * 23 + 9, y(Math.min(v, p[3])) - 3, v.toFixed(2), { anchor: "middle", size: 8.5, fill: PAL.ink });
      });
    });
    const kept = gate.map((g, i) => [g, i]).sort((a, b) => b[0] - a[0]);
    d3.select("#se-read").html("squeeze width S = " + S + " (SE params for E = 8: " + (8 * S + S + S * 8 + 8) + ") · gates: " +
      gate.map(g => g.toFixed(2)).join(" ") + " · most kept: <b>c" + (kept[0][1] + 1) + "</b> (" + kept[0][0].toFixed(2) + "), most suppressed: <b>c" +
      (kept[7][1] + 1) + "</b> (" + kept[7][0].toFixed(2) + ")");
  }
  bind(["se-scene", "se-boost", "se-s"], draw); draw();
});

/* ───────────────────────── 20 · Swish vs ReLU ───────────────────────── */
fig("swish", function(){
  const W = 640, H = 300, m = { l: 52, r: 110, t: 16, b: 42 };
  const sig = v => 1 / (1 + Math.exp(-v));
  function draw(){
    const b = +$v("swish-b"), px = +$v("swish-x");
    const f = frame("#swish-svg", W, H, m);
    const x = d3.scaleLinear().domain([-5, 5]).range([0, f.w]), y = d3.scaleLinear().domain([-1, 5]).range([f.h, 0]);
    gridlines(f, y, 6);
    axes(f, x, y, "x", "value", ax => ax.ticks(10), ay => ay.ticks(6));
    const xs = d3.range(-5, 5.0001, 0.02);
    const sw = v => v * sig(b * v), dsw = v => sig(b * v) + b * v * sig(b * v) * (1 - sig(b * v));
    const relu = v => Math.max(0, v), drelu = v => v > 0 ? 1 : 0;
    const line = fn => d3.line().x(v => x(v)).y(v => y(fn(v)));
    f.g.append("path").datum(xs).attr("d", line(relu)).attr("fill", "none").attr("stroke", PAL.muted).attr("stroke-width", 1.5);
    f.g.append("path").datum(xs).attr("d", line(drelu)).attr("fill", "none").attr("stroke", PAL.muted).attr("stroke-width", 1.2).attr("stroke-dasharray", "4,3");
    f.g.append("path").datum(xs).attr("d", line(sw)).attr("fill", "none").attr("stroke", PAL.good).attr("stroke-width", 2);
    f.g.append("path").datum(xs).attr("d", line(dsw)).attr("fill", "none").attr("stroke", PAL.good).attr("stroke-width", 1.4).attr("stroke-dasharray", "4,3");
    f.g.append("circle").attr("cx", x(px)).attr("cy", y(sw(px))).attr("r", 4.5).attr("fill", PAL.B);
    legend(f.g, f.w + 14, 10, [["swish_b", PAL.good], ["ReLU", PAL.muted]]);
    txt(f.g, f.w + 14, 52, "dashed: derivative", { size: 10 });
    // numeric minimum of swish_b on the plotted range
    let mn = [0, 0]; xs.forEach(v => { const s = sw(v); if (s < mn[1]) mn = [v, s]; });
    d3.select("#swish-read").html("b = " + f1(b) + " · at x = " + f2(px) + ": swish = <b>" + f3(sw(px)) + "</b>, swish′ = <b>" + f3(dsw(px)) + "</b>, ReLU = " + f3(relu(px)) +
      " · minimum ≈ " + f3(mn[1]) + " at x ≈ " + f2(mn[0]) + (b === 0 ? " (b = 0 is the line x/2)" : ""));
  }
  bind(["swish-b", "swish-x"], draw); draw();
});

/* ───────────────────────── 21 · schedules: LR and stochastic depth ───────────────────────── */
fig("train", function(){
  const W = 640, H = 300, m = { l: 56, r: 16, t: 16, b: 42 };
  function draw(){
    const show = $v("train-show"), bs = +$v("train-bs"), mi = +$v("train-model");
    const f = frame("#train-svg", W, H, m);
    if (show === "lr"){
      const lr0 = 0.016 * bs / 256, E = 350, WU = 5;
      const lr = ep => ep < WU ? lr0 * ep / WU : lr0 * Math.pow(0.97, Math.floor(ep / 2.4));
      const x = d3.scaleLinear().domain([0, E]).range([0, f.w]), y = d3.scaleLinear().domain([0, lr0 * 1.05]).range([f.h, 0]);
      gridlines(f, y, 5);
      axes(f, x, y, "epoch", "learning rate", ax => ax.ticks(7), ay => ay.ticks(5));
      const eps = d3.range(0, E + 0.001, 0.2);
      f.g.append("path").datum(eps).attr("fill", "none").attr("stroke", PAL.A).attr("stroke-width", 1.6).attr("d", d3.line().x(e => x(e)).y(e => y(lr(e))));
      [24, 100, 350].forEach(e => { f.g.append("circle").attr("cx", x(e)).attr("cy", y(lr(e))).attr("r", 3.5).attr("fill", PAL.B);
        txt(f.g, x(e) + 5, y(lr(e)) - 6, lr(e).toFixed(4), { size: 9.5, fill: PAL.B, mono: true, anchor: e === 350 ? "end" : "start" }); });
      d3.select("#train-read").html("batch " + bs + " → peak lr 0.016·" + bs + "/256 = <b>" + lr0.toFixed(3) + "</b> (paper's 0.256 ⇔ batch 4096) · 5-epoch warmup, ×0.97 every 2.4 epochs · lr at epoch 350 = " +
        lr(350).toFixed(5) + " (" + (100 * lr(350) / lr0).toFixed(2) + "% of peak)");
    } else {
      const net = EN.released[mi].net, N = net.blocks.length;
      const x = d3.scaleBand().domain(d3.range(N)).range([0, f.w]).padding(0.15), y = d3.scaleLinear().domain([0.75, 1]).range([f.h, 0]);
      gridlines(f, y, 5);
      axes(f, x, y, "block index (" + EN.released[mi].name + ", " + N + " MBConv blocks)", "survival probability", ax => ax.tickValues(d3.range(0, N, Math.ceil(N / 16))), ay => ay.ticks(5));
      let expDrop = 0, nskip = 0;
      net.blocks.forEach((b, i) => {
        const p = 1 - 0.2 * i / N;
        if (b.skip){ expDrop += 1 - p; nskip++; }
        f.g.append("rect").attr("x", x(i)).attr("y", y(p)).attr("width", x.bandwidth()).attr("height", f.h - y(p))
          .attr("fill", b.skip ? PAL.good : XC.dim).attr("fill-opacity", b.skip ? 0.8 : 0.45);
      });
      txt(f.g, 6, 12, "green: skip-connected (droppable) · grey: first block of a stage (never dropped)", { size: 9.5 });
      d3.select("#train-read").html(EN.released[mi].name + ": survival 1 − 0.2·i/" + N + " (last block " + (1 - 0.2 * (N - 1) / N).toFixed(4) + ") · droppable blocks <b>" + nskip + "</b> of " + N +
        " · expected blocks dropped per step <b>" + expDrop.toFixed(2) + "</b>");
    }
  }
  bind(["train-show", "train-bs", "train-model"], draw); draw();
});

/* ───────────────────────── 22 · Table 2 scatter ───────────────────────── */
fig("sc", function(){
  const W = 640, H = 300, m = { l: 52, r: 16, t: 16, b: 42 };
  const eff = [["B0",77.1,5.3,0.39],["B1",79.1,7.8,0.70],["B2",80.1,9.2,1.0],["B3",81.6,12,1.8],["B4",82.9,19,4.2],["B5",83.6,30,9.9],["B6",84.0,43,19],["B7",84.3,66,37]];
  const oth = [["ResNet-50",76.0,26,4.1],["DenseNet-169",76.2,14,3.5],["ResNet-152",77.8,60,11],["DenseNet-264",77.9,34,6.0],["Inception-v3",78.8,24,5.7],
    ["Xception",79.0,23,8.4],["Inception-v4",80.0,48,13],["Inception-ResNet-v2",80.1,56,13],["ResNeXt-101",80.9,84,32],["PolyNet",81.3,92,35],
    ["SENet",82.7,146,42],["NASNet-A",82.7,89,24],["AmoebaNet-A",82.8,87,23],["PNASNet",82.9,86,23],["AmoebaNet-C",83.5,155,41],["GPipe",84.3,557,null]];
  let hov = null;
  function draw(){
    const xk = $v("sc-x") === "F" ? 3 : 2, log = $v("sc-scale") === "log";
    const f = frame("#sc-svg", W, H, m);
    const ext = xk === 2 ? [3, 700] : [0.3, 50];
    const x = log ? d3.scaleLog().domain(ext).range([0, f.w]) : d3.scaleLinear().domain([0, xk === 2 ? 600 : 45]).range([0, f.w]);
    const y = d3.scaleLinear().domain([75.5, 85]).range([f.h, 0]);
    gridlines(f, y, 6);
    axes(f, x, y, (xk === 2 ? "parameters (M)" : "FLOPs (B)") + (log ? ", log" : ""), "ImageNet top-1 (%)", ax => ax.ticks(6, log ? "~g" : null));
    const pts = oth.filter(o => o[xk] != null);
    f.g.selectAll(null).data(pts).enter().append("circle").attr("cx", o => x(o[xk])).attr("cy", o => y(o[1])).attr("r", 4.5)
      .attr("fill", XC.dim).attr("stroke", hov && hov[0] === "o" ? PAL.ink : "none").style("cursor", "pointer")
      .on("mouseenter", (ev, o) => { d3.select("#sc-read").html("<b>" + o[0] + "</b>: " + o[1] + "% top-1, " + o[2] + " M params, " + (o[3] == null ? "FLOPs not given" : o[3] + " B FLOPs")); });
    f.g.append("path").datum(eff).attr("fill", "none").attr("stroke", PAL.good).attr("stroke-width", 1.8).attr("d", d3.line().x(e => x(e[xk])).y(e => y(e[1])));
    f.g.selectAll(null).data(eff).enter().append("circle").attr("cx", e => x(e[xk])).attr("cy", e => y(e[1])).attr("r", 4.5).attr("fill", PAL.good).style("cursor", "pointer")
      .on("mouseenter", (ev, e) => { d3.select("#sc-read").html("<b>EfficientNet-" + e[0] + "</b>: " + e[1] + "% top-1, " + e[2] + " M params, " + e[3] + " B FLOPs"); });
    eff.forEach(e => txt(f.g, x(e[xk]) - 6, y(e[1]) - 7, e[0], { size: 9, fill: PAL.good, anchor: "end" }));
    pts.forEach(o => txt(f.g, x(o[xk]) + 6, y(o[1]) + 3, o[0], { size: 8.5, fill: XC.dim }));
    // summary: for each comparison model, the smallest EfficientNet at ≥ its accuracy, and the ratio
    const rows = pts.map(o => { const e = eff.find(e => e[1] >= o[1]); return e ? o[xk] / e[xk] : null; }).filter(v => v);
    d3.select("#sc-read").html("for each comparison ConvNet, the smallest EfficientNet with ≥ its top-1 uses " + d3.min(rows).toFixed(1) + "–" + d3.max(rows).toFixed(1) + "× fewer " +
      (xk === 2 ? "parameters" : "FLOPs") + " (geometric mean " + Math.exp(d3.mean(rows.map(Math.log))).toFixed(1) + "×) — hover a point");
  }
  bind(["sc-x", "sc-scale"], draw); draw();
});

/* ───────────────────────── 25 · transfer learning ───────────────────────── */
fig("tr", function(){
  const W = 640, H = 300, m = { l: 52, r: 16, t: 20, b: 58 };
  // [dataset, classes, public model, public acc, public params(M), our B, our acc, best model, best acc, best params (M or null), our B, our acc]
  const D = [
    ["CIFAR-10", 10, "NASNet-A", 98.0, 85, 0, 98.1, "GPipe", 99.0, 556, 7, 98.9],
    ["CIFAR-100", 100, "NASNet-A", 87.5, 85, 0, 88.1, "GPipe", 91.3, 556, 7, 91.7],
    ["Birdsnap", 500, "Inception-v4", 81.8, 41, 5, 82.0, "GPipe", 83.6, 556, 7, 84.3],
    ["Cars", 196, "Inception-v4", 93.4, 41, 3, 93.6, "DAT", 94.8, null, 7, 94.7],
    ["Flowers", 102, "Inception-v4", 98.5, 41, 5, 98.5, "DAT", 97.7, null, 7, 98.8],
    ["Aircraft", 100, "Inception-v4", 90.9, 41, 3, 90.7, "DAT", 92.9, null, 7, 92.9],
    ["Pets", 37, "ResNet-152", 94.5, 58, 4, 94.8, "GPipe", 95.9, 556, 6, 95.4],
    ["Food-101", 101, "Inception-v4", 90.8, 41, 4, 91.5, "GPipe", 93.0, 556, 7, 93.0]
  ];
  const params = (b, cls) => { const n = EN.released[b].net; return (n.P - (n.head * 1000 + 1000) + n.head * cls + cls) / 1e6; };
  function draw(){
    const pub = $v("tr-cmp") === "pub", showAcc = $v("tr-show") === "acc";
    const f = frame("#tr-svg", W, H, m);
    const rows = D.map(r => {
      const b = pub ? r[5] : r[10], acc = pub ? r[6] : r[11], cacc = pub ? r[3] : r[8], cp = pub ? r[4] : r[9];
      const p = params(b, r[1]);
      return { k: r[0], b: b, acc: acc, cacc: cacc, cm: pub ? r[2] : r[7], p: p, cp: cp, diff: acc - cacc, ratio: cp ? cp / p : null };
    });
    const x = d3.scaleBand().domain(rows.map(r => r.k)).range([0, f.w]).padding(0.3);
    let y;
    if (showAcc){
      y = d3.scaleLinear().domain([-1.2, 1.2]).range([f.h, 0]);
      gridlines(f, y, 6);
      axes(f, x, y, null, "EfficientNet − comparison (points)", null, ay => ay.ticks(6));
      f.g.append("line").attr("x1", 0).attr("x2", f.w).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", PAL.muted);
      rows.forEach(r => {
        const v = r.diff;
        f.g.append("rect").attr("x", x(r.k)).attr("y", y(Math.max(0, v))).attr("width", x.bandwidth()).attr("height", Math.max(1.5, Math.abs(y(v) - y(0))))
          .attr("fill", Math.abs(v) < 0.05 ? PAL.B : v > 0 ? PAL.good : PAL.bad).attr("fill-opacity", 0.8);
        txt(f.g, x(r.k) + x.bandwidth() / 2, (v >= 0 ? y(v) - 4 : y(v) + 12), (v > 0 ? "+" : "") + v.toFixed(1), { anchor: "middle", size: 9.5, fill: PAL.ink, mono: true });
      });
    } else {
      y = d3.scaleLog().domain([1, 30]).range([f.h, 0]);
      gridlines(f, y, 4);
      axes(f, x, y, null, "comparison params ÷ EfficientNet params", null, ay => ay.ticks(4, "~g"));
      rows.forEach(r => {
        if (!r.ratio){ txt(f.g, x(r.k) + x.bandwidth() / 2, f.h - 6, "n/a", { anchor: "middle", size: 9.5 }); return; }
        f.g.append("rect").attr("x", x(r.k)).attr("y", y(r.ratio)).attr("width", x.bandwidth()).attr("height", f.h - y(r.ratio)).attr("fill", PAL.A).attr("fill-opacity", 0.8);
        txt(f.g, x(r.k) + x.bandwidth() / 2, y(r.ratio) - 4, r.ratio.toFixed(1) + "×", { anchor: "middle", size: 9.5, fill: PAL.ink, mono: true });
      });
    }
    rows.forEach(r => {
      txt(f.g, x(r.k) + x.bandwidth() / 2, f.h + 14, r.k, { anchor: "middle", size: 9.5, fill: PAL.ink });
      txt(f.g, x(r.k) + x.bandwidth() / 2, f.h + 27, "B" + r.b + " " + r.p.toFixed(1) + "M", { anchor: "middle", size: 9, mono: true });
      txt(f.g, x(r.k) + x.bandwidth() / 2, f.h + 40, "vs " + r.cm, { anchor: "middle", size: 8.5 });
    });
    const wins = rows.filter(r => r.diff > 0.05).length, ties = rows.filter(r => Math.abs(r.diff) <= 0.05).length;
    const rat = rows.filter(r => r.ratio).map(r => r.ratio);
    d3.select("#tr-read").html("vs " + (pub ? "best public models" : "best reported results") + ": <b>" + wins + "</b> higher, <b>" + ties + "</b> tied, <b>" + (8 - wins - ties) + "</b> lower · " +
      "parameter ratio with recomputed EfficientNet heads: geometric mean <b>" + Math.exp(d3.mean(rat.map(Math.log))).toFixed(1) + "×</b> over " + rat.length + " datasets with known counts");
  }
  bind(["tr-cmp", "tr-show"], draw); draw();
});

/* ───────────────────────── 28 · activation memory vs resolution ───────────────────────── */
fig("mem", function(){
  const W = 640, H = 300, m = { l: 56, r: 16, t: 16, b: 42 };
  const r50 = EN.resnet50(224);
  function draw(){
    const mi = +$v("mem-model"), gpu = +$v("mem-gpu"), bytes = $v("mem-prec") === "16" ? 2 : 4;
    const R = EN.released[mi];
    const f = frame("#mem-svg", W, H, m);
    const ress = d3.range(128, 801, 16);
    const pts = ress.map(r => ({ r: r, g: EN.build(R.w, R.d, r).act * bytes / Math.pow(2, 30) }));
    const rpts = ress.map(r => ({ r: r, g: EN.resnet50(r).act * bytes / Math.pow(2, 30) }));
    const x = d3.scaleLinear().domain([128, 800]).range([0, f.w]);
    const y = d3.scaleLog().domain([0.005, 20]).range([f.h, 0]);
    gridlines(f, y, 5);
    axes(f, x, y, "input resolution (px)", "activation GiB per image (log)", ax => ax.ticks(8), ay => ay.ticks(5, "~g"));
    f.g.append("path").datum(rpts).attr("fill", "none").attr("stroke", PAL.muted).attr("stroke-width", 1.4).attr("stroke-dasharray", "5,3").attr("d", d3.line().x(p => x(p.r)).y(p => y(p.g)));
    f.g.append("path").datum(pts).attr("fill", "none").attr("stroke", XC.pu).attr("stroke-width", 2).attr("d", d3.line().x(p => x(p.r)).y(p => y(p.g)));
    const here = R.net.act * bytes / Math.pow(2, 30);
    f.g.append("circle").attr("cx", x(R.res)).attr("cy", y(here)).attr("r", 5).attr("fill", XC.pu);
    txt(f.g, x(R.res) + 7, y(here) - 6, R.name + " @ " + R.res, { fill: XC.pu, size: 10 });
    const r50here = r50.act * bytes / Math.pow(2, 30);
    f.g.append("circle").attr("cx", x(224)).attr("cy", y(r50here)).attr("r", 4).attr("fill", PAL.muted);
    txt(f.g, x(224) + 6, y(r50here) + 14, "ResNet-50 @ 224", { size: 9.5 });
    txt(f.g, 6, 12, "purple: " + R.name + "'s network at other input sizes · dashed: ResNet-50 · rough count of stored activations, no workspace", { size: 9.5 });
    const fit = Math.floor(gpu / here);
    d3.select("#mem-read").html(R.name + " @ " + R.res + " px: ~<b>" + here.toFixed(2) + " GiB</b> of " + (bytes === 2 ? "fp16" : "fp32") + " activations per training image (ResNet-50 @ 224: " + r50here.toFixed(2) +
      " GiB) · MACs per stored element: " + (R.net.M / R.net.act).toFixed(1) + " vs ResNet-50's " + (r50.M / r50.act).toFixed(1) +
      " · a " + gpu + " GiB budget for activations holds ≈ <b>" + fit + "</b> images per device");
  }
  bind(["mem-model", "mem-gpu", "mem-prec"], draw); draw();
});

/* ───────────────────────── 29 · Fused-MBConv (EfficientNetV2 Table 3) ───────────────────────── */
fig("fused", function(){
  const W = 640, H = 300, m = { l: 52, r: 16, t: 20, b: 42 };
  // reported: EfficientNet-B4 with MBConv replaced by Fused-MBConv in successive stages
  const rows = [["no fused", 19.3, 4.5, 82.8, 262, 155], ["fused stage 1–3", 20.0, 7.5, 83.1, 362, 216], ["fused stage 1–5", 43.4, 21.3, 83.1, 327, 223], ["fused stage 1–7", 132.0, 34.4, 81.7, 254, 206]];
  const metrics = { tpu: [4, "TPUv3 imgs/sec/core (training)"], gpu: [5, "V100 imgs/sec/gpu (training)"], flops: [2, "FLOPs (B)"], params: [1, "params (M)"], acc: [3, "top-1 (%)"] };
  function draw(){
    const key = $v("fused-metric"), mk = metrics[key];
    const f = frame("#fused-svg", W, H, m);
    const x = d3.scaleBand().domain(rows.map(r => r[0])).range([0, f.w]).padding(0.35);
    const vals = rows.map(r => r[mk[0]]);
    const y = key === "acc" ? d3.scaleLinear().domain([80, 84]).range([f.h, 0]) : d3.scaleLinear().domain([0, d3.max(vals) * 1.15]).range([f.h, 0]);
    gridlines(f, y, 5);
    axes(f, x, y, null, mk[1], null, ay => ay.ticks(5));
    rows.forEach((r, i) => {
      const v = r[mk[0]];
      f.g.append("rect").attr("x", x(r[0])).attr("y", y(v)).attr("width", x.bandwidth()).attr("height", f.h - y(v)).attr("fill", i === 1 ? PAL.good : PAL.A).attr("fill-opacity", 0.8);
      txt(f.g, x(r[0]) + x.bandwidth() / 2, y(v) - 4, String(v), { anchor: "middle", size: 10, fill: PAL.ink, mono: true });
    });
    const tpuPerF = rows.map(r => r[4] * r[2]);   // throughput × FLOPs per image ∝ achieved FLOP rate
    d3.select("#fused-read").html("fused stage 1–3 vs no fused: FLOPs ×" + (7.5 / 4.5).toFixed(2) + " yet TPU throughput ×" + (362 / 262).toFixed(2) + " and V100 ×" + (216 / 155).toFixed(2) +
      " → achieved TPU FLOP rate ×<b>" + (tpuPerF[1] / tpuPerF[0]).toFixed(2) + "</b> · the arithmetic did not get cheaper, the hardware got busier");
  }
  bind(["fused-metric"], draw); draw();
});
