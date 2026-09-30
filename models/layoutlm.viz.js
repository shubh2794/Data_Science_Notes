/* layoutlm.viz.js — every interactive figure on models/layoutlm.html.
   Loaded after data.js / notes.js (palette C comes from notes.js when present).
   Each figure is its own IIFE-style block wrapped in try/catch so one failure cannot blank the rest.
   House rule: every number a figure displays is computed here from a spec or from a data
   array labelled with its source — never typed into a label. */

const LV = (function(){
  const P = Object.assign({}, (typeof C !== "undefined" ? C : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"}),
    { panel:"#15181f", paper:"#e9e4d8", paperInk:"#2b2b2b", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6", grey:"#6b7280" });

  function safe(name, fn){ try { fn(); } catch(e){ if (typeof console !== "undefined") console.error("[layoutlm.viz] "+name+" failed:", e); } }
  function txt(g, x, y, s, o){
    o = o || {};
    return g.append("text").attr("x",x).attr("y",y).attr("font-size",o.size||11).attr("fill",o.fill||P.ink)
      .attr("text-anchor",o.anchor||"start").attr("font-family",o.mono?"SF Mono,Menlo,monospace":null)
      .attr("font-weight",o.bold?600:null).text(s);
  }
  function lcg(seed){ let s = seed >>> 0; return () => { s = (1664525*s + 1013904223) >>> 0; return s/4294967296; }; }
  function comma(x){ return Math.round(x).toLocaleString("en-US"); }
  function fmtM(x){ return (x/1e6).toFixed(1)+"M"; }
  function shuffle(a, r){ const b = a.slice(); for (let i=b.length-1;i>0;i--){ const j = Math.floor(r()*(i+1)); const t=b[i]; b[i]=b[j]; b[j]=t; } return b; }

  /* ── The mock invoice. Page = A4 at 300 dpi, 2480 × 3508 px. Boxes are [x0, y0, x1, y1] in pixels
        at 300 dpi; `lab` is a FUNSD-style entity label, `ent` groups words into entities.
        Illustrative page, authored for this note (not from any dataset). ── */
  const PW = 2480, PH = 3508;
  const WORDS = [
    {t:"ACME",       b:[200,180,520,290],   lab:"header",   ent:"h1"},
    {t:"SUPPLIES",   b:[560,180,1060,290],  lab:"header",   ent:"h1"},
    {t:"Invoice",    b:[1600,200,1850,260], lab:"question", ent:"q1"},
    {t:"No:",        b:[1870,200,1960,260], lab:"question", ent:"q1"},
    {t:"INV-0427",   b:[2000,200,2300,260], lab:"answer",   ent:"a1"},
    {t:"Date:",      b:[1600,300,1760,360], lab:"question", ent:"q2"},
    {t:"2024-03-18", b:[2000,300,2320,360], lab:"answer",   ent:"a2"},
    {t:"Bill",       b:[200,520,300,580],   lab:"question", ent:"q3"},
    {t:"to:",        b:[320,520,400,580],   lab:"question", ent:"q3"},
    {t:"Jane",       b:[200,610,330,670],   lab:"answer",   ent:"a3"},
    {t:"Doe",        b:[350,610,460,670],   lab:"answer",   ent:"a3"},
    {t:"12",         b:[200,690,260,750],   lab:"answer",   ent:"a3"},
    {t:"Elm",        b:[280,690,380,750],   lab:"answer",   ent:"a3"},
    {t:"Street",     b:[400,690,580,750],   lab:"answer",   ent:"a3"},
    {t:"Ship",       b:[1600,520,1720,580], lab:"question", ent:"q4"},
    {t:"to:",        b:[1740,520,1820,580], lab:"question", ent:"q4"},
    {t:"Same",       b:[1600,610,1760,670], lab:"answer",   ent:"a4"},
    {t:"as",         b:[1780,610,1840,670], lab:"answer",   ent:"a4"},
    {t:"billing",    b:[1860,610,2060,670], lab:"answer",   ent:"a4"},
    {t:"Item",       b:[200,1000,340,1060], lab:"header",   ent:"h2"},
    {t:"Qty",        b:[1300,1000,1400,1060], lab:"header", ent:"h3"},
    {t:"Amount",     b:[1900,1000,2120,1060], lab:"header", ent:"h4"},
    {t:"Paper",      b:[200,1120,370,1180], lab:"other",    ent:"o1"},
    {t:"10",         b:[1320,1120,1380,1180], lab:"other",  ent:"o2"},
    {t:"$240.00",    b:[1900,1120,2140,1180], lab:"other",  ent:"o3"},
    {t:"Toner",      b:[200,1220,380,1280], lab:"other",    ent:"o4"},
    {t:"2",          b:[1330,1220,1360,1280], lab:"other",  ent:"o5"},
    {t:"$1,000.00",  b:[1900,1220,2200,1280], lab:"other",  ent:"o6"},
    {t:"Total:",     b:[1650,2900,1850,2960], lab:"question", ent:"q5"},
    {t:"$1,240.00",  b:[1900,2900,2200,2960], lab:"answer",   ent:"a5"},
    {t:"Thank",      b:[200,3300,380,3360], lab:"other",    ent:"o7"},
    {t:"you",        b:[400,3300,500,3360], lab:"other",    ent:"o8"}
  ];
  WORDS.forEach((w,i) => { w.id = i; });
  const LABCOL = {header:P.purple, question:P.A, answer:P.good, other:P.grey};

  /* normalised ids: floor(1000·x / W) */
  function norm(w, W, H){ W = W||PW; H = H||PH; const b = w.b;
    return [Math.floor(1000*b[0]/W), Math.floor(1000*b[1]/H), Math.floor(1000*b[2]/W), Math.floor(1000*b[3]/H)]; }

  /* reading orders */
  function lines(ws){
    const s = ws.slice().sort((a,b) => a.b[1]-b.b[1] || a.b[0]-b.b[0]);
    const L = [];
    s.forEach(w => {
      const cur = L[L.length-1];
      if (cur && Math.abs(w.b[1] - cur.y0) < 40) cur.w.push(w); else L.push({y0:w.b[1], w:[w]});
    });
    L.forEach(l => { l.w.sort((a,b) => a.b[0]-b.b[0]); l.y1 = Math.max.apply(null, l.w.map(w=>w.b[3])); });
    return L;
  }
  function rasterOrder(ws){ const out = []; lines(ws).forEach(l => l.w.forEach(w => out.push(w))); return out; }
  function blockOrder(ws){
    const L = lines(ws), bands = [];
    L.forEach(l => { const b = bands[bands.length-1];
      if (b && l.y0 - b.y1 < 100){ b.lines.push(l); b.y1 = Math.max(b.y1, l.y1); } else bands.push({lines:[l], y1:l.y1}); });
    const out = [];
    bands.forEach(b => {
      const all = []; b.lines.forEach(l => l.w.forEach(w => all.push(w)));
      const iv = all.map(w => [w.b[0], w.b[2]]).sort((a,c) => a[0]-c[0]);
      let reach = iv[0][1], cut = null, best = 150;
      for (let i=1;i<iv.length;i++){ const gap = iv[i][0] - reach; if (gap > best){ best = gap; cut = iv[i][0]; } reach = Math.max(reach, iv[i][1]); }
      if (cut === null){ rasterOrder(all).forEach(w => out.push(w)); }
      else { rasterOrder(all.filter(w => w.b[0] < cut)).forEach(w => out.push(w)); rasterOrder(all.filter(w => w.b[0] >= cut)).forEach(w => out.push(w)); }
    });
    return out;
  }

  /* draw the page into rect (x, y, w, h); opts.stretch draws into a non-uniform rect */
  function drawPage(g, x, y, w, h, o){
    o = o || {};
    g.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",h).attr("fill",P.paper).attr("rx",2);
    const sx = v => x + v/PW*w, sy = v => y + v/PH*h;
    const sel = g.selectAll(null).data(WORDS).enter().append("g").style("cursor", o.click ? "pointer" : null);
    sel.append("rect").attr("x",d=>sx(d.b[0])).attr("y",d=>sy(d.b[1])).attr("width",d=>Math.max(1,sx(d.b[2])-sx(d.b[0])))
      .attr("height",d=>Math.max(1,sy(d.b[3])-sy(d.b[1])))
      .attr("fill", d => o.fill ? o.fill(d) : "none").attr("stroke", d => o.stroke ? o.stroke(d) : "#8a8577").attr("stroke-width", d => o.sw ? o.sw(d) : 0.6);
    sel.append("text").attr("x",d=>sx(d.b[0])+0.5).attr("y",d=>sy(d.b[3])-0.8).attr("font-size", o.fs || 6.5)
      .attr("fill", d => o.tcol ? o.tcol(d) : P.paperInk).text(d => o.label ? o.label(d) : d.t);
    if (o.click) sel.on("click", (ev, d) => o.click(d));
    return {sx, sy};
  }

  /* T5 / LayoutLMv2 relative-position bucket (bidirectional), as in the reference implementation */
  function bucket(r, nb, md){
    let ret = 0; const half = nb >> 1;
    if (r > 0) ret += half;
    const n = Math.abs(r), me = half >> 1;
    if (n < me) return ret + n;
    const v = me + Math.floor(Math.log(n/me) / Math.log(md/me) * (half - me));
    return ret + Math.min(v, half - 1);
  }
  /* Illustrative bias tables (made up, as in the worked example): value falls with |offset| rank */
  function biasIll(kind, b, nb){
    const half = nb >> 1, idx = b % half;
    if (kind === "d1") return 0.5 - 0.1*idx;
    if (kind === "x")  return 0.51 - 0.01*idx;
    return 0.9 - 0.05*idx;
  }
  function poisson(lam, r){ const L = Math.exp(-lam); let k = 0, p = 1; do { k++; p *= r(); } while (p > L); return k - 1; }

  return {P, safe, txt, lcg, comma, fmtM, shuffle, PW, PH, WORDS, LABCOL, norm, lines, rasterOrder, blockOrder, drawPage, bucket, biasIll, poisson};
})();

/* ───────────────────────── 04 · reading order ───────────────────────── */
LV.safe("order", function(){
  const {P, txt, WORDS, drawPage, rasterOrder, blockOrder} = LV;
  const svg = d3.select("#order-svg");
  /* key → value pairs to measure (word ids in WORDS) */
  const PAIRS = [[8,9,"Bill to: → Jane"],[15,16,"Ship to: → Same"],[20,23,"Qty → 10"],[22,24,"Paper → $240.00"],[28,29,"Total: → $1,240.00"]];
  let sel = 29;
  function draw(){
    svg.selectAll("*").remove();
    const mode = d3.select("#order-mode").property("value");
    const ord = mode === "raster" ? rasterOrder(WORDS) : blockOrder(WORDS);
    const pos = {}; ord.forEach((w,i) => { pos[w.id] = i; });
    const g = svg.append("g");
    const px = 10, py = 10, ph = 320, pw = ph * LV.PW / LV.PH;
    const m = drawPage(g, px, py, pw, ph, {fill: d => d.id === sel ? "rgba(255,180,84,.55)" : "rgba(91,156,255,.12)", click: d => { sel = d.id; draw(); }});
    /* path through the stream */
    const pts = ord.map(w => [ (m.sx(w.b[0])+m.sx(w.b[2]))/2, (m.sy(w.b[1])+m.sy(w.b[3]))/2 ]);
    g.append("path").attr("d", d3.line()(pts)).attr("fill","none").attr("stroke",P.bad).attr("stroke-width",0.9).attr("opacity",0.7);
    ord.forEach((w,i) => txt(g, m.sx(w.b[0]), m.sy(w.b[1])-1, String(i), {size:6.5, fill:"#b91c1c", bold:true}));
    /* right panel: the serialised stream */
    const rx = pw + 30;
    txt(g, rx, 22, mode === "raster" ? "serialised stream (line raster)" : "serialised stream (block-aware XY-cut)", {bold:true, fill:P.B});
    let x = rx, y = 42;
    ord.forEach((w,i) => {
      const s = i + ":" + w.t, wd = s.length*5.6 + 8;
      if (x + wd > 632){ x = rx; y += 17; }
      g.append("rect").attr("x",x).attr("y",y-11).attr("width",wd-3).attr("height",14).attr("rx",3)
        .attr("fill", w.id === sel ? "rgba(255,180,84,.35)" : "rgba(255,255,255,.04)").attr("stroke",P.line);
      txt(g, x+3, y, s, {size:9.5, mono:true, fill: w.id === sel ? P.B : P.ink});
      x += wd;
    });
    y += 30;
    txt(g, rx, y, "key → value distance in stream positions", {bold:true, fill:P.B});
    PAIRS.forEach((p,k) => {
      const d = Math.abs(pos[p[1]] - pos[p[0]]);
      txt(g, rx, y + 18 + k*16, p[2], {size:10});
      txt(g, rx + 170, y + 18 + k*16, String(d), {size:10, mono:true, fill: d > 1 ? P.bad : P.good, bold:true});
    });
    const w = WORDS[sel], partner = PAIRS.find(p => p[0] === sel || p[1] === sel);
    const rs = PAIRS.map(p => Math.abs(pos[p[1]] - pos[p[0]]));
    d3.select("#order-read").html(
      `<b>${w.t}</b> is at 1-D index <b>${pos[sel]}</b> of ${ord.length}` +
      (partner ? ` · pair "${partner[2]}" is <b>${Math.abs(pos[partner[1]]-pos[partner[0]])}</b> positions apart` : "") +
      ` · across the five pairs the mean key → value distance is <b>${(rs.reduce((a,b)=>a+b,0)/rs.length).toFixed(1)}</b>` +
      (mode === "block" ? " — the XY-cut keeps the address columns together but cuts the table at its widest gap, reading the item column before the quantity and amount columns, so each item is separated from its amount." : " — the raster order interleaves the Bill-to and Ship-to columns but keeps each table row together."));
  }
  d3.select("#order-mode").on("change", draw);
  draw();
});

/* ───────────────────────── 05 · boxes → ids → lookups ───────────────────────── */
LV.safe("boxes", function(){
  const {P, txt, WORDS, drawPage, comma} = LV;
  const svg = d3.select("#box-svg");
  let sel = 28;
  function draw(){
    svg.selectAll("*").remove();
    const dpi = +d3.select("#box-dpi").property("value"), wh = d3.select("#box-wh").property("checked");
    const f = dpi / 300, W = Math.round(LV.PW*f), H = Math.round(LV.PH*f);
    const g = svg.append("g");
    const ph = 340, pw = ph * LV.PW / LV.PH;
    drawPage(g, 10, 10, pw, ph, {fill: d => d.id === sel ? "rgba(255,180,84,.6)" : "rgba(91,156,255,.10)", click: d => { sel = d.id; draw(); }});
    const w = WORDS[sel], pb = w.b.map(v => Math.round(v*f));
    const n = [Math.floor(1000*pb[0]/W), Math.floor(1000*pb[1]/H), Math.floor(1000*pb[2]/W), Math.floor(1000*pb[3]/H)];
    const wv = n[2]-n[0], hv = n[3]-n[1];
    const rx = pw + 28;
    txt(g, rx, 24, `"${w.t}"`, {bold:true, size:13, fill:P.B});
    txt(g, rx, 44, `page ${W} × ${H} px   pixel box (${pb.join(", ")})`, {size:10, mono:true, fill:P.muted});
    const rows = [["x̂₀", pb[0], W, n[0]], ["ŷ₀", pb[1], H, n[1]], ["x̂₁", pb[2], W, n[2]], ["ŷ₁", pb[3], H, n[3]]];
    rows.forEach((r,i) => txt(g, rx, 64 + i*15, `${r[0]} = floor(1000·${r[1]}/${r[2]}) = floor(${(1000*r[1]/r[2]).toFixed(2)}) = ${r[3]}`, {size:10, mono:true}));
    if (wh) txt(g, rx, 64 + 4*15, `ŵ = ${n[2]} − ${n[0]} = ${wv}    ĥ = ${n[3]} − ${n[1]} = ${hv}`, {size:10, mono:true, fill:P.teal});
    /* tables as tall bars with row markers */
    const tables = [{name:"X", col:P.A, hits:[["x̂₀",n[0]],["x̂₁",n[2]]]}, {name:"Y", col:P.good, hits:[["ŷ₀",n[1]],["ŷ₁",n[3]]]}];
    if (wh){ tables.push({name:"W", col:P.teal, hits:[["ŵ",wv]]}); tables.push({name:"H", col:P.pink, hits:[["ĥ",hv]]}); }
    const ty = 150, th = 170, tw = 26, gap = (632 - rx - tables.length*tw) / Math.max(1, tables.length);
    tables.forEach((t,k) => {
      const x = rx + 10 + k*(tw + gap);
      g.append("rect").attr("x",x).attr("y",ty).attr("width",tw).attr("height",th).attr("fill","rgba(255,255,255,.04)").attr("stroke",t.col);
      txt(g, x + tw/2, ty - 6, `${t.name} [1024 × d]`, {anchor:"middle", size:9.5, fill:t.col, bold:true});
      t.hits.forEach((h,j) => {
        const yy = ty + h[1]/1024*th;
        g.append("line").attr("x1",x-4).attr("x2",x+tw+4).attr("y1",yy).attr("y2",yy).attr("stroke",P.B).attr("stroke-width",2);
        txt(g, x + tw + 6, yy + 3 + (j? 9 : -3), `${h[0]}→row ${h[1]}`, {size:9, mono:true, fill:P.B});
      });
      txt(g, x - 2, ty + th + 12, "0", {size:8, fill:P.muted}); txt(g, x - 2, ty + 8, "", {size:8});
    });
    const terms = `X[${n[0]}] + Y[${n[1]}] + X[${n[2]}] + Y[${n[3]}]` + (wh ? ` + W[${wv}] + H[${hv}]` : "");
    txt(g, rx, ty + th + 26, "layout term = " + terms, {size:9.5, mono:true, fill:P.ink});
    const base300 = LV.norm(w);
    const drift = n.map((v,i) => v - base300[i]);
    d3.select("#box-read").html(
      `<b>${w.t}</b>: ids (${n.join(", ")})` + (wh ? `, width ${wv}, height ${hv}` : "") +
      ` · ${wh ? 6 : 4} lookups from ${wh ? 4 : 2} tables` +
      ` · vs the 300 dpi ids (${base300.join(", ")}) the change is (${drift.join(", ")}), i.e. at most ${Math.max.apply(null, drift.map(Math.abs))} from rounding` +
      ` · layout table parameters at base width: ${comma((wh?4:2)*1024*768)}`);
  }
  d3.select("#box-dpi").on("change", draw);
  d3.select("#box-wh").on("change", draw);
  draw();
});

/* ───────────────────────── 08 · MVLM ───────────────────────── */
LV.safe("mvlm", function(){
  const {P, txt, WORDS, drawPage, lcg, shuffle} = LV;
  const svg = d3.select("#mvlm-svg");
  let seed = 11;
  function draw(){
    svg.selectAll("*").remove();
    const rate = +d3.select("#mvlm-rate").property("value") / 100, v2 = d3.select("#mvlm-v2").property("checked");
    d3.select("#mvlm-rv").text(Math.round(rate*100) + "%");
    const r = lcg(seed);
    const kind = {}, repl = {};
    WORDS.forEach(w => { if (r() < rate){ const u = r(); kind[w.id] = u < 0.8 ? "mask" : (u < 0.9 ? "rand" : "same");
      if (kind[w.id] === "rand"){ let j = Math.floor(r()*WORDS.length); if (j === w.id) j = (j+1) % WORDS.length; repl[w.id] = WORDS[j].t.toLowerCase(); } } });
    const g = svg.append("g");
    const ph = 320, pw = ph * LV.PW / LV.PH;
    drawPage(g, 10, 10, pw, ph, {
      fill: d => kind[d.id] ? (v2 && kind[d.id] ? "#9ca3af" : (kind[d.id]==="mask" ? "rgba(248,113,113,.45)" : kind[d.id]==="rand" ? "rgba(192,132,252,.45)" : "rgba(74,222,128,.45)")) : "none",
      stroke: d => kind[d.id] ? "#b91c1c" : "#8a8577", sw: d => kind[d.id] ? 1.2 : 0.6,
      label: d => kind[d.id] === "mask" ? "[MASK]" : (kind[d.id] === "rand" ? repl[d.id] : d.t),
      tcol: d => kind[d.id] ? "#111" : P.paperInk});
    const rx = pw + 30;
    txt(g, rx, 22, "model input → prediction target", {bold:true, fill:P.B});
    const sel = WORDS.filter(w => kind[w.id]);
    sel.slice(0, 13).forEach((w,i) => {
      const inp = kind[w.id] === "mask" ? "[MASK]" : (kind[w.id] === "rand" ? repl[w.id] : w.t.toLowerCase());
      const col = kind[w.id] === "mask" ? P.bad : (kind[w.id] === "rand" ? P.purple : P.good);
      const n = LV.norm(w);
      txt(g, rx, 42 + i*17, `${inp}`, {size:10, mono:true, fill:col});
      txt(g, rx + 92, 42 + i*17, `(${n.join(",")})`, {size:9.5, mono:true, fill:P.muted});
      txt(g, rx + 232, 42 + i*17, `→ ${w.t.toLowerCase()}`, {size:10, mono:true});
    });
    if (sel.length > 13) txt(g, rx, 42 + 13*17, `… and ${sel.length - 13} more`, {size:10, fill:P.muted});
    if (!sel.length) txt(g, rx, 42, "no words selected this draw — resample", {size:10, fill:P.muted});
    const cnt = k => sel.filter(w => kind[w.id] === k).length;
    const N = 512, e = N*rate;
    d3.select("#mvlm-read").html(
      `this page: <b>${sel.length}</b> of ${WORDS.length} words selected — ${cnt("mask")} [MASK], ${cnt("rand")} random, ${cnt("same")} unchanged; every box kept` +
      ` · expected in a full 512-token window at ${Math.round(rate*100)}%: <b>${e.toFixed(1)}</b> selected ≈ ${(0.8*e).toFixed(1)} / ${(0.1*e).toFixed(1)} / ${(0.1*e).toFixed(1)}` +
      (v2 ? " · v2: selected words' image regions are also blanked (grey)" : ""));
  }
  d3.select("#mvlm-rate").on("input", draw);
  d3.select("#mvlm-v2").on("change", draw);
  d3.select("#mvlm-new").on("click", () => { seed += 1; draw(); });
  draw();
});

/* ───────────────────────── 12 · the v2 visual grid ───────────────────────── */
LV.safe("grid", function(){
  const {P, txt, WORDS, drawPage} = LV;
  const svg = d3.select("#grid-svg");
  let cell = [5, 4];
  function draw(){
    svg.selectAll("*").remove();
    const k = +d3.select("#grid-k").property("value");
    d3.select("#grid-kv").text(k + " × " + k);
    if (cell[0] >= k || cell[1] >= k) cell = [Math.min(cell[0], k-1), Math.min(cell[1], k-1)];
    const g = svg.append("g");
    const ph = 320, pw = ph * LV.PW / LV.PH, x0 = 10, y0 = 10;
    const bnd = d3.range(k+1).map(i => Math.floor(1000*i/k));
    const r = cell[0], c = cell[1];
    const box = [bnd[c], bnd[r], bnd[c+1], bnd[r+1]];
    const inCell = w => { const n = LV.norm(w); return n[2] > box[0] && n[0] < box[2] && n[3] > box[1] && n[1] < box[3]; };
    drawPage(g, x0, y0, pw, ph, {fill: d => inCell(d) ? "rgba(255,180,84,.5)" : "none"});
    for (let i=0;i<k;i++) for (let j=0;j<k;j++){
      g.append("rect").attr("x", x0 + bnd[j]/1000*pw).attr("y", y0 + bnd[i]/1000*ph)
        .attr("width", (bnd[j+1]-bnd[j])/1000*pw).attr("height", (bnd[i+1]-bnd[i])/1000*ph)
        .attr("fill", i===r && j===c ? "rgba(255,180,84,.18)" : "rgba(91,156,255,.03)")
        .attr("stroke", i===r && j===c ? P.B : "rgba(91,156,255,.7)").attr("stroke-width", i===r && j===c ? 2 : 0.6)
        .style("cursor","pointer").on("click", () => { cell = [i, j]; draw(); });
    }
    const rx = pw + 30;
    const p2 = 56 / k, px224 = 224 / k;
    txt(g, rx, 22, "shape arithmetic (Transformers implementation)", {bold:true, fill:P.B});
    const lines = [
      "224 × 224 × 3  →  FPN P2: 56 × 56 × 256",
      `AdaptiveAvgPool → ${k} × ${k} × 256  →  ${k*k} tokens`,
      `each token ≈ ${p2.toFixed(2)} × ${p2.toFixed(2)} P2 cells`,
      `           ≈ ${px224.toFixed(1)} × ${px224.toFixed(1)} px of the 224 image`,
      `           ≈ ${(LV.PW/k).toFixed(0)} × ${(LV.PH/k).toFixed(0)} px of the A4 scan`,
      `boundaries: ${bnd.length <= 9 ? bnd.join(", ") : bnd.slice(0,5).join(", ") + ", …, " + bnd[bnd.length-1]}`
    ];
    lines.forEach((s,i) => txt(g, rx, 44 + i*17, s, {size:10, mono:true}));
    txt(g, rx, 160, `selected token (row ${r}, col ${c})`, {bold:true, fill:P.B});
    txt(g, rx, 178, `box (${box.join(", ")})`, {size:10.5, mono:true});
    const ws = WORDS.filter(inCell).map(w => w.t);
    txt(g, rx, 196, "words under it: " + (ws.length ? ws.slice(0,6).join(" ") + (ws.length > 6 ? " …" : "") : "(none — whitespace)"), {size:10});
    const seq = 512 + k*k;
    txt(g, rx, 226, `sequence with 512 text tokens: ${seq}`, {size:10.5, mono:true, fill:P.teal});
    txt(g, rx, 244, `attention scores per head per layer: ${(seq*seq).toLocaleString("en-US")}`, {size:10.5, mono:true, fill:P.teal});
    d3.select("#grid-read").html(`pool ${k} × ${k} → <b>${k*k}</b> visual tokens · selected cell (${r}, ${c}) has box (${box.join(", ")}) and covers ${ws.length} word(s)` +
      (k === 7 ? " · 7 × 7 is the released configuration" : " · the released configuration uses 7 × 7") + (56 % k ? ` · 56 is not divisible by ${k}, so adaptive pooling uses unequal, overlapping bins` : ""));
  }
  d3.select("#grid-k").on("input", draw);
  draw();
});

/* ───────────────────────── 14a · the bucketing curve ───────────────────────── */
LV.safe("bucket", function(){
  const {P, txt, bucket} = LV;
  const svg = d3.select("#bk-svg");
  const W = 640, H = 280, m = {l:44, r:16, t:16, b:36};
  const x = d3.scaleLinear().domain([-600, 600]).range([m.l, W-m.r]);
  const y = d3.scaleLinear().domain([0, 63]).range([H-m.b, m.t]);
  function draw(){
    svg.selectAll("*").remove();
    const rr = +d3.select("#bk-r").property("value");
    d3.select("#bk-rv").text(rr);
    const g = svg.append("g");
    g.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).ticks(12)).selectAll("text").attr("fill",P.muted);
    g.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(8)).selectAll("text").attr("fill",P.muted);
    g.selectAll(".domain, .tick line").attr("stroke",P.line);
    txt(g, W/2, H-4, "relative offset r (tokens for 1-D, grid units for 2-D)", {anchor:"middle", size:10.5, fill:P.muted});
    txt(g, 12, m.t+4, "bucket", {size:10.5, fill:P.muted});
    const xs = d3.range(-600, 601);
    const l1 = xs.map(r => [x(r), y(bucket(r, 32, 128))]), l2 = xs.map(r => [x(r), y(bucket(r, 64, 256))]);
    g.append("path").attr("d", d3.line().curve(d3.curveStepAfter)(l2)).attr("fill","none").attr("stroke",P.A).attr("stroke-width",1.6);
    g.append("path").attr("d", d3.line().curve(d3.curveStepAfter)(l1)).attr("fill","none").attr("stroke",P.B).attr("stroke-width",1.6);
    [[-128,P.B],[128,P.B],[-256,P.A],[256,P.A]].forEach(v => g.append("line").attr("x1",x(v[0])).attr("x2",x(v[0])).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",v[1]).attr("stroke-dasharray","3,3").attr("opacity",0.5));
    g.append("line").attr("x1",x(rr)).attr("x2",x(rr)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.ink).attr("stroke-width",1);
    const b1 = bucket(rr, 32, 128), b2 = bucket(rr, 64, 256);
    g.append("circle").attr("cx",x(rr)).attr("cy",y(b1)).attr("r",4).attr("fill",P.B);
    g.append("circle").attr("cx",x(rr)).attr("cy",y(b2)).attr("r",4).attr("fill",P.A);
    txt(g, m.l+8, m.t+12, "2-D: 64 buckets, max distance 256", {size:10, fill:P.A});
    txt(g, m.l+8, m.t+26, "1-D: 32 buckets, max distance 128", {size:10, fill:P.B});
    const exact2 = Math.abs(rr) < 16, exact1 = Math.abs(rr) < 8;
    /* "saturated" = the last bucket of its sign (15 or 31 for 1-D, 31 or 63 for 2-D), reached before D because of the floor */
    const sat1 = (b1 % 16) === 15, sat2 = (b2 % 32) === 31;
    d3.select("#bk-read").html(`r = <b>${rr}</b> → 1-D bucket <b>${b1}</b> (${exact1 ? "exact" : (sat1 ? "saturated" : "log-spaced")}) · 2-D bucket <b>${b2}</b> (${exact2 ? "exact" : (sat2 ? "saturated" : "log-spaced")})` +
      ` · offsets sharing this 2-D bucket: ${(() => { const s = d3.range(-1000, 1001).filter(v => bucket(v,64,256) === b2); return s[0] + " … " + s[s.length-1] + " (" + s.length + " values)"; })()}`);
  }
  d3.select("#bk-r").on("input", draw);
  draw();
});

/* ───────────────────────── 14b · pairwise bucket heatmap ───────────────────────── */
LV.safe("attn", function(){
  const {P, txt, WORDS, rasterOrder, norm, bucket, biasIll} = LV;
  const svg = d3.select("#attn-svg"), ksel = d3.select("#attn-key");
  const ord = rasterOrder(WORDS), N = ord.length;
  ord.forEach((w,i) => ksel.append("option").attr("value", i).text(i + " · " + w.t));
  const iTotal = ord.findIndex(w => w.t === "Total:");
  ksel.property("value", iTotal);
  let key = iTotal + 1;
  /* implementation anchors: x̂₀ and ŷ₁ */
  const ax = ord.map(w => norm(w)[0]), ay = ord.map(w => norm(w)[3]);
  function comps(i, j){
    const b1 = bucket(j - i, 32, 128), bx = bucket(ax[j] - ax[i], 64, 256), by = bucket(ay[j] - ay[i], 64, 256);
    return {b1, bx, by, sum: biasIll("d1", b1, 32) + biasIll("x", bx, 64) + biasIll("y", by, 64)};
  }
  function draw(){
    svg.selectAll("*").remove();
    const comp = d3.select("#attn-comp").property("value"), q = +ksel.property("value");
    const g = svg.append("g"), cs = 10, ox = 70, oy = 60;
    const val = (i,j) => { const c = comps(i,j); return comp === "x" ? c.bx : comp === "y" ? c.by : comp === "d1" ? c.b1 : c.sum; };
    const dom = comp === "sum" ? [-1.5, 2.3] : comp === "d1" ? [0, 31] : [0, 63];
    const col = d3.scaleSequential(d3.interpolateViridis).domain(dom);
    for (let i=0;i<N;i++) for (let j=0;j<N;j++){
      g.append("rect").attr("x", ox + j*cs).attr("y", oy + i*cs).attr("width", cs-0.5).attr("height", cs-0.5)
        .attr("fill", col(val(i,j))).attr("stroke", (i===q && j===key) ? "#fff" : null).attr("stroke-width", 1.5)
        .style("cursor","pointer").on("click", () => { ksel.property("value", i); key = j; draw(); });
    }
    g.append("rect").attr("x", ox-1).attr("y", oy + q*cs - 1).attr("width", N*cs+1).attr("height", cs+1).attr("fill","none").attr("stroke",P.B).attr("stroke-width",1.2);
    ord.forEach((w,i) => { txt(g, ox-4, oy + i*cs + 8, w.t.slice(0,9), {size:7, anchor:"end", fill: i===q ? P.B : P.muted}); });
    ord.forEach((w,j) => { g.append("text").attr("transform",`translate(${ox + j*cs + 7},${oy-4}) rotate(-60)`).attr("font-size",7).attr("fill", j===key ? P.B : P.muted).text(w.t.slice(0,9)); });
    txt(g, ox, 12, "rows: query i · columns: key j (raster order)", {size:10, fill:P.muted});
    /* legend */
    const lx = ox + N*cs + 20;
    d3.range(0, 11).forEach(k => { const v = dom[0] + (dom[1]-dom[0])*k/10;
      g.append("rect").attr("x", lx).attr("y", oy + 200 - k*16).attr("width", 14).attr("height", 16).attr("fill", col(v));
      if (k % 5 === 0) txt(g, lx + 18, oy + 212 - k*16, comp === "sum" ? v.toFixed(1) : String(Math.round(v)), {size:9, fill:P.muted}); });
    const c = comps(q, key);
    const wi = ord[q], wj = ord[key];
    txt(g, lx, oy + 240, `query  ${wi.t}`, {size:10, fill:P.B});
    txt(g, lx, oy + 256, `key    ${wj.t}`, {size:10, fill:P.B});
    txt(g, lx, oy + 276, `Δ1D ${key - q}  → b ${c.b1}`, {size:10, mono:true});
    txt(g, lx, oy + 292, `Δx ${ax[key]-ax[q]}  → b ${c.bx}`, {size:10, mono:true});
    txt(g, lx, oy + 308, `Δy ${ay[key]-ay[q]}  → b ${c.by}`, {size:10, mono:true});
    d3.select("#attn-read").html(`query <b>${wi.t}</b> (x̂₀ ${ax[q]}, ŷ₁ ${ay[q]}) → key <b>${wj.t}</b> (x̂₀ ${ax[key]}, ŷ₁ ${ay[key]}): ` +
      `1-D offset ${key-q} → bucket <b>${c.b1}</b>; Δx ${ax[key]-ax[q]} → bucket <b>${c.bx}</b>; Δy ${ay[key]-ay[q]} → bucket <b>${c.by}</b>` +
      ` · illustrative bias ${biasIll("d1",c.b1,32).toFixed(2)} + ${biasIll("x",c.bx,64).toFixed(2)} + ${biasIll("y",c.by,64).toFixed(2)} = <b>${c.sum.toFixed(2)}</b> added to the logit`);
  }
  d3.select("#attn-comp").on("change", draw);
  ksel.on("change", () => { draw(); });
  draw();
});

/* ───────────────────────── 15 · TIA / TIM ───────────────────────── */
LV.safe("tia", function(){
  const {P, txt, WORDS, drawPage, lines, lcg, shuffle} = LV;
  const svg = d3.select("#tia-svg");
  const L = lines(WORDS);
  let seed = 5;
  function draw(){
    svg.selectAll("*").remove();
    const rate = +d3.select("#tia-rate").property("value") / 100, neg = d3.select("#tia-tim").property("value") === "neg";
    d3.select("#tia-rv").text(Math.round(rate*100) + "%");
    const r = lcg(seed);
    const nCov = Math.round(rate * L.length);
    const covLines = new Set(shuffle(d3.range(L.length), r).slice(0, nCov));
    const covered = {}; L.forEach((l,i) => l.w.forEach(w => { covered[w.id] = covLines.has(i); }));
    const nMask = Math.round(0.15 * WORDS.length);
    const masked = new Set(shuffle(WORDS.map(w=>w.id), r).slice(0, nMask));
    const label = w => masked.has(w.id) ? "excluded" : (neg ? "Covered" : (covered[w.id] ? "Covered" : "Not covered"));
    const g = svg.append("g");
    const ph = 320, pw = ph * LV.PW / LV.PH;
    const m = drawPage(g, 10, 10, pw, ph, {
      fill: d => masked.has(d.id) ? "rgba(156,163,175,.8)" : "none",
      label: d => masked.has(d.id) ? "[MASK]" : d.t});
    if (!neg) L.forEach((l,i) => { if (!covLines.has(i)) return;
      const x0 = Math.min.apply(null, l.w.map(w=>w.b[0])), x1 = Math.max.apply(null, l.w.map(w=>w.b[2]));
      g.append("rect").attr("x", m.sx(x0)-1).attr("y", m.sy(l.y0)-1).attr("width", m.sx(x1)-m.sx(x0)+2).attr("height", m.sy(l.y1)-m.sy(l.y0)+2).attr("fill","#1f2937"); });
    else { g.append("rect").attr("x",10).attr("y",10).attr("width",pw).attr("height",ph).attr("fill","rgba(15,17,23,.55)");
      txt(g, 10 + pw/2, 10 + ph/2, "image from another page", {anchor:"middle", size:11, fill:"#fff", bold:true}); }
    const rx = pw + 30;
    txt(g, rx, 22, neg ? "TIM negative: all TIA labels = Covered" : "TIA labels per token", {bold:true, fill:P.B});
    const colOf = s => s === "Covered" ? P.bad : s === "Not covered" ? P.good : P.muted;
    let x = rx, y = 42;
    WORDS.forEach(w => { const s = label(w), t = w.t + " · " + (s === "Not covered" ? "N" : s === "Covered" ? "C" : "—"), wd = t.length*5.4 + 8;
      if (x + wd > 632){ x = rx; y += 16; }
      txt(g, x, y, t, {size:9.5, mono:true, fill:colOf(s)}); x += wd; });
    const cnt = s => WORDS.filter(w => label(w) === s).length;
    txt(g, rx, y + 26, "C = Covered · N = Not covered · — = MVLM-masked, excluded from the TIA loss", {size:9.5, fill:P.muted});
    d3.select("#tia-read").html(`${L.length} OCR lines, <b>${nCov}</b> covered (${Math.round(rate*100)}%) · ${nMask} words MVLM-masked and excluded · labels: <b>${cnt("Covered")}</b> Covered, <b>${cnt("Not covered")}</b> Not covered` +
      (neg ? " · TIM target for this sample: Not matched" : " · TIM target for this sample: Matched") +
      ` · in pretraining 15% of images are replaced and 5% dropped, so ${100-15-5}% of samples are TIM positives`);
  }
  d3.select("#tia-rate").on("input", draw);
  d3.select("#tia-tim").on("change", draw);
  d3.select("#tia-new").on("click", () => { seed += 1; draw(); });
  draw();
});

/* ───────────────────────── 18 · v3 patchify, MIM, WPA ───────────────────────── */
LV.safe("patch", function(){
  const {P, txt, WORDS, norm, lcg, shuffle} = LV;
  const svg = d3.select("#patch-svg");
  const G = 14, IMG = 224, PS = 16;
  let seed = 3, sel = 28;
  /* BEiT-style blockwise masking */
  function blockwise(target, r){
    const m = d3.range(G).map(() => new Array(G).fill(false));
    let count = 0, stall = 0;
    while (count < target && stall < 50){
      const maxP = Math.max(16, target - count);
      const s = 16 + r() * (maxP - 16);
      const logA = Math.log(0.3) + r() * (Math.log(1/0.3) - Math.log(0.3)), a = Math.exp(logA);
      const h = Math.round(Math.sqrt(s*a)), w = Math.round(Math.sqrt(s/a));
      let added = 0;
      if (h >= 1 && w >= 1 && h < G && w < G){
        const t = Math.floor(r() * (G - h + 1)), l = Math.floor(r() * (G - w + 1));
        for (let i=t;i<t+h;i++) for (let j=l;j<l+w;j++){ if (!m[i][j] && count + added < target){ m[i][j] = true; added++; } }
      }
      count += added; stall = added ? 0 : stall + 1;
    }
    return {m, count};
  }
  function patchesOf(w){
    const n = norm(w);
    const c0 = Math.floor(n[0]/1000*IMG/PS), c1 = Math.min(G-1, Math.floor(n[2]/1000*IMG/PS));
    const r0 = Math.floor(n[1]/1000*IMG/PS), r1 = Math.min(G-1, Math.floor(n[3]/1000*IMG/PS));
    const out = []; for (let i=r0;i<=r1;i++) for (let j=c0;j<=c1;j++) out.push([i,j]);
    return out;
  }
  function draw(){
    svg.selectAll("*").remove();
    const ratio = +d3.select("#patch-ratio").property("value") / 100, tr = +d3.select("#patch-tratio").property("value") / 100;
    d3.select("#patch-rv").text(Math.round(ratio*100) + "%"); d3.select("#patch-tv").text(Math.round(tr*100) + "%");
    const r = lcg(seed);
    const target = Math.round(ratio * G * G);
    const {m, count} = blockwise(target, r);
    const tMask = new Set(shuffle(WORDS.map(w=>w.id), r).slice(0, Math.round(tr * WORDS.length)));
    const lab = w => tMask.has(w.id) ? "masked" : (patchesOf(w).some(p => m[p[0]][p[1]]) ? "unaligned" : "aligned");
    const g = svg.append("g"), S = 330, ox = 10, oy = 14, cs = S / G;
    g.append("rect").attr("x",ox).attr("y",oy).attr("width",S).attr("height",S).attr("fill",LV.P.paper);
    const sx = v => ox + v/LV.PW*S, sy = v => oy + v/LV.PH*S;
    WORDS.forEach(w => {
      const L = lab(w);
      g.append("rect").attr("x",sx(w.b[0])).attr("y",sy(w.b[1])).attr("width",Math.max(1,sx(w.b[2])-sx(w.b[0]))).attr("height",Math.max(1,sy(w.b[3])-sy(w.b[1])))
        .attr("fill", L === "aligned" ? "rgba(74,222,128,.55)" : L === "unaligned" ? "rgba(248,113,113,.6)" : "rgba(156,163,175,.8)")
        .attr("stroke", w.id === sel ? "#000" : "none").attr("stroke-width", 1.4).style("cursor","pointer").on("click", () => { sel = w.id; draw(); });
      g.append("text").attr("x",sx(w.b[0])+0.5).attr("y",sy(w.b[3])-0.5).attr("font-size",5.5).attr("fill","#111").text(w.t).style("pointer-events","none");
    });
    for (let i=0;i<G;i++) for (let j=0;j<G;j++){
      g.append("rect").attr("x",ox + j*cs).attr("y",oy + i*cs).attr("width",cs).attr("height",cs)
        .attr("fill", m[i][j] ? "rgba(17,24,39,.55)" : "none").attr("stroke","rgba(91,156,255,.45)").attr("stroke-width",0.5).style("pointer-events","none");
    }
    const ps = patchesOf(WORDS[sel]);
    ps.forEach(p => g.append("rect").attr("x",ox + p[1]*cs).attr("y",oy + p[0]*cs).attr("width",cs).attr("height",cs).attr("fill","none").attr("stroke",P.B).attr("stroke-width",2).style("pointer-events","none"));
    const rx = ox + S + 22;
    txt(g, rx, 24, `224 × 224, P = 16 → ${G} × ${G} = ${G*G} patches`, {bold:true, fill:P.B});
    txt(g, rx, 42, `(+1 visual [CLS] → ${G*G+1} image tokens)`, {size:10, fill:P.muted});
    txt(g, rx, 64, `MIM: ${count} of ${G*G} patches masked (${(100*count/(G*G)).toFixed(1)}%)`, {size:10.5, mono:true});
    txt(g, rx, 82, `MLM: ${tMask.size} of ${WORDS.length} words masked`, {size:10.5, mono:true});
    const c = s => WORDS.filter(w => lab(w) === s).length;
    txt(g, rx, 104, `WPA labels: ${c("aligned")} aligned`, {size:10.5, mono:true, fill:P.good});
    txt(g, rx, 120, `            ${c("unaligned")} unaligned`, {size:10.5, mono:true, fill:P.bad});
    txt(g, rx, 136, `            ${c("masked")} excluded (text masked)`, {size:10.5, mono:true, fill:P.muted});
    const w = WORDS[sel], n = norm(w);
    txt(g, rx, 166, `"${w.t}"  ids (${n.join(", ")})`, {size:10.5, bold:true, fill:P.B});
    txt(g, rx, 184, `in 224 px: x ${(n[0]*IMG/1000).toFixed(2)}–${(n[2]*IMG/1000).toFixed(2)}, y ${(n[1]*IMG/1000).toFixed(2)}–${(n[3]*IMG/1000).toFixed(2)}`, {size:10, mono:true});
    txt(g, rx, 202, "patches (row, col): " + ps.map(p => `(${p[0]},${p[1]})`).join(" "), {size:10, mono:true});
    txt(g, rx, 220, "label: " + lab(w), {size:10.5, mono:true, fill: lab(w)==="aligned" ? P.good : lab(w)==="unaligned" ? P.bad : P.muted});
    txt(g, rx, 250, "MIM target per masked patch: one of 8,192", {size:10, fill:P.muted});
    txt(g, rx, 266, "codes from the dVAE tokenizer (112 px input,", {size:10, fill:P.muted});
    txt(g, rx, 282, "downsampling 8 → 14 × 14 codes)", {size:10, fill:P.muted});
    d3.select("#patch-read").html(`image masking reached <b>${count}</b> / ${G*G} patches (target ${target}) · text masked ${tMask.size} words · WPA: ${c("aligned")} aligned, ${c("unaligned")} unaligned, ${c("masked")} excluded` +
      ` · selected <b>${w.t}</b> overlaps ${ps.length} patch(es) → <b>${lab(w)}</b>`);
  }
  d3.select("#patch-ratio").on("input", draw);
  d3.select("#patch-tratio").on("input", draw);
  d3.select("#patch-new").on("click", () => { seed += 1; draw(); });
  draw();
});

/* ───────────────────────── 19 · span masking with Poisson lengths ───────────────────────── */
LV.safe("span", function(){
  const {P, txt, lcg, poisson} = LV;
  const svg = d3.select("#span-svg");
  let seed = 7;
  function draw(){
    svg.selectAll("*").remove();
    const lam = +d3.select("#span-lam").property("value"), ratio = +d3.select("#span-ratio").property("value") / 100;
    d3.select("#span-lv").text(lam.toFixed(1)); d3.select("#span-rv").text(Math.round(ratio*100) + "%");
    const N = 512, target = Math.round(ratio * N), r = lcg(seed);
    const mask = new Array(N).fill(false), lens = [];
    let count = 0, tries = 0;
    while (count < target && tries < 5000){
      tries++;
      let L = poisson(lam, r); let redraw = 0; while (L === 0 && redraw < 50){ L = poisson(lam, r); redraw++; } if (L === 0) L = 1;
      L = Math.min(L, target - count);
      const s = Math.floor(r() * (N - L + 1));
      let free = true; for (let i=s;i<s+L;i++) if (mask[i]) { free = false; break; }
      if (!free) continue;
      for (let i=s;i<s+L;i++) mask[i] = true;
      count += L; lens.push(L);
    }
    const g = svg.append("g"), cols = 64, cw = 9, ch = 9, ox = 10, oy = 14;
    for (let i=0;i<N;i++){
      g.append("rect").attr("x", ox + (i%cols)*cw).attr("y", oy + Math.floor(i/cols)*ch).attr("width",cw-1).attr("height",ch-1)
        .attr("fill", mask[i] ? P.bad : "rgba(255,255,255,.08)");
    }
    /* histogram vs Poisson pmf over k ≥ 1 */
    const maxK = 12, hy = 110, hh = 150, hx = 60, hw = 540;
    const hist = d3.range(1, maxK+1).map(k => lens.filter(l => l === k).length / Math.max(1, lens.length));
    const pmf = k => Math.exp(-lam) * Math.pow(lam, k) / d3.range(1, k+1).reduce((a,b)=>a*b, 1);
    const Z = 1 - Math.exp(-lam);
    const pm = d3.range(1, maxK+1).map(k => pmf(k) / Z);
    const yMax = Math.max(0.05, d3.max(hist), d3.max(pm));
    const x = d3.scaleBand().domain(d3.range(1, maxK+1)).range([hx, hx+hw]).padding(0.2);
    const y = d3.scaleLinear().domain([0, yMax]).range([hy+hh, hy+20]);
    g.append("g").attr("transform",`translate(0,${hy+hh})`).call(d3.axisBottom(x)).selectAll("text").attr("fill",P.muted);
    g.append("g").attr("transform",`translate(${hx},0)`).call(d3.axisLeft(y).ticks(4)).selectAll("text").attr("fill",P.muted);
    g.selectAll(".domain, .tick line").attr("stroke",P.line);
    hist.forEach((v,i) => g.append("rect").attr("x", x(i+1)).attr("y", y(v)).attr("width", x.bandwidth()).attr("height", y(0)-y(v)).attr("fill",P.A).attr("opacity",0.8));
    pm.forEach((v,i) => g.append("circle").attr("cx", x(i+1) + x.bandwidth()/2).attr("cy", y(v)).attr("r",3.5).attr("fill",P.B));
    txt(g, hx + hw, hy + 30, "bars: sampled span lengths · dots: Poisson(λ) given ≥ 1", {anchor:"end", size:10, fill:P.muted});
    txt(g, hx + hw/2, hy + hh + 30, "span length", {anchor:"middle", size:10, fill:P.muted});
    const meanL = lens.length ? lens.reduce((a,b)=>a+b,0)/lens.length : 0;
    d3.select("#span-read").html(`masked <b>${count}</b> of ${N} tokens (target ${target}) in <b>${lens.length}</b> spans · mean sampled span ${meanL.toFixed(2)}` +
      ` · Poisson(${lam.toFixed(1)}) conditioned on ≥ 1 has mean ${(lam / Z).toFixed(2)} · ${N - count} tokens stay visible and would receive WPA labels`);
  }
  d3.select("#span-lam").on("input", draw);
  d3.select("#span-ratio").on("input", draw);
  d3.select("#span-new").on("click", () => { seed += 1; draw(); });
  draw();
});

/* ───────────────────────── 21 · parameter calculator ───────────────────────── */
LV.safe("params", function(){
  const {P, txt, comma, fmtM} = LV;
  const svg = d3.select("#par-svg"), sel = d3.select("#par-model");
  /* ── Source: released config.json files on the Hugging Face Hub (microsoft/layoutlm-*-uncased,
        layoutlmv2-*-uncased, layoutxlm-base, layoutlmv3-*); paper figures from each paper's text/tables.
        v1 "paper design" = the two X/Y tables the v1 paper describes; "implementation" = X, Y, H, W. ── */
  const BACKBONE = 86539456 + 3344384;   // ResNeXt-101 32x8d conv weights + FPN (lateral + output convs)
  const M = [
    {id:"v1b-paper", name:"LayoutLM-base (paper: 2 tables)", fam:"v1", V:30522, P:512, T:2, d:768, L:12, tables:2, paper:113e6},
    {id:"v1b-impl",  name:"LayoutLM-base (implementation: 4 tables)", fam:"v1", V:30522, P:512, T:2, d:768, L:12, tables:4, paper:113e6},
    {id:"v1l-paper", name:"LayoutLM-large (paper: 2 tables)", fam:"v1", V:30522, P:512, T:2, d:1024, L:24, tables:2, paper:343e6},
    {id:"v1l-impl",  name:"LayoutLM-large (implementation: 4 tables)", fam:"v1", V:30522, P:512, T:2, d:1024, L:24, tables:4, paper:343e6},
    {id:"v2b", name:"LayoutLMv2-base", fam:"v2", V:30522, P:512, T:2, d:768, L:12, h:12, c:128, s:128, fastqkv:true, seg:false, rel:true, paper:200e6},
    {id:"v2l", name:"LayoutLMv2-large", fam:"v2", V:30522, P:512, T:2, d:1024, L:24, h:16, c:171, s:170, fastqkv:false, seg:false, rel:true, paper:426e6},
    {id:"xlmb", name:"LayoutXLM-base", fam:"v2", V:250002, P:514, T:1, d:768, L:12, h:12, c:128, s:128, fastqkv:false, seg:true, rel:false, paper:345e6},
    {id:"v3b", name:"LayoutLMv3-base", fam:"v3", V:50265, P:514, T:1, d:768, L:12, h:12, c:128, s:128, paper:133e6},
    {id:"v3l", name:"LayoutLMv3-large", fam:"v3", V:50265, P:514, T:1, d:1024, L:24, h:16, c:171, s:170, paper:368e6}
  ];
  M.forEach((m,i) => sel.append("option").attr("value", i).text(m.name));
  sel.property("value", 7);
  function groups(m, heads){
    const d = m.d, R = 1024, layer = 12*d*d + 13*d;
    const G = [];
    G.push(["word embeddings", m.V*d, P.A]);
    G.push(["1-D pos + segment + LN", m.P*d + m.T*d + 2*d, P.teal]);
    if (m.fam === "v1"){
      G.push(["2-D layout tables", m.tables*R*d, P.B]);
      G.push(["encoder layers", m.L*layer, P.purple]);
      G.push(["pooler", d*d + d, P.grey]);
      if (heads) G.push(["MLM head (transform + bias; decoder tied)", d*d + d + 2*d + m.V, P.pink]);
    } else if (m.fam === "v2"){
      G.push(["2-D layout slices", R*(2*m.c + 2*m.s), P.B]);
      G.push(["encoder layers", m.L*layer - (m.fastqkv ? m.L*d : 0), P.purple]);
      if (m.rel) G.push(["relative bias tables", 32*m.h + 2*64*m.h, P.bad]);
      G.push(["pooler", d*d + d, P.grey]);
      G.push(["ResNeXt-101-FPN backbone", BACKBONE, P.good]);
      G.push(["visual proj + LN" + (m.seg ? " + segment" : ""), 256*d + d + 2*d + (m.seg ? d : 0), "#a3e635"]);
    } else {
      const Mp = 196;
      G.push(["2-D layout slices", R*(2*m.c + 2*m.s), P.B]);
      G.push(["encoder layers", m.L*layer, P.purple]);
      G.push(["relative bias tables", 32*m.h + 2*64*m.h, P.bad]);
      G.push(["patch embed + [CLS] + pos + norms", 3*16*16*d + d + d + (Mp+1)*d + 2*d + 2*d, P.good]);
      if (heads){ G.push(["MIM head (inferred shape)", d*d + d + 2*d + d*8192 + 8192, P.pink]); G.push(["WPA head (inferred shape)", d*d + d + 2*d + 2, "#fda4af"]); }
    }
    return G;
  }
  function draw(){
    svg.selectAll("*").remove();
    const m = M[+sel.property("value")], heads = d3.select("#par-heads").property("checked");
    const G = groups(m, heads), total = d3.sum(G, g => g[1]);
    const g = svg.append("g"), x0 = 20, x1 = 620, maxV = Math.max(total, m.paper) * 1.05;
    const x = d3.scaleLinear().domain([0, maxV]).range([x0, x1]);
    let acc = 0;
    G.forEach(q => { g.append("rect").attr("x", x(acc)).attr("y", 30).attr("width", Math.max(0.5, x(acc+q[1]) - x(acc))).attr("height", 34).attr("fill", q[2]); acc += q[1]; });
    g.append("line").attr("x1", x(m.paper)).attr("x2", x(m.paper)).attr("y1", 20).attr("y2", 74).attr("stroke", P.ink).attr("stroke-dasharray","4,3").attr("stroke-width",1.5);
    txt(g, x(m.paper), 14, `paper: ${fmtM(m.paper)}`, {anchor:"middle", size:10, fill:P.ink});
    txt(g, x0, 88, `recomputed: ${comma(total)} (${fmtM(total)})`, {size:11, bold:true, fill:P.B});
    G.forEach((q,i) => {
      const yy = 110 + i*18;
      g.append("rect").attr("x", x0).attr("y", yy-9).attr("width", 10).attr("height", 10).attr("fill", q[2]);
      txt(g, x0 + 16, yy, q[0], {size:10.5});
      txt(g, 420, yy, comma(q[1]), {size:10.5, mono:true, anchor:"end"});
      txt(g, 500, yy, (100*q[1]/total).toFixed(1) + "%", {size:10.5, mono:true, anchor:"end", fill:P.muted});
    });
    const diff = total - m.paper;
    d3.select("#par-read").html(`<b>${m.name}</b>: recomputed <b>${fmtM(total)}</b> vs paper ${fmtM(m.paper)} (difference ${Math.abs(diff) < 5e4 ? "under 0.05M" : (diff > 0 ? "+" : "") + fmtM(diff) + ", " + (100*diff/m.paper).toFixed(1) + "%"})` +
      ` · configuration: L = ${m.L}, d = ${m.d}, vocab ${comma(m.V)}` + (m.c ? `, layout slices 4·${m.c} + 2·${m.s}` : `, ${m.tables} layout tables of 1024 × ${m.d}`) +
      (heads ? " · pretraining heads included" : ""));
  }
  sel.on("change", draw);
  d3.select("#par-heads").on("change", draw);
  draw();
});

/* ───────────────────────── 22 · benchmarks ───────────────────────── */
LV.safe("bench", function(){
  const {P, txt} = LV;
  const svg = d3.select("#bench-svg");
  /* ── Source: v1 = arXiv 1912.13318 Tables 1, 4, 5; v2 = arXiv 2012.14740 Tables 2–4 and appendix Tables 6–8;
        XLM = arXiv 2104.08836 Table 2; v3 = arXiv 2204.08387 Table 1. Scores as percentages. ── */
  const B = {
    funsd: [
      ["LayoutLM-base", 78.66, "v1", "text+layout, 11M pages, word-level F1 (v1 T1)"],
      ["LayoutLM-base + image", 79.27, "v1", "Faster R-CNN at fine-tuning (v1 T1)"],
      ["LayoutLM-large", 77.89, "v1", "11M pages, 1 epoch (v1 T1)"],
      ["LayoutLM-large (rerun)", 78.95, "v1", "entity-level F1, v2 paper T2"],
      ["LayoutXLM-base", 79.40, "xlm", "English FUNSD column, language-specific (XLM T2)"],
      ["LayoutXLM-large", 82.25, "xlm", "English FUNSD column (XLM T2)"],
      ["LayoutLMv2-base", 82.76, "v2", "entity-level F1 (v2 T2)"],
      ["LayoutLMv2-large", 84.20, "v2", "entity-level F1 (v2 T2)"],
      ["LayoutLMv3-base", 90.29, "v3", "segment-level boxes (v3 T1)"],
      ["LayoutLMv3-large", 92.08, "v3", "segment-level boxes (v3 T1)"]],
    cord: [
      ["LayoutLM-base", 94.72, "v1", "run by the v2 paper (T2)"],
      ["LayoutLM-large", 94.93, "v1", "run by the v2 paper (T2)"],
      ["LayoutLMv2-base", 94.95, "v2", "official OCR, entity F1 (v2 T2)"],
      ["LayoutLMv2-large", 96.01, "v2", "official OCR, entity F1 (v2 T2)"],
      ["LayoutLMv3-base", 96.56, "v3", "official OCR (v3 T1)"],
      ["LayoutLMv3-large", 97.46, "v3", "official OCR (v3 T1)"]],
    sroie: [
      ["LayoutLM-base", 94.38, "v1", "ground-truth OCR, 11M, 2 epochs (v1 T4)"],
      ["LayoutLM-base + image", 94.67, "v1", "(v1 T4)"],
      ["LayoutLM-large", 95.24, "v1", "11M, 1 epoch (v1 T4)"],
      ["LayoutLMv2-base", 96.25, "v2", "(v2 T2)"],
      ["LayoutLMv2-large", 96.61, "v2", "raw test set (v2 appendix T8)"],
      ["LayoutLMv2-large (excl. OCR mismatch)", 97.81, "v2", "as quoted in the abstract (v2 T8)"]],
    rvl: [
      ["LayoutLM-base", 91.78, "v1", "text+layout (v1 T5)"],
      ["LayoutLM-base + image", 94.42, "v1", "image at fine-tuning (v1 T5)"],
      ["LayoutLM-large", 91.90, "v1", "text+layout (v1 T5)"],
      ["LayoutLM-large + image", 94.43, "v1", "(v2 T3)"],
      ["LayoutLMv2-base", 95.25, "v2", "(v2 T3)"],
      ["LayoutLMv2-large", 95.64, "v2", "(v2 T3)"],
      ["LayoutLMv3-base", 95.44, "v3", "(v3 T1)"],
      ["LayoutLMv3-large", 95.93, "v3", "(v3 T1)"]],
    docvqa: [
      ["LayoutLM-base", 69.79, "v1", "train only (v2 T4)"],
      ["LayoutLM-large", 72.59, "v1", "train only (v2 T4)"],
      ["LayoutLMv2-base", 78.08, "v2", "train only (v2 T4)"],
      ["LayoutLMv2-large", 83.48, "v2", "train only (v2 T4)"],
      ["LayoutLMv2-large", 85.29, "v2", "train + dev (v2 T4)"],
      ["LayoutLMv2-large + QG", 86.72, "v2", "train + dev + ~1M generated QA (v2 T4)"],
      ["LayoutLMv3-base", 78.76, "v3", "train only (v3 T1)"],
      ["LayoutLMv3-large", 83.37, "v3", "train only (v3 T1)"]],
    xfund: [
      ["XLM-R base", 70.47, "base", "text only (XLM T2)"],
      ["InfoXLM base", 72.07, "base", "text only (XLM T2)"],
      ["LayoutXLM-base", 80.56, "xlm", "language-specific fine-tuning (XLM T2)"],
      ["XLM-R large", 73.74, "base", "text only (XLM T2)"],
      ["InfoXLM large", 74.71, "base", "text only (XLM T2)"],
      ["LayoutXLM-large", 82.82, "xlm", "language-specific fine-tuning (XLM T2)"]]
  };
  const COL = {v1:P.muted, v2:P.A, v3:P.good, xlm:P.B, base:P.grey};
  function draw(){
    svg.selectAll("*").remove();
    const task = d3.select("#bench-task").property("value"), D = B[task];
    const g = svg.append("g"), lx = 190, rx = 600, bh = Math.min(26, 300 / D.length);
    const lo = Math.floor(d3.min(D, d => d[1]) - 5), hi = Math.ceil(d3.max(D, d => d[1]) + 1);
    const x = d3.scaleLinear().domain([lo, hi]).range([lx, rx]);
    g.append("g").attr("transform",`translate(0,${12 + D.length*bh + 4})`).call(d3.axisBottom(x).ticks(6)).selectAll("text").attr("fill",P.muted);
    g.selectAll(".domain, .tick line").attr("stroke",P.line);
    D.forEach((d,i) => {
      const y = 12 + i*bh;
      g.append("rect").attr("x", lx).attr("y", y).attr("width", x(d[1]) - lx).attr("height", bh*0.62).attr("fill", COL[d[2]]).attr("opacity",0.85);
      txt(g, lx - 6, y + bh*0.45, d[0], {anchor:"end", size:10});
      txt(g, x(d[1]) + 4, y + bh*0.45, d[1].toFixed(2), {size:10, mono:true, fill:P.ink});
      txt(g, lx + 4, y + bh*0.62 + 8, d[3], {size:8, fill:P.muted});
    });
    const best = D.reduce((a,b) => b[1] > a[1] ? b : a);
    d3.select("#bench-read").html(`${D.length} results · highest: <b>${best[0]}</b> at <b>${best[1].toFixed(2)}</b> (${best[3]}) · spread ${(best[1] - d3.min(D, d=>d[1])).toFixed(2)} points` +
      ` · axis starts at ${lo}`);
  }
  d3.select("#bench-task").on("change", draw);
  draw();
});

/* ───────────────────────── 24 · sequence budget ───────────────────────── */
LV.safe("budget", function(){
  const {P, txt, comma} = LV;
  const svg = d3.select("#bud-svg");
  /* ── Source: released configurations — 512 text positions for all; v2/LayoutXLM image_feature_pool_shape 7×7;
        v3 input_size 224, patch 16 → 196 patches + 1 visual [CLS] in the Transformers implementation. ── */
  const VIS = {v1:0, v2:49, v3:197};
  function draw(){
    svg.selectAll("*").remove();
    const mdl = d3.select("#bud-model").property("value");
    const words = +d3.select("#bud-words").property("value"), spw = +d3.select("#bud-spw").property("value");
    const q = +d3.select("#bud-q").property("value"), s = +d3.select("#bud-stride").property("value");
    d3.select("#bud-wv").text(words); d3.select("#bud-sv").text(spw.toFixed(1)); d3.select("#bud-qv").text(q); d3.select("#bud-stv").text(s);
    /* BERT-style pair: [CLS] q [SEP] ctx [SEP]; v3's RoBERTa-style pair adds a second separator: <s> q </s></s> ctx </s> */
    const pairSep = q ? (mdl === "v3" ? 2 : 1) : 0;
    const n = Math.round(words * spw), cap = 512 - 2 - (q ? q + pairSep : 0), stride = Math.min(s, cap - 1);
    const k = 1 + Math.ceil(Math.max(0, n - cap) / (cap - stride));
    const vis = VIS[mdl], per = 512 + vis, tot = per * k;
    const g = svg.append("g"), x0 = 20, x1 = 620;
    const span = Math.max(n, cap);
    const x = d3.scaleLinear().domain([0, span]).range([x0, x1]);
    g.append("rect").attr("x", x(0)).attr("y", 30).attr("width", x(n) - x(0)).attr("height", 18).attr("fill", P.A).attr("opacity", 0.35);
    txt(g, x0, 22, `document: ${comma(n)} sub-word tokens (${words} words × ${spw.toFixed(1)})`, {size:10.5, fill:P.A});
    const shown = Math.min(k, 8);
    for (let i=0;i<shown;i++){
      const a = i*(cap - stride), b = Math.min(n, a + cap), y = 62 + i*20;
      g.append("rect").attr("x", x(a)).attr("y", y).attr("width", Math.max(1, x(b) - x(a))).attr("height", 14).attr("fill", P.B).attr("opacity", 0.7);
      txt(g, x(a) + 3, y + 11, `window ${i+1}: tokens ${a}–${b-1}` + (vis ? ` + ${vis} visual` : ""), {size:9.5, fill:"#111"});
    }
    if (k > shown) txt(g, x0, 62 + shown*20 + 10, `… ${k - shown} more windows`, {size:10, fill:P.muted});
    const yb = 62 + Math.min(k, 8)*20 + 30;
    txt(g, x0, yb, `per window: [CLS] + ${q ? q + " question + " + (pairSep === 2 ? "[SEP][SEP]" : "[SEP]") + " + " : ""}≤ ${cap} text + [SEP] + padding to 512` + (vis ? ` + ${vis} visual = ${per}` : ` = ${per}`), {size:10.5, mono:true});
    txt(g, x0, yb + 18, `attention scores per head per layer: ${per}² = ${comma(per*per)}`, {size:10.5, mono:true, fill:P.teal});
    d3.select("#bud-read").html(`<b>${comma(n)}</b> text tokens, capacity ${cap} per window, overlap ${stride} → <b>${k}</b> window${k>1?"s":""}` +
      ` · ${per} tokens per forward pass (${vis} visual), ${comma(tot)} tokens processed in total` +
      (vis ? ` · visual tokens repeated ${k}× = ${comma(vis*k)}` : "") +
      ` · text overhead from overlap: ${comma(Math.max(0, (k-1)*stride))} tokens seen twice`);
  }
  ["#bud-model"].forEach(id => d3.select(id).on("change", draw));
  ["#bud-words","#bud-spw","#bud-q","#bud-stride"].forEach(id => d3.select(id).on("input", draw));
  draw();
});
