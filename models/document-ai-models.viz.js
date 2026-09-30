/* document-ai-models.viz.js — every interactive figure on models/document-ai-models.html.
   Loaded after data.js / notes.js (palette C comes from notes.js).
   Each figure is its own IIFE wrapped in try/catch so one failure cannot blank the rest.
   House rule: every number a figure displays is computed here from a spec, never typed into a label. */

const DV = (function(){
  const P = Object.assign({}, (typeof C !== "undefined" ? C : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"}),
    { panel:"#15181f", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6", paper:"#1d212b" });
  function fmtN(x){
    const a = Math.abs(x);
    if (a >= 1e9)  return (x/1e9).toFixed(a>=1e10?0:1)+"B";
    if (a >= 1e6)  return (x/1e6).toFixed(a>=1e8?1:2)+"M";
    if (a >= 1e3)  return (x/1e3).toFixed(1)+"k";
    return String(Math.round(x));
  }
  function comma(x){ return Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function txt(g, x, y, s, o){
    o = o || {};
    return g.append("text").attr("x",x).attr("y",y).attr("font-size",o.size||11).attr("fill",o.fill||P.ink)
      .attr("text-anchor",o.anchor||"start").attr("font-family",o.mono?"SF Mono,Menlo,monospace":null)
      .attr("font-weight",o.bold?600:null).text(s);
  }
  function safe(name, fn){ try { fn(); } catch(e){ if (typeof console!=="undefined") console.error("[document-ai.viz] "+name+" failed:", e); } }
  const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  /* arXiv identifier YYMM.NNNNN → fractional year and a "Mon YYYY" label */
  function arxivDate(id){
    const yy = +id.slice(0,2), mm = +id.slice(2,4);
    return { t: 2000 + yy + (mm - 1)/12, label: MONTHS[mm-1] + " " + (2000 + yy) };
  }
  return {P, fmtN, comma, txt, safe, arxivDate};
})();

/* shared spec: the models compared on the hub (only facts stated in each model's paper) */
const DOC_MODELS = [
  {name:"LayoutLM",   arxiv:"1912.13318", lane:"ocr",  text:"OCR", layout:"absolute 2D", image:"CNN regions (fine-tune only)", head:"encoder", res:null, resNote:"no image in pretraining", params:343e6, paramsNote:"LARGE"},
  {name:"LayoutLMv2", arxiv:"2012.14740", lane:"ocr",  text:"OCR", layout:"2D + spatial bias", image:"CNN grid", head:"encoder", res:[224,224], params:426e6, paramsNote:"LARGE"},
  {name:"LayoutXLM",  arxiv:"2104.08836", lane:"ocr",  text:"OCR", layout:"2D + spatial bias", image:"CNN grid", head:"encoder", res:[224,224], params:625e6, paramsNote:"LARGE; BASE ≈ 345M"},
  {name:"DocFormer",  arxiv:"2106.11539", lane:"ocr",  text:"OCR", layout:"shared spatial emb.", image:"CNN grid", head:"encoder", res:null, resNote:"not stated here", params:536e6, paramsNote:"LARGE"},
  {name:"Donut",      arxiv:"2111.15664", lane:"free", text:"pixels", layout:"implicit (pixels)", image:"Swin", head:"encoder–decoder", res:[2560,1920], params:143e6, paramsNote:"single size"},
  {name:"LiLT",       arxiv:"2202.13669", lane:"ocr",  text:"OCR", layout:"layout stream", image:"none", head:"encoder", res:null, resNote:"text + layout only", params:null, paramsNote:"text model + 6.1M layout flow"},
  {name:"LayoutLMv3", arxiv:"2204.08387", lane:"ocr",  text:"OCR", layout:"2D + spatial bias", image:"linear patches", head:"encoder", res:[224,224], params:368e6, paramsNote:"LARGE"},
  {name:"Pix2Struct", arxiv:"2210.03347", lane:"free", text:"pixels", layout:"implicit (pixels)", image:"ViT, variable", head:"encoder–decoder", res:null, resNote:"≤ 2,048 patches of 16 × 16 (pretraining)", params:1.3e9, paramsNote:"LARGE"},
  {name:"UDOP",       arxiv:"2212.02623", lane:"ocr",  text:"OCR", layout:"layout tokens", image:"linear patches", head:"encoder–decoder", res:[1024,1024], params:794e6, paramsNote:"single size"},
  {name:"Nougat",     arxiv:"2308.13418", lane:"free", text:"pixels", layout:"implicit (pixels)", image:"Swin", head:"encoder–decoder", res:[896,672], params:350e6, paramsNote:"base"},
  {name:"Qwen2-VL",   arxiv:"2409.12191", lane:"free", text:"pixels", layout:"implicit (pixels)", image:"ViT, variable", head:"encoder–decoder", res:null, resNote:"dynamic, 28 × 28 px per token", params:72e9, paramsNote:"largest open size", general:true}
];

/* ───────────────────────── 02 · design-axis matrix ───────────────────────── */
DV.safe("axes", function(){
  const {P, txt, arxivDate} = DV;
  const svg = d3.select("#ax-svg"), W = 640;
  const cols = [
    {k:"text", label:"text source"}, {k:"layout", label:"layout"}, {k:"image", label:"image input"},
    {k:"head", label:"output head"}, {k:"res", label:"resolution"}
  ];
  const palette = [P.A, P.B, P.teal, P.purple, P.pink, P.good, "#94a3b8", "#fcd34d"];
  const colour = {};
  cols.forEach(c => {
    const vals = [...new Set(DOC_MODELS.map(d => c.k === "res" ? resClass(d) : d[c.k]))];
    colour[c.k] = d3.scaleOrdinal().domain(vals).range(palette);
  });
  function resClass(d){
    if (!d.res) return "n/a or variable";
    const px = d.res[0]*d.res[1];
    return px >= 1e6 ? "≥ 1 megapixel" : (px >= 3e5 ? "0.3–1 megapixel" : "≤ 0.3 megapixel");
  }
  function cellText(d, k){
    if (k === "res") return d.res ? d.res[0] + " × " + d.res[1] : "—";
    return d[k];
  }
  let sel = 4, sortKey = "year";
  function draw(){
    svg.selectAll("*").remove();
    const rows = DOC_MODELS.map((d,i)=>Object.assign({i}, d));
    if (sortKey === "year") rows.sort((a,b)=>arxivDate(a.arxiv).t - arxivDate(b.arxiv).t);
    else rows.sort((a,b)=> String(sortKey==="res"?resClass(a):a[sortKey]).localeCompare(String(sortKey==="res"?resClass(b):b[sortKey])) || arxivDate(a.arxiv).t - arxivDate(b.arxiv).t);
    const x0 = 96, cw = (W - x0 - 8) / cols.length, rh = 26, y0 = 34;
    const g = svg.append("g");
    cols.forEach((c,j)=> txt(g, x0 + j*cw + cw/2, 20, c.label, {anchor:"middle", size:11, fill: sortKey===c.k ? P.B : P.muted, bold: sortKey===c.k}));
    rows.forEach((d,r)=>{
      const y = y0 + r*rh, on = d.i === sel;
      const row = g.append("g").style("cursor","pointer").on("click",()=>{ sel = d.i; draw(); });
      row.append("rect").attr("x",2).attr("y",y).attr("width",W-6).attr("height",rh-3).attr("rx",4)
        .attr("fill", on ? "rgba(255,180,84,.10)" : "transparent").attr("stroke", on ? P.B : "none");
      txt(row, 8, y + 16, d.name, {size:11.5, fill: on ? P.B : P.ink, bold:on});
      cols.forEach((c,j)=>{
        const v = c.k === "res" ? resClass(d) : d[c.k];
        row.append("rect").attr("x", x0 + j*cw + 3).attr("y", y + 3).attr("width", cw - 6).attr("height", rh - 9).attr("rx",3)
          .attr("fill", colour[c.k](v)).attr("fill-opacity", .22).attr("stroke", colour[c.k](v)).attr("stroke-opacity", .6);
        const s = cellText(d, c.k);
        txt(row, x0 + j*cw + cw/2, y + 16, s.length > 18 ? s.slice(0,17) + "…" : s, {anchor:"middle", size:10, fill:P.ink});
      });
    });
    const d = DOC_MODELS[sel], dt = arxivDate(d.arxiv);
    const same = DOC_MODELS.filter(m => m !== d && m.text === d.text && m.head === d.head).map(m=>m.name);
    const px = d.res ? (d.res[0]*d.res[1]/1e6).toFixed(2) + " megapixels" : d.resNote;
    d3.select("#ax-read").html(`<b>${d.name}</b> (${dt.label}): text from <b>${d.text}</b>, layout via <b>${d.layout}</b>, image as <b>${d.image}</b>, <b>${d.head}</b> head, input ${px}. ` +
      (same.length ? `Shares text source and head with ${same.length}: ${same.join(", ")}.` : "No other model here shares its text source and head."));
  }
  d3.select("#ax-sort").on("change", function(){ sortKey = this.value; draw(); });
  draw();
});

/* ───────────────────────── 03 · bounding-box normalisation ───────────────────────── */
DV.safe("bbox", function(){
  const {P, txt} = DV;
  const svg = d3.select("#bb-svg");
  /* a 3 × 5 inch receipt; word boxes in inches: [x0, y0, x1, y1] */
  const PAGE = {w:3, h:5};
  const WORDS = [
    {t:"BLUE",   b:[0.55,0.30,1.00,0.50], line:0}, {t:"BOTTLE", b:[1.08,0.30,1.72,0.50], line:0}, {t:"CAFE", b:[1.80,0.30,2.40,0.50], line:0},
    {t:"12",     b:[1.02,0.62,1.20,0.76], line:1}, {t:"Main",   b:[1.26,0.62,1.60,0.76], line:1}, {t:"St",   b:[1.66,0.62,1.86,0.76], line:1},
    {t:"Latte",  b:[0.30,1.30,0.78,1.46], line:2}, {t:"1",      b:[1.70,1.30,1.78,1.46], line:2}, {t:"4.50", b:[2.28,1.30,2.70,1.46], line:2},
    {t:"Scone",  b:[0.30,1.62,0.82,1.78], line:3}, {t:"2",      b:[1.70,1.62,1.79,1.78], line:3}, {t:"7.00", b:[2.28,1.62,2.70,1.78], line:3},
    {t:"Tax",    b:[0.30,2.10,0.62,2.26], line:4}, {t:"0.92",   b:[2.28,2.10,2.70,2.26], line:4},
    {t:"TOTAL",  b:[0.30,2.60,1.00,2.82], line:5}, {t:"12.42",  b:[2.12,2.60,2.70,2.82], line:5},
    {t:"Thank",  b:[0.92,4.30,1.40,4.46], line:6}, {t:"you",    b:[1.48,4.30,1.86,4.46], line:6}
  ];
  let sel = 15;
  const S = 64, ox = 24, oy = 18;   /* display scale: px per inch on screen */
  function lineBox(l){
    const ws = WORDS.filter(w=>w.line===l);
    return [d3.min(ws,w=>w.b[0]), d3.min(ws,w=>w.b[1]), d3.max(ws,w=>w.b[2]), d3.max(ws,w=>w.b[3])];
  }
  function draw(){
    svg.selectAll("*").remove();
    const dpi = +d3.select("#bb-dpi").property("value");
    const seg = d3.select("#bb-seg").property("checked");
    const Wpx = Math.round(PAGE.w*dpi), Hpx = Math.round(PAGE.h*dpi);
    const g = svg.append("g");
    g.append("rect").attr("x",ox).attr("y",oy).attr("width",PAGE.w*S).attr("height",PAGE.h*S).attr("fill",P.paper).attr("stroke",P.muted);
    WORDS.forEach((w,i)=>{
      const b = seg ? lineBox(w.line) : w.b, on = i===sel || (seg && w.line===WORDS[sel].line);
      const node = g.append("g").style("cursor","pointer").on("click",()=>{ sel=i; draw(); });
      node.append("rect").attr("x",ox+b[0]*S).attr("y",oy+b[1]*S).attr("width",(b[2]-b[0])*S).attr("height",(b[3]-b[1])*S)
        .attr("fill", on ? "rgba(255,180,84,.18)" : "rgba(91,156,255,.08)").attr("stroke", on ? P.B : P.A).attr("stroke-width", on?1.5:0.8);
      txt(node, ox + (w.b[0]+w.b[2])/2*S, oy + w.b[3]*S - 3, w.t, {anchor:"middle", size: w.line===0||w.line===5 ? 10.5 : 9.5, fill:P.ink, bold: w.line===0||w.line===5});
    });
    txt(g, ox, oy + PAGE.h*S + 14, `page ${Wpx} × ${Hpx} px at ${dpi} dpi`, {size:10, fill:P.muted});
    /* right panel: the selected box in both coordinate systems */
    const w = WORDS[sel], b = seg ? lineBox(w.line) : w.b;
    const px = b.map((v,k)=> Math.round(v*dpi));
    const nb = px.map((v,k)=> Math.floor(1000 * v / (k%2===0 ? Wpx : Hpx)));
    const rx = 250, ry = 30;
    txt(g, rx, ry, `word "${w.t}"  ·  1D position ${sel + 1} of ${WORDS.length}`, {size:12, bold:true});
    const rows = [["", "x₀", "y₀", "x₁", "y₁"], ["pixels", ...px], ["÷ page", ...px.map((v,k)=> (v/(k%2===0?Wpx:Hpx)).toFixed(4))], ["0–1000 id", ...nb]];
    rows.forEach((r,i)=> r.forEach((c,j)=> txt(g, rx + (j===0?0:62 + j*68), ry + 30 + i*24, String(c), {size:11, mono: j>0, fill: i===3 ? P.B : (i===0 ? P.muted : P.ink), anchor: j===0?"start":"end"})));
    /* the same word at every scan resolution, to show the ids barely move */
    const dpis = [100,200,300];
    txt(g, rx, ry + 150, "same box, all three scan resolutions → 0–1000 ids", {size:11, fill:P.muted});
    dpis.forEach((d,i)=>{
      const Wd = Math.round(PAGE.w*d), Hd = Math.round(PAGE.h*d);
      const ids = b.map((v,k)=> Math.floor(1000*Math.round(v*d)/(k%2===0?Wd:Hd)));
      txt(g, rx, ry + 174 + i*20, `${d} dpi`, {size:11, fill: d===dpi ? P.B : P.ink});
      txt(g, rx + 60, ry + 174 + i*20, "(" + ids.join(", ") + ")", {size:11, mono:true, fill: d===dpi ? P.B : P.ink});
    });
    const all = dpis.map(d=>{ const Wd=Math.round(PAGE.w*d), Hd=Math.round(PAGE.h*d); return b.map((v,k)=> Math.floor(1000*Math.round(v*d)/(k%2===0?Wd:Hd))); });
    const spread = d3.max([0,1,2,3], k => d3.max(all, a=>a[k]) - d3.min(all, a=>a[k]));
    d3.select("#bb-read").html(`<b>${w.t}</b>${seg ? " (segment box of its line)" : ""}: pixels (${px.join(", ")}) on a ${Wpx} × ${Hpx} page → ids <b>(${nb.join(", ")})</b>. ` +
      `Across 100/200/300 dpi the ids differ by at most <b>${spread}</b>` + (seg ? `; all ${WORDS.filter(v=>v.line===w.line).length} words on this line share these ids.` : "."));
  }
  d3.select("#bb-dpi").on("change", draw);
  d3.select("#bb-seg").on("change", draw);
  draw();
});

/* ───────────────────────── 04 · error propagation ───────────────────────── */
DV.safe("err", function(){
  const {P, txt} = DV;
  const svg = d3.select("#pe-svg"), W = 640, H = 280, m = {l:52, r:20, t:18, b:40};
  const K = d3.range(1, 13);
  const x = d3.scaleLinear().domain([1, 12]).range([m.l, W-m.r]);
  const y = d3.scaleLinear().domain([0.4, 1]).range([H-m.b, m.t]);
  function draw(){
    const a = +d3.select("#pe-ocr").property("value"), e = +d3.select("#pe-ext").property("value"), q = +d3.select("#pe-e2e").property("value");
    d3.select("#pe-ocrv").text(a.toFixed(3)); d3.select("#pe-extv").text(e.toFixed(3)); d3.select("#pe-e2ev").text(q.toFixed(3));
    svg.selectAll("*").remove();
    const g = svg.append("g");
    [0.4,0.5,0.6,0.7,0.8,0.9,1].forEach(v=>{
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, m.l-6, y(v)+4, v.toFixed(1), {anchor:"end", size:10, fill:P.muted});
    });
    K.forEach(k=> txt(g, x(k), H-m.b+15, String(k), {anchor:"middle", size:10, fill:P.muted}));
    txt(g, (m.l+W-m.r)/2, H-6, "words per field k", {anchor:"middle", size:10.5, fill:P.muted});
    txt(g, 8, m.t+2, "field acc.", {size:10, fill:P.muted});
    const pipe = K.map(k=>({k, v: Math.pow(a,k)*e}));
    g.append("path").datum(pipe).attr("fill","none").attr("stroke",P.A).attr("stroke-width",2)
      .attr("d", d3.line().x(d=>x(d.k)).y(d=>y(Math.max(0.4,d.v))));
    pipe.forEach(d=> g.append("circle").attr("cx",x(d.k)).attr("cy",y(Math.max(0.4,d.v))).attr("r",3).attr("fill",P.A));
    g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(q)).attr("y2",y(q)).attr("stroke",P.B).attr("stroke-width",2);
    txt(g, W-m.r-4, y(q)-6, "end-to-end", {anchor:"end", size:10.5, fill:P.B});
    txt(g, x(1)+6, y(Math.max(0.4,pipe[0].v))-8, "pipeline aᵏ · e", {size:10.5, fill:P.A});
    const cross = pipe.find(d=>d.v < q);
    if (cross) g.append("line").attr("x1",x(cross.k)).attr("x2",x(cross.k)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.bad).attr("stroke-dasharray","4,3");
    const at = k => (Math.pow(a,k)*e*100).toFixed(1) + "%";
    d3.select("#pe-read").html(`pipeline field accuracy: 1 word ${at(1)}, 3 words ${at(3)}, 8 words ${at(8)} · end-to-end ${(q*100).toFixed(1)}% · ` +
      (cross ? (cross.k === 1 ? `<b>end-to-end wins at every field length</b> in this setting.` : `<b>crossover at k = ${cross.k}</b>: fields of ${cross.k}+ words favour the end-to-end model.`)
             : `<b>pipeline wins up to k = 12</b> — OCR is accurate enough that compounding never catches up.`));
  }
  ["#pe-ocr","#pe-ext","#pe-e2e"].forEach(id => d3.select(id).on("input", draw));
  draw();
});

/* ───────────────────────── 05 · token budgets ───────────────────────── */
DV.safe("tokens", function(){
  const {P, txt, comma} = DV;
  const svg = d3.select("#tb-svg"), W = 640, H = 300, m = {l:118, r:70, t:14, b:30};
  const PAGES = {
    letter200:  {label:"US Letter at 200 dpi", w:8.5,  h:11,    dpi:200},
    a4300:      {label:"A4 at 300 dpi",        w:8.27, h:11.69, dpi:300},
    receipt200: {label:"receipt 3 × 8 in at 200 dpi", w:3, h:8, dpi:200}
  };
  const SPEC = {
    lmv2:  {img:49,  imgExtra:0, win:512},              /* 7 × 7 pooled ResNeXt grid */
    lmv3:  {img:(224/16)*(224/16), imgExtra:1, win:512},/* 196 patches + image [CLS] */
    donut: {H:2560, W:1920, patch:4, stages:4},
    nougat:{H:896,  W:672,  patch:4, stages:4},
    p2s:   {maxPatches:2048, p:16},
    qwen:  {f:28, minPx:4*28*28, maxPx:16384*28*28}   /* released checkpoints' preprocessor_config: min_pixels 3,136, max_pixels 12,845,056 */
  };
  const swinFinal = s => (s.H/(s.patch*Math.pow(2,s.stages-1))) * (s.W/(s.patch*Math.pow(2,s.stages-1)));
  const swinFirst = s => (s.H/s.patch) * (s.W/s.patch);
  function pix2struct(Hp, Wp, s){
    const scale = Math.sqrt(s.maxPatches * (s.p/Hp) * (s.p/Wp));
    const rows = Math.max(1, Math.min(Math.floor(scale*Hp/s.p), s.maxPatches));
    const cols = Math.max(1, Math.min(Math.floor(scale*Wp/s.p), s.maxPatches));
    return {rows, cols, n: rows*cols};
  }
  function qwen(Hp, Wp, s){
    let h = Math.round(Hp/s.f)*s.f, w = Math.round(Wp/s.f)*s.f;
    if (h*w > s.maxPx){ const beta = Math.sqrt(Hp*Wp/s.maxPx); h = Math.floor(Hp/beta/s.f)*s.f; w = Math.floor(Wp/beta/s.f)*s.f; }
    else if (h*w < s.minPx){ const beta = Math.sqrt(s.minPx/(Hp*Wp)); h = Math.ceil(Hp*beta/s.f)*s.f; w = Math.ceil(Wp*beta/s.f)*s.f; }
    return {h, w, n: (h/s.f)*(w/s.f)};
  }
  function draw(){
    const pg = PAGES[d3.select("#tb-page").property("value")];
    const words = +d3.select("#tb-words").property("value"), spw = +d3.select("#tb-spw").property("value");
    d3.select("#tb-wordsv").text(words); d3.select("#tb-spwv").text(spw.toFixed(2));
    const Hp = Math.round(pg.h*pg.dpi), Wp = Math.round(pg.w*pg.dpi);
    const T = Math.ceil(words*spw);
    const winCap = SPEC.lmv2.win - 2, wins = Math.ceil(T / winCap);
    const p2 = pix2struct(Hp, Wp, SPEC.p2s), qw = qwen(Hp, Wp, SPEC.qwen);
    const bars = [
      {name:"LayoutLMv2", text:T + 2*wins, img:wins*SPEC.lmv2.img, fam:"ocr"},
      {name:"LayoutLMv3", text:T + 2*wins, img:wins*(SPEC.lmv3.img + SPEC.lmv3.imgExtra), fam:"ocr"},
      {name:"Donut (final stage)", text:0, img:swinFinal(SPEC.donut), fam:"free"},
      {name:"Nougat (final stage)", text:0, img:swinFinal(SPEC.nougat), fam:"free"},
      {name:"Pix2Struct", text:0, img:p2.n, fam:"free"},
      {name:"Qwen2-VL", text:0, img:qw.n, fam:"free"},
      {name:"Donut (stage 1)", text:0, img:swinFirst(SPEC.donut), fam:"free", ghost:true}
    ];
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const x = d3.scaleLog().domain([10, 1e6]).range([m.l, W-m.r]);
    const bh = (H - m.t - m.b) / bars.length;
    [10,100,1e3,1e4,1e5,1e6].forEach(v=>{
      g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, x(v), H-m.b+14, v>=1e3 ? (v/1e3)+"k" : String(v), {anchor:"middle", size:10, fill:P.muted});
    });
    bars.forEach((b,i)=>{
      const y0 = m.t + i*bh + 4, h = bh - 8, tot = b.text + b.img;
      txt(g, m.l-8, y0 + h/2 + 4, b.name, {anchor:"end", size:10.5, fill: b.ghost ? P.muted : P.ink});
      const col = b.fam === "ocr" ? P.A : P.B;
      if (b.text > 0){
        g.append("rect").attr("x",m.l).attr("y",y0).attr("width",x(b.text)-m.l).attr("height",h).attr("fill",col).attr("fill-opacity",.8);
        g.append("rect").attr("x",x(b.text)).attr("y",y0).attr("width",Math.max(1,x(tot)-x(b.text))).attr("height",h).attr("fill",P.teal).attr("fill-opacity",.85);
      } else {
        g.append("rect").attr("x",m.l).attr("y",y0).attr("width",x(tot)-m.l).attr("height",h).attr("fill",col).attr("fill-opacity", b.ghost ? .3 : .8);
      }
      txt(g, x(tot)+5, y0 + h/2 + 4, comma(tot), {size:10, fill:P.ink, mono:true});
    });
    d3.select("#tb-read").html(`page ${Wp} × ${Hp} px · ${words} words × ${spw.toFixed(2)} = <b>${comma(T)}</b> text tokens → <b>${wins}</b> window${wins>1?"s":""} of ≤ ${winCap} for the OCR encoders · ` +
      `Pix2Struct grid ${p2.rows} × ${p2.cols} = ${comma(p2.n)} · Qwen2-VL resizes to ${qw.w} × ${qw.h} → ${comma(qw.n)} · Donut: ${comma(swinFirst(SPEC.donut))} stage-1 tokens in 10 × 10 windows, <b>${comma(swinFinal(SPEC.donut))}</b> for the decoder.`);
  }
  d3.select("#tb-page").on("change", draw);
  d3.select("#tb-words").on("input", draw);
  d3.select("#tb-spw").on("input", draw);
  draw();
});

/* ───────────────────────── 07 · timeline ───────────────────────── */
DV.safe("timeline", function(){
  const {P, txt, fmtN, arxivDate} = DV;
  const svg = d3.select("#tl-svg"), W = 640, H = 280, m = {l:80, r:24, t:20, b:34};
  const ms = DOC_MODELS.map(d => Object.assign({}, d, arxivDate(d.arxiv)));
  const x = d3.scaleLinear().domain([2019.6, 2024.9]).range([m.l, W-m.r]);
  const laneY = {ocr: 90, free: 196};
  const r = d3.scaleSqrt().domain([1e8, 7.2e10]).range([5, 22]).clamp(true);
  let sel = 0;
  function draw(){
    svg.selectAll("*").remove();
    const g = svg.append("g");
    [2020,2021,2022,2023,2024].forEach(v=>{
      g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, x(v), H-m.b+16, String(v), {anchor:"middle", size:10, fill:P.muted});
    });
    Object.keys(laneY).forEach(k=>{
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",laneY[k]).attr("y2",laneY[k]).attr("stroke",k==="ocr"?P.A:P.B).attr("stroke-opacity",.35);
      txt(g, 8, laneY[k]+4, k==="ocr" ? "OCR-based" : "OCR-free", {size:11, fill:k==="ocr"?P.A:P.B, bold:true});
    });
    ms.forEach((d,i)=>{
      const cx = x(d.t), cy = laneY[d.lane], on = i===sel, col = d.lane==="ocr" ? P.A : P.B;
      const node = g.append("g").style("cursor","pointer").on("click",()=>{ sel=i; draw(); });
      if (d.params) node.append("circle").attr("cx",cx).attr("cy",cy).attr("r",r(d.params)).attr("fill",col).attr("fill-opacity",on?.9:.55).attr("stroke",on?P.ink:"none");
      else node.append("rect").attr("x",cx-6).attr("y",cy-6).attr("width",12).attr("height",12).attr("fill","none").attr("stroke",on?P.ink:col).attr("stroke-dasharray","3,2");
      const above = (i % 2 === 0);
      const off = (d.params ? r(d.params) : 7) + 6;
      txt(node, cx, above ? cy - off : cy + off + 9, d.name, {anchor:"middle", size:10.5, fill:on?P.B:P.ink, bold:on});
    });
    const d = ms[sel];
    const prev = ms.filter(p => p.lane === d.lane && p.t < d.t && p.params).pop();
    const size = d.params ? `largest reported size <b>${fmtN(d.params)}</b> (${d.paramsNote})` : `size: ${d.paramsNote} — drawn as a dashed box`;
    const rel = (d.params && prev) ? ` · ${(d.params/prev.params).toFixed(1)}× ${prev.name}` : "";
    const gap = ms.filter(p => p.lane !== d.lane && p.t <= d.t).length;
    d3.select("#tl-read").html(`<b>${d.name}</b> · ${d.label} · ${d.lane==="ocr"?"OCR-based":"OCR-free"}${d.general?" (general VLM)":""} · ${size}${rel} · ${d.text==="OCR"?"text from OCR":"reads pixels"}, ${d.head} · ${gap} model${gap===1?"":"s"} from the other lane precede it.`);
  }
  draw();
});

/* ───────────────────────── 08 · LayoutLM parameter count ───────────────────────── */
DV.safe("params", function(){
  const {P, txt, fmtN, comma} = DV;
  const svg = d3.select("#pc-svg"), W = 640;
  const FIXED = {V:30522, Pn:512, types:2};
  const PRESETS = {
    base:  {name:"LayoutLM BASE",  L:12, d:768,  R:1024, reported:113e6, repLabel:"113M"},
    large: {name:"LayoutLM LARGE", L:24, d:1024, R:1024, reported:343e6, repLabel:"343M"},
    bert:  {name:"BERT-base",      L:12, d:768,  R:0,    reported:110e6, repLabel:"110M"}
  };
  function count(L, d, R){
    const tok = FIXED.V*d, pos = FIXED.Pn*d + FIXED.types*d + 2*d, lay = 4*R*d, enc = L*(12*d*d + 13*d), pool = d*d + d;
    return {parts:[["token embeddings", tok, P.purple], ["1D pos + segment + LN", pos, P.muted], ["2D layout tables", lay, P.B], ["encoder layers", enc, P.A], ["pooler", pool, P.teal]], total: tok+pos+lay+enc+pool};
  }
  let current = "base";
  function setPreset(k){
    const p = PRESETS[k]; current = k;
    d3.select("#pc-L").property("value", p.L); d3.select("#pc-d").property("value", p.d); d3.select("#pc-R").property("value", p.R);
  }
  function draw(){
    const L = +d3.select("#pc-L").property("value"), d = +d3.select("#pc-d").property("value"), R = +d3.select("#pc-R").property("value");
    d3.select("#pc-Lv").text(L); d3.select("#pc-dv").text(d); d3.select("#pc-Rv").text(R);
    const p = PRESETS[current], matches = p && p.L===L && p.d===d && p.R===R;
    const c = count(L, d, R);
    svg.selectAll("*").remove();
    const g = svg.append("g"), x0 = 20, x1 = W - 20, y0 = 40, h = 44;
    const sx = d3.scaleLinear().domain([0, c.total]).range([x0, x1]);
    let acc = 0;
    c.parts.forEach((pt,i)=>{
      const a = sx(acc), b = sx(acc + pt[1]);
      g.append("rect").attr("x",a).attr("y",y0).attr("width",Math.max(0.5,b-a)).attr("height",h).attr("fill",pt[2]).attr("fill-opacity",.8);
      acc += pt[1];
      const lx = x0 + (i % 3) * 205, ly = y0 + h + 30 + Math.floor(i/3)*22;
      g.append("rect").attr("x",lx).attr("y",ly-9).attr("width",10).attr("height",10).attr("fill",pt[2]);
      txt(g, lx+15, ly, `${pt[0]}: ${fmtN(pt[1])} (${(100*pt[1]/c.total).toFixed(1)}%)`, {size:10.5});
    });
    txt(g, x0, y0-10, `total ${comma(c.total)} parameters  (L = ${L}, d = ${d}, R = ${R})`, {size:12, bold:true});
    if (matches){
      const gap = (c.total - p.reported)/p.reported*100;
      txt(g, x1, y0-10, `${p.name}: reported ${p.repLabel}`, {anchor:"end", size:11, fill:P.B});
      d3.select("#pc-read").html(`<b>${p.name}</b>: configuration gives <b>${fmtN(c.total)}</b> against the reported ${p.repLabel} (${gap>=0?"+":""}${gap.toFixed(1)}%). 2D layout tables add ${fmtN(4*R*d)} — ${(100*4*R*d/c.total).toFixed(1)}% of the model.`);
    } else {
      d3.select("#pc-read").html(`custom configuration: <b>${fmtN(c.total)}</b> parameters; 2D layout tables ${fmtN(4*R*d)} (${(100*4*R*d/c.total).toFixed(1)}%); encoder layers ${(100*c.parts[3][1]/c.total).toFixed(1)}%.`);
    }
  }
  d3.select("#pc-preset").on("change", function(){ setPreset(this.value); draw(); });
  ["#pc-L","#pc-d","#pc-R"].forEach(id => d3.select(id).on("input", draw));
  setPreset("base");
  draw();
});

/* ───────────────────────── 09 · benchmark explorer ───────────────────────── */
DV.safe("bench", function(){
  const {P, txt} = DV;
  const svg = d3.select("#bm-svg"), W = 640, H = 330, m = {l:150, r:56, t:12, b:26};
  /* scores as reported (see the comparison table); flag = different protocol */
  const R = [
    {n:"LayoutLM BASE",       fam:"ocr",  funsd:78.66, rvl:91.78, docvqa:69.79},
    {n:"LayoutLM BASE + img", fam:"ocr",  funsd:79.27, rvl:94.42},
    {n:"LayoutLM LARGE",      fam:"ocr",  funsd:77.89, rvl:91.90, docvqa:72.59},
    {n:"LayoutLMv2 BASE",     fam:"ocr",  funsd:82.76, cord:94.95, rvl:95.25, docvqa:78.08},
    {n:"LayoutLMv2 LARGE",    fam:"ocr",  funsd:84.20, cord:96.01, rvl:95.64, docvqa:83.48},
    {n:"DocFormer BASE",      fam:"ocr",  funsd:83.34, cord:96.33, rvl:96.17},
    {n:"DocFormer LARGE",     fam:"ocr",  funsd:84.55, cord:96.99, rvl:95.50},
    {n:"LiLT BASE",           fam:"ocr",  funsd:88.41, cord:96.07, rvl:95.68},
    {n:"LayoutLMv3 BASE",     fam:"ocr",  funsd:90.29, cord:96.56, rvl:95.44, docvqa:78.76},
    {n:"LayoutLMv3 LARGE",    fam:"ocr",  funsd:92.08, cord:97.46, rvl:95.93, docvqa:83.37},
    {n:"UDOP",                fam:"ocr",  funsd:91.62, cord:97.58, rvl:96.00, docvqa:84.7},
    {n:"Donut",               fam:"free", cord:84.1, cordFlag:true, rvl:95.30, docvqa:67.5},
    {n:"Pix2Struct BASE",     fam:"free", docvqa:72.1},
    {n:"Pix2Struct LARGE",    fam:"free", docvqa:76.6},
    {n:"Qwen2-VL 2B",         fam:"free", docvqa:90.1},
    {n:"Qwen2-VL 7B",         fam:"free", docvqa:94.5},
    {n:"Qwen2-VL 72B",        fam:"free", docvqa:96.5}
  ];
  const LABEL = {funsd:"FUNSD F1", cord:"CORD F1", rvl:"RVL-CDIP accuracy", docvqa:"DocVQA ANLS"};
  function draw(){
    const k = d3.select("#bm-metric").property("value"), inclFree = d3.select("#bm-ocrfree").property("checked");
    const rows = R.filter(r => r[k] != null && (inclFree || r.fam === "ocr")).sort((a,b)=>b[k]-a[k]);
    svg.selectAll("*").remove();
    const defs = svg.append("defs");
    const pat = defs.append("pattern").attr("id","bm-hatch").attr("width",6).attr("height",6).attr("patternUnits","userSpaceOnUse").attr("patternTransform","rotate(45)");
    pat.append("rect").attr("width",6).attr("height",6).attr("fill",P.B).attr("fill-opacity",.25);
    pat.append("line").attr("x1",0).attr("y1",0).attr("x2",0).attr("y2",6).attr("stroke",P.B).attr("stroke-width",2);
    const g = svg.append("g");
    const x = d3.scaleLinear().domain([60, 100]).range([m.l, W-m.r]);
    const bh = Math.min(22, (H - m.t - m.b) / Math.max(1, rows.length));
    [60,70,80,90,100].forEach(v=>{
      g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",m.t).attr("y2",m.t + rows.length*bh).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, x(v), m.t + rows.length*bh + 14, String(v), {anchor:"middle", size:10, fill:P.muted});
    });
    rows.forEach((r,i)=>{
      const y0 = m.t + i*bh + 2, h = bh - 5, flag = k === "cord" && r.cordFlag;
      txt(g, m.l-8, y0 + h/2 + 4, r.n, {anchor:"end", size:10.5});
      g.append("rect").attr("x",m.l).attr("y",y0).attr("width",Math.max(1, x(r[k]) - m.l)).attr("height",h)
        .attr("fill", flag ? "url(#bm-hatch)" : (r.fam==="ocr" ? P.A : P.B)).attr("fill-opacity", flag ? 1 : .8);
      txt(g, x(r[k]) + 5, y0 + h/2 + 4, r[k].toFixed(2), {size:10, mono:true});
    });
    const comparable = rows.filter(r => !(k==="cord" && r.cordFlag));
    const best = comparable[0], ocrBest = comparable.find(r=>r.fam==="ocr"), freeBest = comparable.find(r=>r.fam==="free");
    const spread = comparable.length ? comparable[0][k] - comparable[comparable.length-1][k] : 0;
    let s = `<b>${LABEL[k]}</b>: ${rows.length} models · best comparable <b>${best ? best.n + " " + best[k].toFixed(2) : "none"}</b> · spread ${spread.toFixed(2)} points`;
    if (ocrBest && freeBest) s += ` · best OCR-based ${ocrBest[k].toFixed(2)} vs best OCR-free ${freeBest[k].toFixed(2)} (${(freeBest[k]-ocrBest[k]>=0?"+":"")}${(freeBest[k]-ocrBest[k]).toFixed(2)})`;
    if (k === "cord" && rows.some(r=>r.cordFlag)) s += " · Donut's hatched bar uses a different protocol and is excluded from these comparisons";
    if (!rows.length) s = `<b>${LABEL[k]}</b>: no models with a reported score under this filter`;
    d3.select("#bm-read").html(s + ".");
  }
  d3.select("#bm-metric").on("change", draw);
  d3.select("#bm-ocrfree").on("change", draw);
  draw();
});
