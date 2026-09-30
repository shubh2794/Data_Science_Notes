/* vision-models.viz.js — every interactive figure on models/vision-models.html.
   Loaded after data.js / notes.js (palette C comes from notes.js).
   Each figure is its own IIFE wrapped in try/catch so one failure cannot blank the rest.
   House rule: every number a figure displays is computed here from a spec, never typed into a label. */

const VMV = (function(){
  const P = Object.assign({}, (typeof C !== "undefined" ? C : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"}),
    { panel:"#15181f", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6" });
  function txt(g, x, y, s, o){
    o = o || {};
    return g.append("text").attr("x",x).attr("y",y).attr("font-size",o.size||11).attr("fill",o.fill||P.ink)
      .attr("text-anchor",o.anchor||"start").attr("font-weight",o.bold?600:null).text(s);
  }
  function fmtN(x){ // x in raw units
    const a = Math.abs(x);
    if (a >= 1e12) return (x/1e12).toFixed(2)+"T";
    if (a >= 1e9)  return (x/1e9).toFixed(a>=1e11?1:2)+"G";
    if (a >= 1e6)  return (x/1e6).toFixed(a>=1e8?0:1)+"M";
    if (a >= 1e3)  return (x/1e3).toFixed(1)+"k";
    return String(Math.round(x));
  }
  function fmtP(m){ return m >= 1000 ? (m/1000).toFixed(1)+" B" : (m >= 10 ? m.toFixed(0) : m.toFixed(1))+" M"; }
  function safe(name, fn){ try { fn(); } catch(e){ if (typeof console!=="undefined") console.error("[vision-models.viz] "+name+" failed:", e); } }
  return {P, txt, fmtN, fmtP, safe};
})();

/* ───────────────────────── 02 · design-axis matrix ───────────────────────── */
VMV.safe("axes", function(){
  const {P, txt} = VMV;
  const AXES = [
    {key:"mix",    label:"token mixing"},
    {key:"layout", label:"feature layout"},
    {key:"sup",    label:"supervision"},
    {key:"head",   label:"head / output"},
    {key:"res",    label:"input res"}
  ];
  /* spec: one row per model; values are the option labels shown in the cells */
  const M = [
    {name:"ResNet-50",    role:"reference", mix:"convolution", layout:"pyramid", sup:"IN-1k labels", head:"class logits", res:224},
    {name:"EfficientNet", role:"built",     mix:"convolution", layout:"pyramid", sup:"IN-1k labels", head:"class logits", res:600, note:"B7; B0 uses 224"},
    {name:"ViT",          role:"related",   mix:"global attn", layout:"single-scale", sup:"large labelled", head:"class logits", res:384, note:"fine-tuning resolution in the ViT tables"},
    {name:"DeiT",         role:"related",   mix:"global attn", layout:"single-scale", sup:"IN-1k + distil", head:"class logits", res:224},
    {name:"Swin",         role:"related",   mix:"window attn", layout:"pyramid", sup:"IN-1k labels", head:"class logits", res:224, note:"Swin-L uses IN-22k at 384"},
    {name:"ConvNeXt",     role:"reference", mix:"convolution", layout:"pyramid", sup:"IN-1k labels", head:"class logits", res:224, note:"XL uses IN-22k at 384"},
    {name:"DETR",         role:"planned",   mix:"conv + attn", layout:"single-scale", sup:"COCO boxes", head:"box set", res:800, note:"shorter side 800 on COCO"},
    {name:"DINOv2",       role:"planned",   mix:"global attn", layout:"single-scale", sup:"self-supervised", head:"frozen features", res:518, note:"224, then a short 518 phase"},
    {name:"SAM",          role:"planned",   mix:"window attn", layout:"single-scale", sup:"data engine", head:"prompted masks", res:1024}
  ];
  const resBin = r => r <= 224 ? "≤ 224" : (r <= 518 ? "225–599" : "≥ 600");
  const val = (m, k) => k === "res" ? resBin(m.res) : m[k];
  const cell = (m, k) => k === "res" ? String(m.res) : m[k];
  const pal = [P.A, P.B, P.good, P.purple, P.pink, P.teal, P.bad];
  const svg = d3.select("#ax-svg"), W = 640, H = 330;
  const sel = d3.select("#ax-axis");
  let row = 1;
  function draw(){
    const ax = sel.property("value") || "mix";
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const x0 = 112, cw = (W - x0 - 8) / AXES.length, y0 = 34, rh = 30;
    AXES.forEach((a,j)=> txt(g, x0 + j*cw + cw/2, 22, a.label, {anchor:"middle", size:11, fill:a.key===ax?P.B:P.muted, bold:a.key===ax}));
    const opts = [...new Set(M.map(m=>val(m,ax)))];
    const col = d3.scaleOrdinal().domain(opts).range(pal);
    M.forEach((m,i)=>{
      const y = y0 + i*rh, on = i===row;
      const rg = g.append("g").style("cursor","pointer").on("click", ()=>{ row=i; draw(); });
      rg.append("rect").attr("x",2).attr("y",y).attr("width",W-6).attr("height",rh-4).attr("rx",4)
        .attr("fill", on ? "#222838" : "transparent").attr("stroke", on ? P.B : "none");
      txt(rg, 10, y+17, m.name, {size:11.5, bold:true, fill:on?P.B:P.ink});
      txt(rg, 10, y+26, m.role, {size:8, fill:P.muted});
      AXES.forEach((a,j)=>{
        const cx = x0 + j*cw;
        const active = a.key===ax;
        rg.append("rect").attr("x",cx+3).attr("y",y+3).attr("width",cw-6).attr("height",rh-10).attr("rx",3)
          .attr("fill", active ? col(val(m,a.key)) : "#1b1f2a").attr("fill-opacity", active ? 0.28 : 1)
          .attr("stroke", active ? col(val(m,a.key)) : P.line);
        txt(rg, cx + cw/2, y+17, cell(m,a.key), {anchor:"middle", size:10.5, fill: active ? P.ink : P.muted});
      });
    });
    const counts = opts.map(o => ({o, n: M.filter(m=>val(m,ax)===o).length, who: M.filter(m=>val(m,ax)===o).map(m=>m.name)}));
    counts.sort((a,b)=>b.n-a.n);
    const axLabel = AXES.find(a=>a.key===ax).label;
    const m = M[row];
    const prof = AXES.map(a=>`${a.label}: <b>${cell(m,a.key)}</b>`).join(" · ");
    d3.select("#ax-read").html(
      `<b>${axLabel}</b> — ${counts.length} distinct options across ${M.length} models: ` +
      counts.map(c=>`${c.o} ×${c.n} (${c.who.join(", ")})`).join("; ") +
      `. <br><b>${m.name}</b> (${m.role}): ${prof}` + (m.note ? ` — ${m.note}` : "") + ".");
  }
  sel.on("change", draw);
  draw();
});

/* ───────────────────────── 03 · token / cost calculator ───────────────────────── */
VMV.safe("tokens", function(){
  const {P, txt, fmtN} = VMV;
  const ENC = { B:{name:"ViT-B", d:768, L:12}, L:{name:"ViT-L", d:1024, L:24}, H:{name:"ViT-H", d:1280, L:32} };
  const WIN = 14, NGLOBAL = 4, MLP = 4, PUB_VITB = 17.6;   // published ViT-B/16 @224 (DeiT, ConvNeXt tables)
  /* cost model — multiply-adds */
  function cost(e, Pz, res, mode){
    const G = Math.floor(res / Pz);
    const cls = mode === "global" ? 1 : 0;              // SAM-style encoders have no class token
    const N = G*G + cls, d = e.d;
    const proj = 4*N*d*d, mlp = 2*MLP*N*d*d;
    const attnGlobal = 2*N*N*d;
    const Gp = Math.ceil(G / WIN) * WIN;                 // pad grid to a multiple of the window
    const attnWin = 2*(Gp*Gp)*(WIN*WIN)*d;
    const nG = mode === "global" ? e.L : Math.min(NGLOBAL, e.L), nW = e.L - nG;
    const attnTot = nG*attnGlobal + nW*attnWin;
    const embed = G*G*(Pz*Pz*3)*d;
    const total = e.L*(proj + mlp) + attnTot + embed;
    return {G, N, proj, mlp, attnGlobal, attnWin, attnTot, embed, total, nG, nW, avgAttn: attnTot/e.L};
  }
  const svg = d3.select("#tk-svg"), W = 640, H = 300;
  const sM = d3.select("#tk-model"), sP = d3.select("#tk-patch"), sA = d3.select("#tk-attn"), sR = d3.select("#tk-res");
  function draw(){
    const e = ENC[sM.property("value")] || ENC.B;
    const Pz = +sP.property("value") || 16, mode = sA.property("value") || "global", res = +sR.property("value") || 224;
    d3.select("#tk-resv").text(res);
    const c = cost(e, Pz, res, mode);
    svg.selectAll("*").remove();
    const g = svg.append("g");
    /* left: one average layer, split */
    const parts = [
      {k:"projections 4Nd²", v:c.proj, col:P.A},
      {k:"attention (avg/layer)", v:c.avgAttn, col:P.B},
      {k:"MLP 8Nd²", v:c.mlp, col:P.good}
    ];
    const sum = parts.reduce((a,p)=>a+p.v,0);
    const bx = 20, bw = 200, by = 40, bh = 200;
    txt(g, bx, 22, "one layer, share of multiply-adds", {size:11, fill:P.muted});
    let acc = 0;
    parts.forEach(p=>{
      const h = bh * p.v / sum;
      g.append("rect").attr("x",bx).attr("y",by+acc).attr("width",60).attr("height",Math.max(h,0.5)).attr("fill",p.col).attr("fill-opacity",.8);
      txt(g, bx+70, by+acc+Math.max(h/2,6)+4, `${p.k}: ${(100*p.v/sum).toFixed(1)}%`, {size:10.5});
      acc += h;
    });
    txt(g, bx, by+bh+22, `N = ${c.N} tokens (${c.G}×${c.G} grid${mode==="global"?" + class token":""})`, {size:10.5, fill:P.muted});
    /* right: total cost vs resolution for both attention modes */
    const rx0 = 300, rx1 = W-16, ry0 = 30, ry1 = H-40;
    const grid = d3.range(224, 1024+1, 32);
    const serG = grid.map(r=>({r, t:cost(e,Pz,r,"global").total}));
    const serW = grid.map(r=>({r, t:cost(e,Pz,r,"sam").total}));
    const ymax = d3.max(serG.concat(serW), s=>s.t);
    const x = d3.scaleLinear().domain([224,1024]).range([rx0+40, rx1]);
    const y = d3.scaleLog().domain([d3.min(serG.concat(serW), s=>s.t)*0.8, ymax*1.2]).range([ry1, ry0]);
    y.ticks(4).filter((v,i,a)=> a.length<6 || i%2===0).forEach(v=>{
      g.append("line").attr("x1",rx0+40).attr("x2",rx1).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, rx0+36, y(v)+4, fmtN(v), {anchor:"end", size:9.5, fill:P.muted});
    });
    [224,448,640,832,1024].forEach(v=> txt(g, x(v), ry1+16, String(v), {anchor:"middle", size:9.5, fill:P.muted}));
    txt(g, (rx0+40+rx1)/2, ry1+32, "input side (pixels)", {anchor:"middle", size:10, fill:P.muted});
    txt(g, rx0+40, 20, `${e.name}/${Pz} encoder total, log scale`, {size:11, fill:P.muted});
    const line = d3.line().x(s=>x(s.r)).y(s=>y(s.t));
    g.append("path").datum(serG).attr("fill","none").attr("stroke",P.B).attr("stroke-width",2).attr("d",line);
    g.append("path").datum(serW).attr("fill","none").attr("stroke",P.teal).attr("stroke-width",2).attr("stroke-dasharray","5,3").attr("d",line);
    txt(g, x(1024)-4, y(serG[serG.length-1].t)-6, "global", {anchor:"end", size:10, fill:P.B});
    txt(g, x(1024)-4, y(serW[serW.length-1].t)+14, "windowed + 4 global", {anchor:"end", size:10, fill:P.teal});
    g.append("circle").attr("cx",x(res)).attr("cy",y(c.total)).attr("r",6).attr("fill",mode==="global"?P.B:P.teal).attr("stroke","#fff");
    /* readout */
    const base = cost(e, Pz, 224, mode).total;
    const other = cost(e, Pz, res, mode==="global"?"sam":"global").total;
    const attnShare = 100*c.attnTot/(c.total - c.embed);
    let ref = "";
    if (e.name==="ViT-B" && Pz===16 && res===224 && mode==="global")
      ref = ` This is the configuration the DeiT and ConvNeXt tables list as ${PUB_VITB} G for ViT-B/16 / DeiT-B; the formula gives ${(c.total/1e9).toFixed(2)} G.`;
    d3.select("#tk-read").html(
      `<b>${e.name}/${Pz} at ${res}²</b>, ${mode==="global" ? "global attention in all "+e.L+" layers" : c.nW+" windowed ("+WIN+"×"+WIN+") + "+c.nG+" global layers"}: ` +
      `N = <b>${c.N}</b> tokens; encoder total <b>${fmtN(c.total)}</b> multiply-adds ` +
      `(attention ${attnShare.toFixed(1)}% of the layer cost; patch embedding ${fmtN(c.embed)}). ` +
      `${(c.total/base).toFixed(1)}× the cost at 224²; the ${mode==="global"?"windowed":"global"} pattern would cost ${fmtN(other)} (${(other/c.total).toFixed(2)}×).` + ref);
  }
  [sM, sP, sA].forEach(s=>s.on("change", draw));
  sR.on("input", draw);
  draw();
});

/* ───────────────────────── 06 · lineage timeline ───────────────────────── */
VMV.safe("lineage", function(){
  const {P, txt, fmtP} = VMV;
  /* top1 = ImageNet top-1 (%) as reported; err = reported top-1 error where the paper gives error */
  const MS = [
    {name:"AlexNet",      t:2012.9,  err:40.7, params:60,   mix:"convolution", sup:"IN-1k labels", proto:"single CNN, ILSVRC-2012 val (paper reports 40.7% top-1 error)", std:true},
    {name:"VGG-19",       t:2014.7,  err:24.8, params:144,  mix:"convolution", sup:"IN-1k labels", proto:"single net, multi-crop + dense val (24.8% top-1 error)", std:true},
    {name:"ResNet-152",   t:2015.95, err:21.43, params:60,  mix:"convolution", sup:"IN-1k labels", proto:"single model, 10-crop val (21.43% top-1 error)", std:true},
    {name:"EfficientNet-B7", t:2019.4, top1:84.3, params:66, mix:"convolution", sup:"IN-1k labels", proto:"single crop at 600²", std:true},
    {name:"ViT-H/14",     t:2020.8,  top1:88.55, params:632, mix:"global attn", sup:"large labelled", proto:"pretrained on JFT-300M, fine-tuned on ImageNet", std:false},
    {name:"DeiT-B",       t:2020.97, top1:81.8, params:86,  mix:"global attn", sup:"IN-1k labels", proto:"ImageNet-1k only, 224²", std:true},
    {name:"Swin-L",       t:2021.23, top1:87.3, params:197, mix:"window attn", sup:"large labelled", proto:"ImageNet-22k pretraining, 384²", std:false},
    {name:"ConvNeXt-XL",  t:2022.03, top1:87.8, params:350, mix:"convolution", sup:"large labelled", proto:"ImageNet-22k pretraining, 384²", std:false},
    {name:"DINOv2 g/14",  t:2023.29, top1:86.5, params:1100, mix:"global attn", sup:"self-supervised", proto:"frozen features + linear probe, no labels in pretraining", std:false}
  ];
  MS.forEach(m=>{ if (m.top1 == null) m.top1 = 100 - m.err; });
  const LANE = [
    {name:"DETR", t:2020.4, params:41, mix:"conv + attn", sup:"COCO boxes", what:"box set prediction; COCO val AP", metric:42.0, unit:"AP"},
    {name:"SAM",  t:2023.26, params:null, mix:"window attn", sup:"data engine", what:"promptable masks from a ViT-H-sized encoder; SA-1B, 11 M images", metric:1.1e9, unit:"masks"},
    {name:"SAM 2", t:2024.58, params:null, mix:"window attn", sup:"data engine", what:"promptable masks in images and video; reported faster than SAM on images", metric:6, unit:"× faster than SAM (images)"}
  ];
  const svg = d3.select("#ln-svg"), W = 640, H = 330, m = {l:48, r:20, t:18, b:92};
  const sel = d3.select("#ln-colour");
  const x = d3.scaleLinear().domain([2012.3, 2025]).range([m.l, W-m.r]);
  const y = d3.scaleLinear().domain([55, 91]).range([H-m.b, m.t]);
  const rr = d3.scaleLog().domain([10, 1200]).range([4, 11]);
  let pick = 3;   // index into combined list
  const ALL = MS.map(d=>Object.assign({lane:false}, d)).concat(LANE.map(d=>Object.assign({lane:true}, d)));
  function draw(){
    const by = sel.property("value") || "mix";
    const opts = [...new Set(ALL.map(d=>d[by]))];
    const col = d3.scaleOrdinal().domain(opts).range([P.A, P.B, P.good, P.purple, P.pink, P.teal]);
    svg.selectAll("*").remove();
    const g = svg.append("g");
    [60,65,70,75,80,85,90].forEach(v=>{
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, m.l-6, y(v)+4, v+"%", {anchor:"end", size:9.5, fill:P.muted});
    });
    d3.range(2012, 2026, 2).forEach(v=> txt(g, x(v), H-m.b+14, String(v), {anchor:"middle", size:9.5, fill:P.muted}));
    txt(g, 4, m.t-4, "ImageNet top-1", {size:9.5, fill:P.muted});
    const laneY = H - 44;
    g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",laneY).attr("y2",laneY).attr("stroke",P.line);
    txt(g, m.l, laneY-10, "detection / segmentation lane (no ImageNet number)", {size:9.5, fill:P.muted});
    g.append("path").datum(MS.filter(d=>d.std)).attr("fill","none").attr("stroke",P.muted).attr("stroke-opacity",.5)
      .attr("d", d3.line().x(d=>x(d.t)).y(d=>y(d.top1)));
    ALL.forEach((d,i)=>{
      const cx = x(d.t), cy = d.lane ? laneY : y(d.top1), on = i===pick;
      const node = g.append("g").style("cursor","pointer").on("click", ()=>{ pick=i; draw(); });
      const r = d.params ? rr(d.params) : 6;
      const hollow = d.lane || !d.std;
      node.append("circle").attr("cx",cx).attr("cy",cy).attr("r",r)
        .attr("fill", hollow ? "#171a23" : col(d[by])).attr("stroke", col(d[by])).attr("stroke-width", on?3:2);
      if (on) node.append("circle").attr("cx",cx).attr("cy",cy).attr("r",r+4).attr("fill","none").attr("stroke",P.ink).attr("stroke-dasharray","2,2");
      const up = d.lane ? (i%2===0 ? 20 : -12) : -r-5;
      txt(node, cx, cy + up, d.name, {anchor:"middle", size:9.5, fill:on?P.B:P.ink, bold:on});
    });
    /* legend */
    opts.forEach((o,i)=>{
      const lx = m.l + i*95, ly = H-8;
      g.append("circle").attr("cx",lx+5).attr("cy",ly-4).attr("r",4).attr("fill",col(o));
      txt(g, lx+13, ly, o, {size:9, fill:P.muted});
    });
    const d = ALL[pick];
    let s = `<b>${d.name}</b> (${Math.floor(d.t)}) — ${by==="mix"?"token mixing":"supervision"}: ${d[by]}; parameters ${d.params ? fmtP(d.params) : "not listed here"}. `;
    if (!d.lane){
      s += `ImageNet top-1 <b>${d.top1.toFixed(2).replace(/0$/,"")}%</b> — protocol: ${d.proto}.`;
      const prev = MS.filter(p=>p.t < d.t && p.std).pop();
      if (prev && d.std) s += ` Change from the previous ImageNet-1k-only point, ${prev.name} (cropping protocols can differ — see the caption): ${(d.top1 - prev.top1 >= 0 ? "+" : "")}${(d.top1-prev.top1).toFixed(2)} points in ${(d.t-prev.t).toFixed(1)} years.`;
      else if (!d.std) s += ` Hollow marker: not comparable to the ImageNet-1k-only line.`;
    } else {
      const mv = d.unit === "masks" ? (d.metric/1e9).toFixed(1)+" B masks" : d.metric+" "+d.unit;
      s += `${d.what} — headline: <b>${mv}</b>.`;
    }
    d3.select("#ln-read").html(s);
  }
  sel.on("change", draw);
  draw();
});

/* ───────────────────────── 07 · accuracy vs cost scatter ───────────────────────── */
VMV.safe("pareto", function(){
  const {P, txt, fmtP} = VMV;
  /* ImageNet-1k-trained, single crop: params (M), FLOPs (G, multiply-adds), top-1 (%) */
  const PTS = [
    {name:"ResNet-50",  fam:"ResNet",   params:25.6, flops:4.1,  top1:76.1},
    {name:"DeiT-S",     fam:"DeiT",     params:22,   flops:4.6,  top1:79.8},
    {name:"DeiT-B",     fam:"DeiT",     params:86,   flops:17.6, top1:81.8},
    {name:"Swin-T",     fam:"Swin",     params:28,   flops:4.5,  top1:81.3},
    {name:"Swin-S",     fam:"Swin",     params:50,   flops:8.7,  top1:83.0},
    {name:"Swin-B",     fam:"Swin",     params:88,   flops:15.4, top1:83.5},
    {name:"ConvNeXt-T", fam:"ConvNeXt", params:29,   flops:4.5,  top1:82.1},
    {name:"ConvNeXt-S", fam:"ConvNeXt", params:50,   flops:8.7,  top1:83.1},
    {name:"ConvNeXt-B", fam:"ConvNeXt", params:89,   flops:15.4, top1:83.8}
  ];
  const EFF = [[5.3,0.39,77.1],[7.8,0.70,79.1],[9.2,1.0,80.1],[12,1.8,81.6],[19,4.2,82.9],[30,9.9,83.6],[43,19,84.0],[66,37,84.3]]
    .map((v,i)=>({name:"EfficientNet-B"+i, fam:"EfficientNet", params:v[0], flops:v[1], top1:v[2]}));
  const FC = {ResNet:"#9aa3b2", DeiT:"#5b9cff", Swin:"#2dd4bf", ConvNeXt:"#ffb454", EfficientNet:"#f472b6"};
  const svg = d3.select("#pa-svg"), W = 640, H = 320, m = {l:52, r:20, t:18, b:44};
  const sx = d3.select("#pa-x"), ce = d3.select("#pa-eff");
  let pick = "ConvNeXt-T";
  function frontier(pts, key){
    const s = pts.slice().sort((a,b)=>a[key]-b[key] || b.top1-a.top1);
    const f = []; let best = -Infinity;
    s.forEach(p=>{ if (p.top1 > best){ f.push(p); best = p.top1; } });
    return f;
  }
  function draw(){
    const key = sx.property("value") || "flops";
    const pts = PTS.concat(ce.property("checked") ? EFF : []);
    if (!pts.find(p=>p.name===pick)) pick = "ConvNeXt-T";
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const xd = d3.extent(pts, p=>p[key]);
    const x = d3.scaleLog().domain([xd[0]*0.7, xd[1]*1.4]).range([m.l, W-m.r]);
    const y = d3.scaleLinear().domain([d3.min(pts,p=>p.top1)-1, d3.max(pts,p=>p.top1)+1]).range([H-m.b, m.t]);
    x.ticks(6).forEach(v=>{
      const lab = x.tickFormat(6, "~g")(v); if (!lab) return;
      g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, x(v), H-m.b+14, lab, {anchor:"middle", size:9.5, fill:P.muted});
    });
    y.ticks(6).forEach(v=>{
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, m.l-6, y(v)+4, v+"%", {anchor:"end", size:9.5, fill:P.muted});
    });
    txt(g, (m.l+W-m.r)/2, H-8, key==="flops" ? "FLOPs (G multiply-adds, log)" : "parameters (M, log)", {anchor:"middle", size:10, fill:P.muted});
    const fr = frontier(pts, key);
    g.append("path").datum(fr).attr("fill","none").attr("stroke",P.ink).attr("stroke-opacity",.6).attr("stroke-dasharray","5,4")
      .attr("d", d3.line().x(p=>x(p[key])).y(p=>y(p.top1)));
    const effPts = pts.filter(p=>p.fam==="EfficientNet");
    if (effPts.length) g.append("path").datum(effPts).attr("fill","none").attr("stroke",FC.EfficientNet).attr("stroke-opacity",.35)
      .attr("d", d3.line().x(p=>x(p[key])).y(p=>y(p.top1)));
    pts.forEach(p=>{
      const on = p.name===pick;
      const node = g.append("g").style("cursor","pointer")
        .on("click", ()=>{ pick=p.name; draw(); })
        .on("mouseover", ()=>{ pick=p.name; draw(); });
      node.append("circle").attr("cx",x(p[key])).attr("cy",y(p.top1)).attr("r",on?7:5).attr("fill",FC[p.fam]).attr("stroke",on?"#fff":"none");
      if (p.fam!=="EfficientNet" || on || p.name.endsWith("0") || p.name.endsWith("7"))
        txt(node, x(p[key])+8, y(p.top1)+4, p.name, {size:9.5, fill:on?P.B:P.ink, bold:on});
    });
    Object.keys(FC).filter(f=>pts.some(p=>p.fam===f)).forEach((f,i)=>{
      g.append("circle").attr("cx",m.l+10+i*100).attr("cy",m.t+4).attr("r",4).attr("fill",FC[f]);
      txt(g, m.l+18+i*100, m.t+8, f, {size:9.5, fill:P.muted});
    });
    const p = pts.find(q=>q.name===pick);
    const onF = fr.includes(p);
    const better = pts.filter(q=>q!==p && q[key] <= p[key] && q.top1 > p.top1).map(q=>q.name);
    const ref = PTS[0];
    d3.select("#pa-read").html(
      `<b>${p.name}</b>: ${p.top1.toFixed(1)}% top-1 at ${p.flops} G FLOPs and ${fmtP(p.params)} parameters · ` +
      `vs ResNet-50: ${(p.top1-ref.top1>=0?"+":"")}${(p.top1-ref.top1).toFixed(1)} points for ${(p[key]/ref[key]).toFixed(2)}× the ${key==="flops"?"FLOPs":"parameters"}. ` +
      (onF ? `On the Pareto frontier for this x axis (${fr.length} of ${pts.length} points are).` :
             `Off the frontier: ${better.join(", ")} ${better.length===1?"is":"are"} at least as cheap and more accurate.`));
  }
  sx.on("change", draw); ce.on("change", draw);
  draw();
});

/* ───────────────────────── 09 · IoU and the ten COCO thresholds ───────────────────────── */
VMV.safe("iou", function(){
  const {P, txt} = VMV;
  const GT = {x:120, y:60, w:100, h:80};                     // ground-truth box (px)
  const TH = d3.range(10).map(i => +(0.50 + 0.05*i).toFixed(2));   // COCO thresholds
  const svg = d3.select("#iou-svg"), W = 640;
  const sdx = d3.select("#iou-dx"), sw = d3.select("#iou-w");
  function iou(a, b){
    const ix = Math.max(0, Math.min(a.x+a.w, b.x+b.w) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y+a.h, b.y+b.h) - Math.max(a.y, b.y));
    const inter = ix*iy, uni = a.w*a.h + b.w*b.h - inter;
    return {inter, uni, v: uni > 0 ? inter/uni : 0};
  }
  function draw(){
    const dx = +sdx.property("value"), w = +sw.property("value");
    d3.select("#iou-dxv").text(dx); d3.select("#iou-wv").text(w);
    const PR = {x:GT.x+dx, y:GT.y, w:w, h:GT.h};
    const r = iou(GT, PR);
    svg.selectAll("*").remove();
    const g = svg.append("g");
    g.append("rect").attr("x",20).attr("y",20).attr("width",330).attr("height",170).attr("fill","#1b1f2a").attr("stroke",P.line);
    g.append("rect").attr("x",GT.x).attr("y",GT.y).attr("width",GT.w).attr("height",GT.h).attr("fill",P.good).attr("fill-opacity",.15).attr("stroke",P.good).attr("stroke-width",2);
    g.append("rect").attr("x",PR.x).attr("y",PR.y).attr("width",PR.w).attr("height",PR.h).attr("fill",P.B).attr("fill-opacity",.15).attr("stroke",P.B).attr("stroke-width",2).attr("stroke-dasharray","5,3");
    const ix0 = Math.max(GT.x, PR.x), ix1 = Math.min(GT.x+GT.w, PR.x+PR.w);
    if (ix1 > ix0) g.append("rect").attr("x",ix0).attr("y",GT.y).attr("width",ix1-ix0).attr("height",GT.h).attr("fill",P.A).attr("fill-opacity",.45);
    txt(g, GT.x, GT.y-8, "true box", {size:10, fill:P.good});
    txt(g, PR.x+PR.w, PR.y+PR.h+14, "prediction", {anchor:"end", size:10, fill:P.B});
    /* threshold strip */
    const sx0 = 380, cw = 24;
    txt(g, sx0, 40, "COCO IoU thresholds", {size:11, fill:P.muted});
    TH.forEach((t,i)=>{
      const pass = r.v >= t, x = sx0 + i*cw;
      g.append("rect").attr("x",x).attr("y",52).attr("width",cw-3).attr("height",34).attr("rx",3)
        .attr("fill", pass ? P.good : P.bad).attr("fill-opacity", pass ? .55 : .25).attr("stroke", pass ? P.good : P.bad);
      txt(g, x+(cw-3)/2, 100, t.toFixed(2).slice(1), {anchor:"middle", size:9, fill:P.muted});
    });
    const passed = TH.filter(t=>r.v>=t).length;
    txt(g, sx0, 132, `IoU = ${r.v.toFixed(3)}`, {size:15, bold:true, fill:P.ink});
    txt(g, sx0, 154, `true positive at ${passed} of ${TH.length} thresholds`, {size:11, fill:P.muted});
    const bar = 230;
    g.append("rect").attr("x",sx0).attr("y",166).attr("width",bar).attr("height",10).attr("fill","#1b1f2a").attr("stroke",P.line);
    g.append("rect").attr("x",sx0).attr("y",166).attr("width",bar*r.v).attr("height",10).attr("fill",P.A);
    const at50 = r.v >= 0.5 ? "counts" : "does not count";
    d3.select("#iou-read").html(
      `Overlap ${r.inter} px², union ${r.uni} px² → IoU <b>${r.v.toFixed(3)}</b>. The prediction ${at50} as a hit at IoU 0.50 (the PASCAL VOC and AP₅₀ criterion) and passes <b>${passed}</b> of the ${TH.length} thresholds COCO AP averages over` +
      (passed === TH.length ? " — every one." : (passed === 0 ? " — none, so it is a false positive everywhere." : `, up to ${TH[passed-1].toFixed(2)}.`)));
  }
  sdx.on("input", draw); sw.on("input", draw);
  draw();
});
