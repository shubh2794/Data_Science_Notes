/* clip.viz.js — every interactive figure on models/clip.html.
   Loaded after data.js / notes.js (palette C comes from notes.js).
   Each figure is its own IIFE wrapped in try/catch so one failure cannot blank the rest.
   House rule: every number a figure displays is computed here from a spec or from a data
   array labelled with its source — never typed into a label. */

const CV = (function(){
  const P = Object.assign({}, (typeof C !== "undefined" ? C : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"}),
    { panel:"#15181f", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6" });

  function softmax(z){
    const m = Math.max.apply(null, z);
    const e = z.map(v => Math.exp(v - m)), s = e.reduce((a,b)=>a+b, 0);
    return e.map(v => v / s);
  }
  function logSigmoid(x){ return x >= 0 ? -Math.log1p(Math.exp(-x)) : x - Math.log1p(Math.exp(x)); }
  function norm(v){ const n = Math.sqrt(v.reduce((a,b)=>a+b*b, 0)) || 1; return v.map(x => x / n); }
  function dot(a,b){ let s = 0; for (let i=0;i<a.length;i++) s += a[i]*b[i]; return s; }
  /* symmetric InfoNCE on an N×N logit matrix → {rowP, colP, li, lt, loss} */
  function clipLoss(L){
    const N = L.length;
    const rowP = L.map(r => softmax(r));
    const colP = [];
    for (let j=0;j<N;j++) colP.push(softmax(L.map(r => r[j])));
    let li = 0, lt = 0;
    for (let i=0;i<N;i++){ li -= Math.log(rowP[i][i]); lt -= Math.log(colP[i][i]); }
    li /= N; lt /= N;
    return {rowP, colP, li, lt, loss:(li+lt)/2};
  }
  function fmtN(x){
    const a = Math.abs(x);
    if (a >= 1e12) return (x/1e12).toFixed(2)+"T";
    if (a >= 1e9)  return (x/1e9).toFixed(2)+"B";
    if (a >= 1e6)  return (x/1e6).toFixed(a>=1e8?0:1)+"M";
    if (a >= 1e3)  return (x/1e3).toFixed(a>=1e5?0:1)+"k";
    return String(Math.round(x));
  }
  function bytes(b){
    const u = ["B","KiB","MiB","GiB","TiB","PiB"]; let i = 0, v = b;
    while (v >= 1024 && i < u.length-1){ v /= 1024; i++; }
    return v.toFixed(v>=100?0:(v>=10?1:2))+" "+u[i];
  }
  function comma(x){ return Math.round(x).toLocaleString("en-US"); }
  function sup(e){ const m={"-":"⁻","0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"}; return String(e).split("").map(c=>m[c]).join(""); }
  function sci(x, p){ if (!isFinite(x) || x === 0) return "0"; const e = Math.floor(Math.log10(Math.abs(x))); return (x/Math.pow(10,e)).toFixed(p==null?2:p)+" × 10"+sup(e); }
  function txt(g, x, y, s, o){
    o = o || {};
    return g.append("text").attr("x",x).attr("y",y).attr("font-size",o.size||11).attr("fill",o.fill||P.ink)
      .attr("text-anchor",o.anchor||"start").attr("font-family",o.mono?"SF Mono,Menlo,monospace":null)
      .attr("font-weight",o.bold?600:null).text(s);
  }
  function safe(name, fn){ try { fn(); } catch(e){ if (typeof console !== "undefined") console.error("[clip.viz] "+name+" failed:", e); } }
  function lcg(seed){ let s = seed >>> 0; return () => { s = (1664525*s + 1013904223) >>> 0; return s/4294967296; }; }
  function gauss(r){ const u = Math.max(1e-12, r()), v = r(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

  /* ── Source: CLIP paper (arXiv 2103.00020) Tables 19 and 20 (hyperparameters),
        Figure 9 axis (ResNet image GFLOPs). ── */
  const MODELS = [
    {id:"RN50",    kind:"rn",  blocks:[3,4,6,3],   width:2048, res:224, embed:1024, tw:512,  th:8,  gflops:6.1},
    {id:"RN101",   kind:"rn",  blocks:[3,4,23,3],  width:2048, res:224, embed:512,  tw:512,  th:8,  gflops:9.9},
    {id:"RN50x4",  kind:"rn",  blocks:[4,6,10,6],  width:2560, res:288, embed:640,  tw:640,  th:10, gflops:21.5},
    {id:"RN50x16", kind:"rn",  blocks:[6,8,18,8],  width:3072, res:384, embed:768,  tw:768,  th:12, gflops:75.3},
    {id:"RN50x64", kind:"rn",  blocks:[3,15,36,10],width:4096, res:448, embed:1024, tw:1024, th:16, gflops:265.9},
    {id:"ViT-B/32",     kind:"vit", layers:12, width:768,  heads:12, patch:32, res:224, embed:512, tw:512, th:8},
    {id:"ViT-B/16",     kind:"vit", layers:12, width:768,  heads:12, patch:16, res:224, embed:512, tw:512, th:8},
    {id:"ViT-L/14",     kind:"vit", layers:24, width:1024, heads:16, patch:14, res:224, embed:768, tw:768, th:12},
    {id:"ViT-L/14@336", kind:"vit", layers:24, width:1024, heads:16, patch:14, res:336, embed:768, tw:768, th:12}
  ];
  const VOCAB = 49408, CTX = 77, TEXT_LAYERS = 12;   // Table 18 vocabulary; reference tokenizer context
  /* parameter counts computed from the configuration (reference implementation layout) */
  function textParams(m){
    const w = m.tw, L = TEXT_LAYERS;
    return VOCAB*w + CTX*w + L*(12*w*w + 13*w) + 2*w + w*m.embed;
  }
  function vitParams(m){
    if (m.kind !== "vit") return null;
    const d = m.width, g = m.res/m.patch, tok = g*g + 1;
    return 3*m.patch*m.patch*d + d + tok*d + 4*d + m.layers*(12*d*d + 13*d) + d*m.embed;
  }
  function tokens(m){ return m.kind === "vit" ? Math.pow(m.res/m.patch, 2) + 1 : Math.pow(m.res/32, 2) + 1; }

  /* ── Source: CLIP paper Table 9 / Table 11 column order (27 datasets). ── */
  const DS = ["Food101","CIFAR10","CIFAR100","Birdsnap","SUN397","StanfordCars","FGVCAircraft","VOC2007","DTD","OxfordPets",
              "Caltech101","Flowers102","MNIST","FER2013","STL10","EuroSAT","RESISC45","GTSRB","KITTI","Country211",
              "PCam","UCF101","Kinetics700","CLEVRCounts","HatefulMemes","RenderedSST2","ImageNet"];
  /* ── Source: CLIP paper Table 11 (zero-shot, 27 datasets, same order as DS). ── */
  const ZS = {
    "RN50":        [81.1,75.6,41.6,32.6,59.6,55.8,19.3,82.1,41.7,85.4,82.1,65.9,66.6,42.2,94.3,41.1,54.2,35.2,42.2,16.1,57.6,63.6,43.5,20.3,59.7,56.9,59.6],
    "RN101":       [83.9,81.0,49.0,37.2,59.9,62.3,19.5,82.4,43.9,86.2,85.1,65.7,59.3,45.6,96.7,33.1,58.5,38.3,33.3,16.9,55.2,62.2,46.7,28.1,61.1,64.2,62.2],
    "RN50x4":      [86.8,79.2,48.9,41.6,62.7,67.9,24.6,83.0,49.3,88.1,86.0,68.0,75.2,51.1,96.4,35.0,59.2,35.7,26.0,20.2,57.5,65.5,49.0,17.0,58.3,66.6,65.8],
    "RN50x16":     [90.5,82.2,54.2,45.9,65.0,72.3,30.3,82.9,52.8,89.7,87.6,71.9,80.0,56.0,97.8,40.3,64.4,39.6,33.9,24.0,62.5,68.7,53.4,17.6,58.9,67.6,70.5],
    "RN50x64":     [91.8,86.8,61.3,48.9,66.9,76.0,35.6,83.8,53.4,93.4,90.6,77.3,90.8,61.0,98.3,59.4,69.7,47.9,33.2,29.6,65.0,74.1,56.8,27.5,62.1,70.7,73.6],
    "ViT-B/32":    [84.4,91.3,65.1,37.8,63.2,59.4,21.2,83.1,44.5,87.0,87.9,66.7,51.9,47.3,97.2,49.4,60.3,32.2,39.4,17.8,58.4,64.5,47.8,24.8,57.6,59.6,63.2],
    "ViT-B/16":    [89.2,91.6,68.7,39.1,65.2,65.6,27.1,83.9,46.0,88.9,89.3,70.4,56.0,52.7,98.2,54.1,65.5,43.3,44.0,23.3,48.1,69.8,52.4,23.4,61.7,59.8,68.6],
    "ViT-L/14":    [92.9,96.2,77.9,48.3,67.7,77.3,36.1,84.1,55.3,93.5,92.6,78.7,87.2,57.5,99.3,59.9,71.6,50.3,23.1,32.7,58.8,76.2,60.3,24.3,63.3,64.0,75.3],
    "ViT-L/14@336":[93.8,95.7,77.5,49.5,68.4,78.8,37.2,84.3,55.7,93.5,92.8,78.3,88.3,57.7,99.4,59.6,71.7,52.3,21.9,34.9,63.0,76.9,61.3,24.8,63.3,67.9,76.2]
  };
  /* ── Source: CLIP paper Table 10 (linear probe, CLIP rows, same column order). ── */
  const LP = {
    "RN50":        [86.4,88.7,70.3,56.4,73.3,78.3,49.1,87.1,76.4,88.2,89.6,96.1,98.3,64.2,96.6,95.2,87.5,82.4,70.2,25.3,82.7,81.6,57.2,53.6,65.7,72.6,73.3],
    "RN101":       [88.9,91.1,73.5,58.6,75.1,84.0,50.7,88.0,76.3,91.0,92.0,96.4,98.4,65.2,97.8,95.9,89.3,82.4,73.6,26.6,82.8,84.0,60.3,50.3,68.2,73.3,75.7],
    "RN50x4":      [91.3,90.5,73.0,65.7,77.0,85.9,57.3,88.4,79.5,91.9,92.5,97.8,98.5,68.1,97.8,96.4,89.7,85.5,59.4,30.3,83.0,85.7,62.6,52.5,68.0,76.6,78.2],
    "RN50x16":     [93.3,92.2,74.9,72.8,79.2,88.7,62.7,89.0,79.1,93.5,93.7,98.3,98.9,68.7,98.6,97.0,91.4,89.0,69.2,34.8,83.5,88.0,66.3,53.8,71.1,80.0,81.5],
    "RN50x64":     [94.8,94.1,78.6,77.2,81.1,90.5,67.7,88.9,82.0,94.5,95.4,98.9,98.9,71.3,99.1,97.1,92.8,90.2,69.2,40.7,83.7,89.5,69.1,55.0,75.0,81.2,83.6],
    "ViT-B/32":    [88.8,95.1,80.5,58.5,76.6,81.8,52.0,87.7,76.5,90.0,93.0,96.9,99.0,69.2,98.3,97.0,90.5,85.3,66.2,27.8,83.9,85.5,61.7,52.1,66.7,70.8,76.1],
    "ViT-B/16":    [92.8,96.2,83.1,67.8,78.4,86.7,59.5,89.2,79.2,93.1,94.7,98.1,99.0,69.5,99.0,97.1,92.7,86.6,67.8,33.3,83.5,88.4,66.1,57.1,70.3,75.5,80.2],
    "ViT-L/14":    [95.2,98.0,87.5,77.0,81.8,90.9,69.4,89.6,82.1,95.1,96.5,99.2,99.2,72.2,99.7,98.2,94.1,92.5,64.7,42.9,85.8,91.5,72.0,57.8,76.2,80.8,83.9],
    "ViT-L/14@336":[95.9,97.9,87.4,79.9,82.2,91.5,71.6,89.9,83.0,95.1,96.0,99.2,99.2,72.9,99.7,98.1,94.9,92.4,69.2,46.4,85.6,92.0,73.0,60.3,77.3,80.5,85.4]
  };
  /* ── Source: CLIP paper Figure 5 (zero-shot CLIP minus linear probe on ResNet-50 features). ── */
  const FIG5 = {StanfordCars:28.9, Country211:23.2, Food101:22.5, Kinetics700:14.5, RenderedSST2:12.4, SUN397:7.8, UCF101:7.7,
    HatefulMemes:6.7, CIFAR10:3.9, CIFAR100:3.0, STL10:3.0, FER2013:2.8, Caltech101:2.0, ImageNet:1.9, OxfordPets:1.1, VOC2007:0.5,
    Birdsnap:-3.2, MNIST:-10.0, FGVCAircraft:-11.3, RESISC45:-11.9, Flowers102:-12.5, DTD:-16.6, CLEVRCounts:-18.2, GTSRB:-18.4,
    PCam:-19.5, KITTI:-34.0, EuroSAT:-37.1};
  function mean(a){ return a.reduce((x,y)=>x+y, 0) / a.length; }
  return {P, softmax, logSigmoid, norm, dot, clipLoss, fmtN, bytes, comma, sci, txt, safe, lcg, gauss,
          MODELS, textParams, vitParams, tokens, DS, ZS, LP, FIG5, mean};
})();

/* ───────────────────────── 03 · the two towers, with shapes ───────────────────────── */
CV.safe("pipe", function(){
  const {P, MODELS, textParams, vitParams, tokens, txt, fmtN, bytes, comma} = CV;
  const svg = d3.select("#pipe-svg"), sel = d3.select("#pipe-model");
  MODELS.forEach((m,i) => sel.append("option").attr("value", i).text(m.id));
  sel.property("value", 5);
  function box(g, x, y, w, h, title, shape, col){
    g.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",h).attr("rx",6)
      .attr("fill",P.panel).attr("stroke",col).attr("stroke-width",1.4);
    txt(g, x+w/2, y+15, title, {anchor:"middle", size:10.5, fill:col, bold:true});
    txt(g, x+w/2, y+31, shape, {anchor:"middle", size:10, mono:true, fill:P.ink});
  }
  function arrow(g, x1, y1, x2, y2){
    g.append("line").attr("x1",x1).attr("y1",y1).attr("x2",x2).attr("y2",y2).attr("stroke",P.muted).attr("stroke-width",1.2);
    g.append("path").attr("d",`M${x2},${y2} l-6,-3.5 l0,7 z`).attr("fill",P.muted);
  }
  function draw(){
    svg.selectAll("*").remove();
    const m = MODELS[+sel.property("value")];
    const mode = d3.select("#pipe-mode").property("value");
    const k = +d3.select("#pipe-n").property("value"), N = Math.pow(2, k);
    d3.select("#pipe-nv").text((mode==="train"?"N = ":"K = ") + comma(N));
    const g = svg.append("g");
    const nI = mode === "train" ? N : 1, nT = N;
    const nL = mode === "train" ? "N" : "1", tL = mode === "train" ? "N" : "K";
    // image path (top row)
    const yI = 30, yT = 190, bw = 118, bh = 42;
    txt(g, 10, yI-10, "image tower", {fill:P.A, bold:true});
    box(g, 10, yI, bw, bh, "pixels", `[${nL},3,${m.res},${m.res}]`, P.A);
    let feat, mid;
    if (m.kind === "vit"){
      const gr = m.res/m.patch;
      mid = `[${nL},${tokens(m)},${m.width}]`;
      box(g, 150, yI, bw, bh, `${gr}×${gr} patches + [CLS]`, mid, P.A);
      box(g, 290, yI, bw, bh, `${m.layers} blocks → [CLS]`, `[${nL},${m.width}]`, P.A);
      feat = m.width;
    } else {
      const gr = m.res/32;
      mid = `[${nL},${gr}×${gr},${m.width}]`;
      box(g, 150, yI, bw, bh, "ResNet feature map", mid, P.A);
      box(g, 290, yI, bw, bh, `attn pool (${gr*gr}+1 tok)`, `[${nL},${m.embed}]`, P.A);
      feat = m.embed;
    }
    box(g, 430, yI, bw, bh, m.kind==="vit" ? "W_i proj + L2" : "L2 normalise", `[${nL},${m.embed}]`, P.A);
    arrow(g, 10+bw, yI+bh/2, 150, yI+bh/2); arrow(g, 150+bw, yI+bh/2, 290, yI+bh/2); arrow(g, 290+bw, yI+bh/2, 430, yI+bh/2);
    // text path (bottom row)
    txt(g, 10, yT-10, "text tower", {fill:P.B, bold:true});
    box(g, 10, yT, bw, bh, "BPE ids", `[${tL},${77}]`, P.B);
    box(g, 150, yT, bw, bh, "token + position emb", `[${tL},77,${m.tw}]`, P.B);
    box(g, 290, yT, bw, bh, "12 causal blocks → [EOS]", `[${tL},${m.tw}]`, P.B);
    box(g, 430, yT, bw, bh, "W_t proj + L2", `[${tL},${m.embed}]`, P.B);
    arrow(g, 10+bw, yT+bh/2, 150, yT+bh/2); arrow(g, 150+bw, yT+bh/2, 290, yT+bh/2); arrow(g, 290+bw, yT+bh/2, 430, yT+bh/2);
    // meeting point
    const mx = 575, my = 125;
    g.append("rect").attr("x",mx-45).attr("y",my-28).attr("width",100).attr("height",56).attr("rx",6)
      .attr("fill",P.panel).attr("stroke",P.good).attr("stroke-width",1.6);
    txt(g, mx+5, my-10, "exp(t) · U·Vᵀ", {anchor:"middle", size:10.5, fill:P.good, bold:true});
    txt(g, mx+5, my+8, `[${nL},${tL}]`, {anchor:"middle", size:11, mono:true});
    txt(g, mx+5, my+22, mode==="train"?"symmetric CE":"softmax over K", {anchor:"middle", size:9.5, fill:P.muted});
    g.append("path").attr("d",`M${430+bw/2},${yI+bh} C${430+bw/2},${my} ${mx-80},${my-10} ${mx-45},${my-10}`).attr("fill","none").attr("stroke",P.A).attr("stroke-width",1.2);
    g.append("path").attr("d",`M${430+bw/2},${yT} C${430+bw/2},${my} ${mx-80},${my+10} ${mx-45},${my+10}`).attr("fill","none").attr("stroke",P.B).attr("stroke-width",1.2);
    txt(g, 10, 300, `d_e = ${m.embed} · text width ${m.tw}, ${m.th} heads · input ${m.res}px · ${m.kind==="vit" ? tokens(m)+" image tokens" : "ResNet blocks ("+m.blocks.join(", ")+")"}`, {size:10.5, fill:P.muted});
    const logits = nI * nT, tp = textParams(m), vp = vitParams(m);
    const cache = mode === "zs" ? " · the K class embeddings are computed once and cached" : "";
    d3.select("#pipe-read").html(
      `<b>${m.id}</b>: logit matrix ${comma(nI)} × ${comma(nT)} = <b>${fmtN(logits)}</b> entries (${bytes(logits*4)} in fp32)${cache}. ` +
      `Text tower ${fmtN(tp)} parameters` + (vp ? `, image tower ${fmtN(vp)}, total <b>${fmtN(vp+tp)}</b>.` : `; for ResNets the attention pool's output layer is the image projection (image GFLOPs ${m.gflops}).`));
  }
  sel.on("change", draw);
  d3.select("#pipe-mode").on("change", draw);
  d3.select("#pipe-n").on("input", draw);
  draw();
});

/* ───────────────────────── 04 · draggable toy embeddings, live loss ───────────────────────── */
CV.safe("mat", function(){
  const {P, clipLoss, txt} = CV;
  const svg = d3.select("#mat-svg");
  const names = ["dog","cat","car","tree"], cols = [P.A, P.B, P.good, P.purple];
  const init = {img:[25, 55, 100, 150], txt:[40, 67, 90, 140]};
  let st = {img:init.img.slice(), txt:init.txt.slice()};
  const cx = 150, cy = 160, R = 110;
  function rad(a){ return a*Math.PI/180; }
  function compute(){
    const s = +d3.select("#mat-s").property("value");
    const cos = st.img.map(a => st.txt.map(b => Math.cos(rad(a-b))));
    const L = cos.map(r => r.map(c => s*c));
    return {s, cos, L, r: clipLoss(L)};
  }
  function draw(){
    svg.selectAll("*").remove();
    const {s, L, r} = compute();
    d3.select("#mat-sv").text(s);
    const g = svg.append("g");
    g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",R).attr("fill","none").attr("stroke",P.line);
    g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",2).attr("fill",P.muted);
    txt(g, cx, 22, "unit circle: squares = images, circles = texts", {anchor:"middle", size:10, fill:P.muted});
    names.forEach((nm,i) => {
      const ai = rad(st.img[i]), at = rad(st.txt[i]);
      const xi = cx + R*Math.cos(ai), yi = cy - R*Math.sin(ai), xt = cx + R*Math.cos(at), yt = cy - R*Math.sin(at);
      g.append("line").attr("x1",cx).attr("y1",cy).attr("x2",xi).attr("y2",yi).attr("stroke",cols[i]).attr("stroke-opacity",0.35);
      g.append("line").attr("x1",cx).attr("y1",cy).attr("x2",xt).attr("y2",yt).attr("stroke",cols[i]).attr("stroke-opacity",0.35).attr("stroke-dasharray","3 3");
      const sq = g.append("rect").attr("class","dragpt").attr("x",xi-7).attr("y",yi-7).attr("width",14).attr("height",14).attr("fill",cols[i]).attr("stroke","#000").style("cursor","grab");
      const ci = g.append("circle").attr("class","dragpt").attr("cx",xt).attr("cy",yt).attr("r",7).attr("fill","none").attr("stroke",cols[i]).attr("stroke-width",2.5).style("cursor","grab");
      txt(g, xi + (Math.cos(ai)>=0?10:-10), yi-9, "I:"+nm, {size:9.5, fill:cols[i], anchor:Math.cos(ai)>=0?"start":"end"});
      txt(g, xt + (Math.cos(at)>=0?10:-10), yt+14, "T:"+nm, {size:9.5, fill:cols[i], anchor:Math.cos(at)>=0?"start":"end"});
      const dragTo = (key) => d3.drag().on("drag", function(ev){
        const [px, py] = d3.pointer(ev, svg.node());
        let a = Math.atan2(cy - py, px - cx) * 180/Math.PI; if (a < 0) a += 360;
        st[key][i] = Math.round(a); draw();
      });
      sq.call(dragTo("img")); ci.call(dragTo("txt"));
    });
    // matrix
    const view = d3.select("#mat-view").property("value");
    const M = view === "logit" ? L : (view === "row" ? r.rowP : names.map((_,i)=>names.map((__,j)=>r.colP[j][i])));
    const ox = 330, oy = 60, cs = 52;
    const vals = [].concat.apply([], M);
    const lo = view === "logit" ? -s : 0, hi = view === "logit" ? s : 1;
    const col = d3.scaleSequential(d3.interpolateViridis).domain([lo, hi]);
    names.forEach((nm,j) => txt(g, ox + j*cs + cs/2, oy-8, "T:"+nm, {anchor:"middle", size:9.5, fill:cols[j]}));
    names.forEach((nm,i) => {
      txt(g, ox-6, oy + i*cs + cs/2 + 4, "I:"+nm, {anchor:"end", size:9.5, fill:cols[i]});
      names.forEach((_,j) => {
        const v = M[i][j];
        g.append("rect").attr("x",ox+j*cs).attr("y",oy+i*cs).attr("width",cs-3).attr("height",cs-3).attr("rx",3)
          .attr("fill",col(v)).attr("stroke", i===j ? P.good : "none").attr("stroke-width", 2);
        txt(g, ox+j*cs+cs/2-1, oy+i*cs+cs/2+4, view==="logit" ? v.toFixed(1) : v.toFixed(2), {anchor:"middle", size:10, fill: (v-lo)/(hi-lo) > 0.55 ? "#111" : "#eee"});
      });
    });
    txt(g, ox + 2*cs, oy + 4*cs + 18, view==="logit" ? "logits = s · cos" : (view==="row" ? "each row sums to 1" : "each column sums to 1 (shown in place)"), {anchor:"middle", size:10, fill:P.muted});
    const acc = names.filter((_,i) => r.rowP[i].indexOf(Math.max.apply(null, r.rowP[i])) === i).length;
    d3.select("#mat-read").html(`ℒ_I (image→text) = <b>${r.li.toFixed(3)}</b> · ℒ_T (text→image) = <b>${r.lt.toFixed(3)}</b> · ℒ = <b>${r.loss.toFixed(3)}</b> nats (a random model scores ln 4 = ${Math.log(4).toFixed(3)}) · rows whose argmax is the true caption: <b>${acc}/4</b>` + (vals.length ? "" : ""));
  }
  d3.select("#mat-s").on("input", draw);
  d3.select("#mat-view").on("change", draw);
  d3.select("#mat-reset").on("click", () => { st = {img:init.img.slice(), txt:init.txt.slice()}; draw(); });
  d3.select("#mat-shuffle").on("click", () => { st.txt = [st.txt[1], st.txt[2], st.txt[3], st.txt[0]]; draw(); });
  draw();
});

/* ───────────────────────── 07 · temperature explorer ───────────────────────── */
CV.safe("temp", function(){
  const {P, softmax, txt} = CV;
  const svg = d3.select("#temp-svg");
  const NEG = [0.27, 0.24, 0.20, 0.15, 0.12, 0.08, 0.05];   // toy negatives (cosines), in the range CLIPScore reports
  function row(s, pos){ return softmax([pos].concat(NEG).map(c => s*c)); }
  function draw(){
    svg.selectAll("*").remove();
    const s = Math.max(1, +d3.select("#temp-s").property("value"));
    const pos = +d3.select("#temp-pos").property("value") / 100;
    d3.select("#temp-sv").text("s = " + s + "  (τ = " + (1/s).toFixed(3) + ")");
    d3.select("#temp-posv").text(pos.toFixed(2));
    const p = row(s, pos), g = svg.append("g");
    // bars
    const bx = 40, bw = 250, by = 30, bh = 200;
    const y = d3.scaleLinear().domain([0,1]).range([by+bh, by]);
    const labels = ["pos"].concat(NEG.map((_,i)=>"n"+(i+1)));
    const step = bw / labels.length;
    g.append("line").attr("x1",bx).attr("x2",bx+bw).attr("y1",by+bh).attr("y2",by+bh).attr("stroke",P.line);
    [0,0.5,1].forEach(v => { txt(g, bx-6, y(v)+4, v.toFixed(1), {anchor:"end", size:9.5, fill:P.muted}); });
    p.forEach((v,i) => {
      g.append("rect").attr("x",bx+i*step+3).attr("y",y(v)).attr("width",step-6).attr("height",Math.max(0.5,by+bh-y(v))).attr("fill", i===0 ? P.good : P.bad).attr("opacity", i===0?0.95:0.75);
      txt(g, bx+i*step+step/2, by+bh+14, labels[i], {anchor:"middle", size:9.5, fill:P.muted});
      txt(g, bx+i*step+step/2, by+bh+26, (i===0?pos:NEG[i-1]).toFixed(2), {anchor:"middle", size:9, fill:P.muted, mono:true});
    });
    txt(g, bx+bw/2, 18, "softmax over one row (label: cosine)", {anchor:"middle", size:10.5, fill:P.muted});
    // loss vs scale
    const lx = 350, lw = 270, ly = 30, lh = 200;
    const xs = d3.scaleLog().domain([1,100]).range([lx, lx+lw]);
    const curve = d3.range(0, 101).map(i => { const sv = Math.pow(100, i/100); return [sv, -Math.log(row(sv, pos)[0])]; });
    const ymax = Math.max(d3.max(curve, d=>d[1]), 0.1);
    const yl = d3.scaleLinear().domain([0, ymax]).range([ly+lh, ly]);
    g.append("g").attr("transform",`translate(0,${ly+lh})`).call(d3.axisBottom(xs).tickValues([1,3,10,30,100]).tickFormat(d3.format("~g"))).attr("class","axis");
    g.append("g").attr("transform",`translate(${lx},0)`).call(d3.axisLeft(yl).ticks(4)).attr("class","axis");
    g.append("path").datum(curve).attr("fill","none").attr("stroke",P.A).attr("stroke-width",2)
      .attr("d", d3.line().x(d=>xs(d[0])).y(d=>yl(d[1])));
    [[1/0.07,"init 14.29"],[100,"cap 100"]].forEach(([v,l]) => {
      g.append("line").attr("x1",xs(v)).attr("x2",xs(v)).attr("y1",ly).attr("y2",ly+lh).attr("stroke",P.muted).attr("stroke-dasharray","4 3");
      txt(g, xs(v)-3, ly+10, l, {anchor:"end", size:9.5, fill:P.muted});
    });
    const cur = -Math.log(p[0]);
    g.append("circle").attr("cx",xs(s)).attr("cy",yl(cur)).attr("r",5).attr("fill",P.B);
    txt(g, lx+lw/2, 18, "row loss −ln P(pos) vs scale s", {anchor:"middle", size:10.5, fill:P.muted});
    txt(g, lx+lw/2, ly+lh+32, "logit scale s (log axis)", {anchor:"middle", size:10, fill:P.muted});
    const negMass = 1 - p[0];
    const hard = negMass > 0 ? p[1] / negMass : 0;
    const hardest = NEG[0] > pos ? " — the 'hardest negative' is now more similar than the positive" : "";
    d3.select("#temp-read").html(`s = <b>${s}</b>: P(positive) = <b>${p[0].toFixed(3)}</b>, row loss <b>${cur.toFixed(3)}</b> nats · of the push on negatives, <b>${(100*hard).toFixed(1)}%</b> goes to the hardest one (cos ${NEG[0]})${hardest} · argmax is the positive: <b>${p[0] === Math.max.apply(null,p) ? "yes" : "no"}</b> at every s.`);
  }
  d3.select("#temp-s").on("input", draw);
  d3.select("#temp-pos").on("input", draw);
  draw();
});

/* ───────────────────────── 08 · batch size, negatives, memory ───────────────────────── */
CV.safe("batch", function(){
  const {P, txt, bytes, comma, fmtN} = CV;
  const svg = d3.select("#batch-svg"), W = 640, H = 300, m = {l:64, r:20, t:20, b:44};
  function draw(){
    svg.selectAll("*").remove();
    const k = +d3.select("#batch-k").property("value");
    let mm = +d3.select("#batch-m").property("value");
    if (mm > k) mm = k;
    const N = Math.pow(2,k), D = Math.pow(2,mm), n = N/D;
    d3.select("#batch-kv").text(comma(N)); d3.select("#batch-mv").text(comma(D));
    const full = N*N*4, shard = 2*n*N*4;
    const g = svg.append("g");
    const x = d3.scaleLog().base(2).domain([256, 1048576]).range([m.l, W-m.r]);
    const y = d3.scaleLog().domain([1e4, 1e13]).range([H-m.b, m.t]);
    g.append("g").attr("transform",`translate(0,${H-m.b})`).attr("class","axis").call(d3.axisBottom(x).tickValues([256,1024,4096,16384,65536,262144,1048576]).tickFormat(d=>fmtN(d)));
    g.append("g").attr("transform",`translate(${m.l},0)`).attr("class","axis").call(d3.axisLeft(y).tickValues([1e4,1e6,1e8,1e10,1e12]).tickFormat(d=>bytes(d)));
    const Ns = d3.range(8, 21).map(e => Math.pow(2,e));
    g.append("path").datum(Ns).attr("fill","none").attr("stroke",P.bad).attr("stroke-width",2).attr("d", d3.line().x(d=>x(d)).y(d=>y(d*d*4)));
    g.append("path").datum(Ns.filter(v => v >= D)).attr("fill","none").attr("stroke",P.good).attr("stroke-width",2).attr("d", d3.line().x(d=>x(d)).y(d=>y(2*(d/D)*d*4)));
    g.append("circle").attr("cx",x(N)).attr("cy",y(full)).attr("r",5).attr("fill",P.bad);
    g.append("circle").attr("cx",x(N)).attr("cy",y(shard)).attr("r",5).attr("fill",P.good);
    g.append("line").attr("x1",x(32768)).attr("x2",x(32768)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.muted).attr("stroke-dasharray","4 3");
    txt(g, x(32768)+4, m.t+12, "CLIP: 32,768", {size:9.5, fill:P.muted});
    txt(g, m.l+8, m.t+12, "full N × N fp32 logits", {size:10, fill:P.bad});
    txt(g, m.l+8, m.t+26, `per-device strips, D = ${comma(D)}`, {size:10, fill:P.good});
    txt(g, (m.l+W-m.r)/2, H-8, "batch size N (log₂ axis)", {anchor:"middle", size:10, fill:P.muted});
    d3.select("#batch-read").html(`N = <b>${comma(N)}</b>: <b>${comma(N-1)}</b> negatives per image and per text · ${fmtN(N*N-N)} wrong pairings per step · random-model loss ln N = <b>${Math.log(N).toFixed(2)}</b> nats · full logit matrix <b>${bytes(full)}</b> · with D = ${comma(D)} devices (local batch ${comma(n)}), two ${comma(n)} × ${comma(N)} strips = <b>${bytes(shard)}</b> per device.`);
  }
  d3.select("#batch-k").on("input", draw);
  d3.select("#batch-m").on("input", draw);
  draw();
});

/* ───────────────────────── 10 · per-query cap on a long-tailed pool ───────────────────────── */
CV.safe("wit", function(){
  const {P, txt, fmtN, comma} = CV;
  const svg = d3.select("#wit-svg"), W = 640, H = 300, m = {l:60, r:20, t:20, b:44};
  /* Source for the calibration point and query counts: MetaCLIP (arXiv 2309.16671) — 500k entries,
     114k with no matches (so 386k matched), 16k entries above 20k matches. The Zipf shape is a toy. */
  const Q = 386000, RANK0 = 16000, C0 = 20000;
  const CAPS = [500,1000,2000,3000,5000,7500,10000,12500,15000,17500,20000,25000,30000,40000,50000,75000,100000,200000,500000,1000000,Infinity];
  function count(r, a){ return C0 * Math.pow(RANK0 / r, a); }
  function draw(){
    svg.selectAll("*").remove();
    const cap = CAPS[+d3.select("#wit-cap").property("value")];
    const a = +d3.select("#wit-a").property("value") / 100;
    d3.select("#wit-capv").text(isFinite(cap) ? comma(cap) : "no cap");
    d3.select("#wit-av").text(a.toFixed(2));
    let raw = 0, kept = 0, over = 0, topRaw = 0, topKept = 0;
    const top = Math.round(Q * 0.01);
    for (let r = 1; r <= Q; r++){
      const c = count(r, a), k = Math.min(c, cap);
      raw += c; kept += k; if (c > cap) over++;
      if (r <= top){ topRaw += c; topKept += k; }
    }
    const g = svg.append("g");
    const x = d3.scaleLog().domain([1, Q]).range([m.l, W-m.r]);
    const y = d3.scaleLog().domain([0.1, 1e10]).range([H-m.b, m.t]);
    g.append("g").attr("transform",`translate(0,${H-m.b})`).attr("class","axis").call(d3.axisBottom(x).tickValues([1,10,100,1000,1e4,1e5]).tickFormat(d=>fmtN(d)));
    g.append("g").attr("transform",`translate(${m.l},0)`).attr("class","axis").call(d3.axisLeft(y).tickValues([1,100,1e4,1e6,1e8,1e10]).tickFormat(d=>fmtN(d)));
    const rs = d3.range(0, 101).map(i => Math.pow(Q, i/100));
    g.append("path").datum(rs).attr("fill","none").attr("stroke",P.muted).attr("stroke-width",1.6).attr("stroke-dasharray","5 3").attr("d", d3.line().x(r=>x(r)).y(r=>y(count(r,a))));
    g.append("path").datum(rs).attr("fill","none").attr("stroke",P.A).attr("stroke-width",2.2).attr("d", d3.line().x(r=>x(r)).y(r=>y(Math.min(count(r,a),cap))));
    if (isFinite(cap)) { g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(cap)).attr("y2",y(cap)).attr("stroke",P.B).attr("stroke-dasharray","3 3"); txt(g, W-m.r-4, y(cap)-5, "cap", {anchor:"end", size:9.5, fill:P.B}); }
    txt(g, m.l+10, m.t+12, "raw pool (toy Zipf)", {size:10, fill:P.muted});
    txt(g, m.l+10, m.t+26, "after the per-query cap", {size:10, fill:P.A});
    txt(g, (m.l+W-m.r)/2, H-8, "query rank (log)  ·  y: pairs matching that query (log)", {anchor:"middle", size:10, fill:P.muted});
    d3.select("#wit-read").html(`Raw pool <b>${fmtN(raw)}</b> matches → kept <b>${fmtN(kept)}</b> (${(100*kept/raw).toFixed(1)}%) · <b>${comma(over)}</b> of ${comma(Q)} matched queries hit the cap · the top 1% of queries hold <b>${(100*topRaw/raw).toFixed(1)}%</b> of the raw pool but <b>${(100*topKept/kept).toFixed(1)}%</b> after capping · rank-1 query: ${fmtN(count(1,a))} → ${fmtN(Math.min(count(1,a),cap))}.`);
  }
  d3.select("#wit-cap").on("input", draw);
  d3.select("#wit-a").on("input", draw);
  draw();
});

/* ───────────────────────── 11 · objective efficiency multipliers ───────────────────────── */
CV.safe("obj", function(){
  const {P, txt, fmtN} = CV;
  const svg = d3.select("#obj-svg"), W = 640, H = 290, m = {l:54, r:20, t:24, b:44};
  /* Source for the factors: CLIP paper §2.3 / Figure 2 — bag-of-words prediction learns 3× faster than the
     Transformer LM; contrastive a further 4×. Curve shape (log-linear) and y-scale are illustrative. */
  const F_BOW = 3, F_CON = 4, XMAX = 400e6, N0 = 1e6;
  const series = [
    {name:"Bag-of-words contrastive (CLIP)", n0:N0,                col:P.good},
    {name:"Bag-of-words prediction",         n0:N0*F_CON,          col:P.B},
    {name:"Transformer language model",      n0:N0*F_CON*F_BOW,    col:P.bad}
  ];
  const span = Math.log(XMAX / N0);
  function yOf(n, s){ return Math.max(0, Math.log(n / s.n0) / span); }
  function nFor(y, s){ return s.n0 * Math.exp(y * span); }
  function draw(){
    svg.selectAll("*").remove();
    const yt = +d3.select("#obj-y").property("value") / 100;
    d3.select("#obj-yv").text(yt.toFixed(2));
    const g = svg.append("g");
    const x = d3.scaleLog().domain([2e6, XMAX]).range([m.l, W-m.r]);
    const y = d3.scaleLinear().domain([0, 1]).range([H-m.b, m.t]);
    g.append("g").attr("transform",`translate(0,${H-m.b})`).attr("class","axis").call(d3.axisBottom(x).tickValues([2e6,33e6,67e6,134e6,268e6,400e6]).tickFormat(d=>fmtN(d)));
    g.append("g").attr("transform",`translate(${m.l},0)`).attr("class","axis").call(d3.axisLeft(y).ticks(5));
    const ns = d3.range(0, 101).map(i => 2e6 * Math.pow(XMAX/2e6, i/100));
    series.forEach(s => g.append("path").datum(ns).attr("fill","none").attr("stroke",s.col).attr("stroke-width",2).attr("d", d3.line().x(n=>x(n)).y(n=>y(yOf(n,s)))));
    g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(yt)).attr("y2",y(yt)).attr("stroke",P.muted).attr("stroke-dasharray","4 3");
    const need = series.map(s => nFor(yt, s));
    need.forEach((n,i) => { if (n <= XMAX) g.append("circle").attr("cx",x(n)).attr("cy",y(yt)).attr("r",5).attr("fill",series[i].col); });
    series.forEach((s,i) => txt(g, m.l+8, m.t+12+14*i, s.name, {size:10, fill:s.col}));
    txt(g, (m.l+W-m.r)/2, H-8, "images processed (log axis, the paper's ticks)  ·  y: zero-shot accuracy, illustrative units", {anchor:"middle", size:10, fill:P.muted});
    const show = n => n <= XMAX ? fmtN(n) : fmtN(n) + " (beyond 400M)";
    d3.select("#obj-read").html(`To reach the dashed level: contrastive needs <b>${show(need[0])}</b> images, bag-of-words prediction <b>${show(need[1])}</b> (${(need[1]/need[0]).toFixed(1)}×), the Transformer LM <b>${show(need[2])}</b> (${(need[2]/need[0]).toFixed(1)}×). The ratios are the paper's; the absolute image counts depend on the illustrative curve.`);
  }
  d3.select("#obj-y").on("input", draw);
  draw();
});

/* ───────────────────────── 13 · model zoo ───────────────────────── */
CV.safe("zoo", function(){
  const {P, txt, MODELS, ZS, LP, DS, mean, vitParams, textParams, tokens, fmtN} = CV;
  const svg = d3.select("#zoo-svg"), W = 640, H = 300, m = {l:48, r:16, t:24, b:54};
  let pick = 8;
  const iIN = DS.indexOf("ImageNet");
  function metric(id, k){
    if (k === "in") return ZS[id][iIN];
    if (k === "zs27") return mean(ZS[id]);
    if (k === "lp27") return mean(LP[id]);
    return mean(LP[id]) - mean(ZS[id]);
  }
  function draw(){
    svg.selectAll("*").remove();
    const k = d3.select("#zoo-metric").property("value");
    const vals = MODELS.map(mo => metric(mo.id, k));
    const g = svg.append("g");
    const x = d3.scaleBand().domain(MODELS.map(mo=>mo.id)).range([m.l, W-m.r]).padding(0.2);
    const lo = k === "gap" ? 0 : Math.floor(Math.min.apply(null, vals)/5)*5 - 5;
    const hi = Math.ceil(Math.max.apply(null, vals)/5)*5;
    const y = d3.scaleLinear().domain([lo, hi]).range([H-m.b, m.t]);
    g.append("g").attr("transform",`translate(${m.l},0)`).attr("class","axis").call(d3.axisLeft(y).ticks(5));
    MODELS.forEach((mo,i) => {
      g.append("rect").attr("x",x(mo.id)).attr("y",y(vals[i])).attr("width",x.bandwidth()).attr("height",y(lo)-y(vals[i]))
        .attr("fill", mo.kind==="vit" ? P.A : P.B).attr("opacity", i===pick ? 1 : 0.65).attr("stroke", i===pick ? P.ink : "none")
        .style("cursor","pointer").on("click", () => { pick = i; draw(); });
      txt(g, x(mo.id)+x.bandwidth()/2, y(vals[i])-4, vals[i].toFixed(1), {anchor:"middle", size:9.5});
      txt(g, x(mo.id)+x.bandwidth()/2, H-m.b+14, mo.id.replace("ViT-",""), {anchor:"middle", size:9.5, fill:P.muted});
    });
    txt(g, m.l+6, m.t-8, "orange = ResNet, blue = ViT · y axis does not start at zero", {size:9.5, fill:P.muted});
    const mo = MODELS[pick];
    const spec = mo.kind === "vit"
      ? `${mo.layers} layers × ${mo.width}, ${mo.heads} heads, patch ${mo.patch}, ${mo.res}px → ${tokens(mo)} tokens; image tower ${fmtN(vitParams(mo))} + text ${fmtN(textParams(mo))} = <b>${fmtN(vitParams(mo)+textParams(mo))}</b> parameters (computed)`
      : `blocks (${mo.blocks.join(", ")}), width ${mo.width}, ${mo.res}px; image forward <b>${mo.gflops} GFLOPs</b> (paper Fig. 9); text tower ${fmtN(textParams(mo))} (computed)`;
    const best = MODELS[vals.indexOf(Math.max.apply(null, vals))].id;
    d3.select("#zoo-read").html(`<b>${mo.id}</b>: ${spec} · zero-shot ImageNet ${ZS[mo.id][iIN].toFixed(1)}, zero-shot 27-mean ${mean(ZS[mo.id]).toFixed(1)}, linear-probe 27-mean ${mean(LP[mo.id]).toFixed(1)} · highest on this metric: <b>${best}</b>.`);
  }
  d3.select("#zoo-metric").on("change", draw);
  draw();
});

/* ───────────────────────── 17 · zero-shot classifier builder ───────────────────────── */
CV.safe("zs", function(){
  const {P, softmax, txt} = CV;
  const svg = d3.select("#zs-svg");
  /* toy class directions (degrees) on a 2-D unit circle — illustrative, not CLIP outputs */
  const CL = [{n:"cat",a:30},{n:"dog",a:62},{n:"tiger",a:8},{n:"car",a:175},{n:"truck",a:205},{n:"airplane",a:278},{n:"bird",a:318}];
  const on = CL.map((c,i) => i !== 2 && i !== 6);
  const box = d3.select("#zs-classes");
  CL.forEach((c,i) => {
    const lab = box.append("label");
    lab.append("input").attr("type","checkbox").attr("id","zs-c"+i).property("checked", on[i]).on("change", function(){ on[i] = this.checked; draw(); });
    lab.append("span").text(" " + c.n);
  });
  const cx = 150, cy = 150, R = 110;
  const rad = a => a*Math.PI/180;
  function draw(){
    svg.selectAll("*").remove();
    const ang = +d3.select("#zs-ang").property("value");
    const s = +d3.select("#zs-scale").property("value");
    d3.select("#zs-angv").text(ang + "°");
    const g = svg.append("g");
    g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",R).attr("fill","none").attr("stroke",P.line);
    const act = CL.map((c,i) => ({c, i})).filter(o => on[o.i]);
    const cos = act.map(o => Math.cos(rad(ang - o.c.a)));
    const p = act.length ? softmax(cos.map(v => s*v)) : [];
    const best = p.length ? p.indexOf(Math.max.apply(null, p)) : -1;
    CL.forEach((c,i) => {
      const x2 = cx + R*Math.cos(rad(c.a)), y2 = cy - R*Math.sin(rad(c.a));
      const isOn = on[i];
      g.append("line").attr("x1",cx).attr("y1",cy).attr("x2",x2).attr("y2",y2).attr("stroke", isOn ? P.B : P.line).attr("stroke-width", isOn?1.6:1).attr("stroke-dasharray", isOn ? null : "3 3");
      g.append("circle").attr("cx",x2).attr("cy",y2).attr("r",4.5).attr("fill", isOn ? P.B : P.panel).attr("stroke",P.B);
      txt(g, cx + (R+14)*Math.cos(rad(c.a)), cy - (R+14)*Math.sin(rad(c.a)) + 4, c.n, {anchor:"middle", size:10, fill: isOn ? P.ink : P.muted});
    });
    const xi = cx + R*Math.cos(rad(ang)), yi = cy - R*Math.sin(rad(ang));
    g.append("line").attr("x1",cx).attr("y1",cy).attr("x2",xi).attr("y2",yi).attr("stroke",P.A).attr("stroke-width",2.4);
    g.append("rect").attr("x",xi-6).attr("y",yi-6).attr("width",12).attr("height",12).attr("fill",P.A);
    txt(g, cx, 22, "image (blue square) vs class prompts (orange)", {anchor:"middle", size:10, fill:P.muted});
    // probability bars
    const bx = 330, bw = 290, top = 40, rh = 28;
    act.forEach((o,j) => {
      const yy = top + j*rh;
      txt(g, bx, yy+14, `"a photo of a ${o.c.n}."`, {size:10});
      g.append("rect").attr("x",bx+150).attr("y",yy+3).attr("width",Math.max(1, (bw-150)*p[j])).attr("height",rh-10).attr("fill", j===best ? P.good : P.A).attr("opacity",0.85);
      txt(g, bx+150+Math.max(1,(bw-150)*p[j])+4, yy+15, `${p[j].toFixed(3)}  (cos ${cos[j].toFixed(2)})`, {size:9.5, fill:P.muted});
    });
    if (!act.length) txt(g, bx, top+14, "tick at least one class", {fill:P.bad});
    // what would the full label set say?
    const allCos = CL.map(c => Math.cos(rad(ang - c.a)));
    const trueBest = allCos.indexOf(Math.max.apply(null, allCos));
    let note = "";
    if (act.length && !on[trueBest]) note = ` · the nearest prompt overall, <b>${CL[trueBest].n}</b>, is not in the label set, so the softmax confidently picks something else — there is no "none of the above".`;
    d3.select("#zs-read").html(act.length
      ? `Prediction: <b>${act[best].c.n}</b> with P = <b>${p[best].toFixed(3)}</b> at scale ${s} · logits = ${s} × cosine; the argmax would be the same at any scale${note}`
      : `No classes selected — a zero-shot classifier needs at least one prompt.`);
  }
  d3.select("#zs-ang").on("input", draw);
  d3.select("#zs-scale").on("change", draw);
  draw();
});

/* ───────────────────────── 18 · prompt ensembling ───────────────────────── */
CV.safe("ens", function(){
  const {P, txt, norm, dot, lcg, gauss} = CV;
  const svg = d3.select("#ens-svg"), W = 640, H = 300;
  const D = 16, K = 6, M = 80, NI = 60;   // toy: 16-D space, 6 classes, 80 templates, 60 test images per class
  let seed = 7;
  let cache = null;
  function build(noise){
    const r = lcg(seed);
    const cls = d3.range(K).map(() => norm(d3.range(D).map(() => gauss(r))));
    const tmplShift = d3.range(M).map(() => d3.range(D).map(() => gauss(r)));   // style shared by every class
    const emb = cls.map(c => d3.range(M).map(mi => norm(c.map((v,d) => v + noise*(0.6*tmplShift[mi][d] + 0.8*gauss(r))/Math.sqrt(D)*3))));
    const imgs = [];
    cls.forEach((c,k) => { for (let i=0;i<NI;i++) imgs.push({k, u: norm(c.map(v => v + 0.9*gauss(r)/Math.sqrt(D)*3))}); });
    // accuracy and prototype quality for every K = 1..M
    const accs = [], qual = [];
    for (let kk = 1; kk <= M; kk++){
      const protos = emb.map(list => { const s = new Array(D).fill(0); for (let mi=0; mi<kk; mi++) for (let d=0; d<D; d++) s[d] += list[mi][d]; return norm(s); });
      let ok = 0;
      imgs.forEach(im => { let b=-1, bi=-1; protos.forEach((p,k) => { const v = dot(p, im.u); if (v > b){ b=v; bi=k; } }); if (bi === im.k) ok++; });
      accs.push(ok / imgs.length);
      qual.push(protos.reduce((a,p,k) => a + dot(p, cls[k]), 0) / K);
    }
    return {accs, qual};
  }
  function draw(){
    svg.selectAll("*").remove();
    const kk = +d3.select("#ens-k").property("value");
    const noise = +d3.select("#ens-noise").property("value") / 100;
    d3.select("#ens-kv").text(kk); d3.select("#ens-noisev").text(noise.toFixed(2));
    const key = seed + ":" + noise;
    if (!cache || cache.key !== key) cache = Object.assign({key}, build(noise));
    const {accs, qual} = cache;
    const g = svg.append("g"), m = {l:54, r:20, t:26, b:44};
    const x = d3.scaleLinear().domain([1, 80]).range([m.l, W-m.r]);
    const y = d3.scaleLinear().domain([0, 1]).range([H-m.b, m.t]);
    g.append("g").attr("transform",`translate(0,${H-m.b})`).attr("class","axis").call(d3.axisBottom(x).tickValues([1,10,20,40,60,80]));
    g.append("g").attr("transform",`translate(${m.l},0)`).attr("class","axis").call(d3.axisLeft(y).ticks(5));
    g.append("path").datum(accs).attr("fill","none").attr("stroke",P.good).attr("stroke-width",2).attr("d", d3.line().x((d,i)=>x(i+1)).y(d=>y(d)));
    g.append("path").datum(qual).attr("fill","none").attr("stroke",P.A).attr("stroke-width",2).attr("stroke-dasharray","5 3").attr("d", d3.line().x((d,i)=>x(i+1)).y(d=>y(d)));
    g.append("line").attr("x1",x(kk)).attr("x2",x(kk)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.muted).attr("stroke-dasharray","3 3");
    g.append("circle").attr("cx",x(kk)).attr("cy",y(accs[kk-1])).attr("r",5).attr("fill",P.good);
    g.append("circle").attr("cx",x(kk)).attr("cy",y(qual[kk-1])).attr("r",5).attr("fill",P.A);
    txt(g, m.l+8, m.t+10, "toy accuracy (6 classes, 360 test points)", {size:10, fill:P.good});
    txt(g, m.l+8, m.t+24, "mean cosine of prototype to the true class direction", {size:10, fill:P.A});
    txt(g, (m.l+W-m.r)/2, H-8, "templates averaged per class, K", {anchor:"middle", size:10, fill:P.muted});
    d3.select("#ens-read").html(`K = <b>${kk}</b>: toy accuracy <b>${(100*accs[kk-1]).toFixed(1)}%</b> (K = 1: ${(100*accs[0]).toFixed(1)}%, K = 80: ${(100*accs[79]).toFixed(1)}%) · prototype–truth cosine <b>${qual[kk-1].toFixed(3)}</b> · the part of the noise every template shares (its style shift) does not average away, so the curve flattens.`);
  }
  d3.select("#ens-k").on("input", draw);
  d3.select("#ens-noise").on("input", draw);
  d3.select("#ens-seed").on("click", () => { seed = (seed * 31 + 11) % 100003; cache = null; draw(); });
  draw();
});

/* ───────────────────────── 19 · per-dataset comparison ───────────────────────── */
CV.safe("bench", function(){
  const {P, txt, DS, ZS, LP, FIG5} = CV;
  const svg = d3.select("#bench-svg"), W = 640, H = 440, m = {l:118, r:40, t:14, b:30};
  function draw(){
    svg.selectAll("*").remove();
    const mode = d3.select("#bench-mode").property("value");
    const rows = DS.map((d,i) => ({d, v: mode === "rn50" ? FIG5[d] : ZS["ViT-L/14@336"][i] - LP["ViT-L/14@336"][i]}));
    rows.sort((a,b) => b.v - a.v);
    const g = svg.append("g");
    const ext = Math.max(40, Math.ceil(d3.max(rows, r => Math.abs(r.v))/10)*10);
    const x = d3.scaleLinear().domain([-ext, mode==="rn50" ? ext : Math.max(5, Math.ceil(d3.max(rows, r => r.v)))]).range([m.l, W-m.r]);
    const y = d3.scaleBand().domain(rows.map(r=>r.d)).range([m.t, H-m.b]).padding(0.15);
    g.append("g").attr("transform",`translate(0,${H-m.b})`).attr("class","axis").call(d3.axisBottom(x).ticks(8));
    g.append("line").attr("x1",x(0)).attr("x2",x(0)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.muted);
    rows.forEach(r => {
      const x0 = x(Math.min(0, r.v)), w = Math.abs(x(r.v) - x(0));
      g.append("rect").attr("x",x0).attr("y",y(r.d)).attr("width",Math.max(1,w)).attr("height",y.bandwidth()).attr("fill", r.v >= 0 ? P.good : P.bad).attr("opacity",0.85);
      txt(g, m.l-6, y(r.d)+y.bandwidth()-2, r.d, {anchor:"end", size:9.5, fill:P.muted});
      txt(g, r.v >= 0 ? x(r.v)+3 : x(r.v)-3, y(r.d)+y.bandwidth()-2, (r.v>0?"+":"") + r.v.toFixed(1), {anchor: r.v >= 0 ? "start" : "end", size:9});
    });
    const wins = rows.filter(r => r.v > 0).length;
    const close = rows.filter(r => r.v >= -3).length;
    const avg = rows.reduce((a,r)=>a+r.v, 0) / rows.length;
    d3.select("#bench-read").html(mode === "rn50"
      ? `Zero-shot CLIP beats the supervised ResNet-50 linear probe on <b>${wins} of ${rows.length}</b> datasets (mean difference ${avg>=0?"+":""}${avg.toFixed(1)} points) · best: ${rows[0].d} (+${rows[0].v.toFixed(1)}), worst: ${rows[rows.length-1].d} (${rows[rows.length-1].v.toFixed(1)}).`
      : `Zero-shot is within 3 points of CLIP's own linear probe on <b>${close}</b> of ${rows.length} datasets; mean gap <b>${avg.toFixed(1)}</b> points · largest gap: ${rows[rows.length-1].d} (${rows[rows.length-1].v.toFixed(1)}). (Computed from Tables 10 and 11; Table 11's Rendered SST2 and Hateful Memes entries use their own metrics.)`);
  }
  d3.select("#bench-mode").on("change", draw);
  draw();
});

/* ───────────────────────── 21 · robustness bars ───────────────────────── */
CV.safe("rob", function(){
  const {P, txt, mean} = CV;
  const svg = d3.select("#rob-svg"), W = 640, H = 300, m = {l:44, r:16, t:30, b:40};
  /* Source: CLIP paper Figure 13 (ResNet-101 and zero-shot CLIP ViT-L/14@336px); Table 16 (linear-probe CLIP). */
  const SETS = ["ImageNet","ImageNetV2","ImageNet-R","ObjectNet","ImageNet-Sketch","ImageNet-A"];
  const RN101 = [76.2, 64.3, 37.7, 32.6, 25.2, 2.7];
  const ZSC   = [76.2, 70.1, 88.9, 72.3, 60.2, 77.1];
  const LPC   = [85.4, 75.9, 84.2, 66.2, 57.4, 75.3];
  function draw(){
    svg.selectAll("*").remove();
    const showLP = d3.select("#rob-lp").property("checked");
    const sort = d3.select("#rob-sort").property("value");
    let idx = SETS.map((_,i)=>i);
    if (sort === "gap") idx = [0].concat(idx.slice(1).sort((a,b) => (ZSC[b]-RN101[b]) - (ZSC[a]-RN101[a])));
    const series = [{n:"ResNet-101", v:RN101, c:P.muted}, {n:"zero-shot CLIP", v:ZSC, c:P.good}].concat(showLP ? [{n:"CLIP + ImageNet probe", v:LPC, c:P.B}] : []);
    const g = svg.append("g");
    const x0 = d3.scaleBand().domain(idx.map(i=>SETS[i])).range([m.l, W-m.r]).padding(0.18);
    const x1 = d3.scaleBand().domain(series.map(s=>s.n)).range([0, x0.bandwidth()]).padding(0.08);
    const y = d3.scaleLinear().domain([0, 100]).range([H-m.b, m.t]);
    g.append("g").attr("transform",`translate(${m.l},0)`).attr("class","axis").call(d3.axisLeft(y).ticks(5));
    idx.forEach(i => {
      series.forEach(s => {
        const xx = x0(SETS[i]) + x1(s.n);
        g.append("rect").attr("x",xx).attr("y",y(s.v[i])).attr("width",x1.bandwidth()).attr("height",y(0)-y(s.v[i])).attr("fill",s.c).attr("opacity",0.9);
        txt(g, xx + x1.bandwidth()/2, y(s.v[i]) - 3, s.v[i].toFixed(1), {anchor:"middle", size:8.5, fill:P.muted});
      });
      txt(g, x0(SETS[i]) + x0.bandwidth()/2, H-m.b+14, SETS[i], {anchor:"middle", size:9.5, fill:P.muted});
    });
    series.forEach((s,k) => { g.append("rect").attr("x",m.l+8+k*170).attr("y",8).attr("width",10).attr("height",10).attr("fill",s.c); txt(g, m.l+22+k*170, 17, s.n, {size:10}); });
    const sh = v => mean(v.slice(1));
    let msg = `Mean over the five shifted sets: ResNet-101 <b>${sh(RN101).toFixed(1)}</b> (drop from ImageNet ${(RN101[0]-sh(RN101)).toFixed(1)}), zero-shot CLIP <b>${sh(ZSC).toFixed(1)}</b> (drop ${(ZSC[0]-sh(ZSC)).toFixed(1)})`;
    if (showLP) msg += `, ImageNet-probed CLIP <b>${sh(LPC).toFixed(1)}</b> (drop ${(LPC[0]-sh(LPC)).toFixed(1)}): +${(LPC[0]-ZSC[0]).toFixed(1)} on ImageNet, ${(sh(LPC)-sh(ZSC)).toFixed(1)} on the shifts`;
    d3.select("#rob-read").html(msg + ". Note: Table 16 lists zero-shot ImageNet-A as 77.2; Figure 13 shows 77.1.");
  }
  d3.select("#rob-lp").on("change", draw);
  d3.select("#rob-sort").on("change", draw);
  draw();
});

/* ───────────────────────── 23 · contamination calculator ───────────────────────── */
CV.safe("ovl", function(){
  const {P, txt} = CV;
  const svg = d3.select("#ovl-svg"), W = 640, H = 260;
  /* Source for presets: CLIP paper §5 — Birdsnap 12.1% overlap, +0.6 overall; Country211 21.5% overlap, +0.2 overall.
     Δ for each preset is implied by All − Clean = f·Δ. The clean accuracy used for drawing is illustrative. */
  const PRE = {bird:{f:12.1, gain:0.6, n:"Birdsnap"}, c211:{f:21.5, gain:0.2, n:"Country211"}};
  const CLEAN = 60;
  function draw(){
    svg.selectAll("*").remove();
    const f = +d3.select("#ovl-f").property("value") / 10;
    const dl = +d3.select("#ovl-d").property("value") / 10;
    d3.select("#ovl-fv").text(f.toFixed(1) + "%"); d3.select("#ovl-dv").text((dl>0?"+":"") + dl.toFixed(1));
    const infl = (f/100) * dl, accO = CLEAN + dl, accA = CLEAN + infl;
    const g = svg.append("g");
    // test set bar
    const bx = 40, bw = 560, by = 40;
    txt(g, bx, by-10, "the test set, split by the duplicate detector", {size:10, fill:P.muted});
    g.append("rect").attr("x",bx).attr("y",by).attr("width",bw*(1-f/100)).attr("height",30).attr("fill",P.A).attr("opacity",0.7);
    g.append("rect").attr("x",bx+bw*(1-f/100)).attr("y",by).attr("width",Math.max(0.5,bw*f/100)).attr("height",30).attr("fill",P.B);
    txt(g, bx+6, by+20, `Clean ${(100-f).toFixed(1)}%`, {size:10.5, fill:"#111", bold:true});
    if (f >= 4) txt(g, bx+bw-6, by+20, `Overlap ${f.toFixed(1)}%`, {anchor:"end", size:10.5, fill:"#111", bold:true});
    // accuracies
    const y = d3.scaleLinear().domain([30, 90]).range([230, 100]);
    const bars = [{n:"Clean", v:CLEAN, c:P.A}, {n:"Overlap", v:accO, c:P.B}, {n:"All", v:accA, c:P.good}];
    bars.forEach((b,i) => {
      const x = 120 + i*150, v = Math.max(30, Math.min(90, b.v));
      g.append("rect").attr("x",x).attr("y",y(v)).attr("width",80).attr("height",230-y(v)).attr("fill",b.c).attr("opacity",0.85);
      txt(g, x+40, y(v)-5, b.v.toFixed(2), {anchor:"middle", size:10.5});
      txt(g, x+40, 246, b.n, {anchor:"middle", size:10, fill:P.muted});
    });
    txt(g, 600, 120, "clean accuracy fixed", {anchor:"end", size:9.5, fill:P.muted});
    txt(g, 600, 133, "at an illustrative 60%", {anchor:"end", size:9.5, fill:P.muted});
    const verdict = Math.abs(infl) <= 0.1 ? "within the 0.1-point band most of the paper's 35 datasets fell in" : "above the 0.1-point band — the paper found only 7 such datasets";
    d3.select("#ovl-read").html(`All − Clean = f · Δ = ${(f/100).toFixed(3)} × ${dl.toFixed(1)} = <b>${infl>=0?"+":""}${infl.toFixed(2)}</b> points of inflation — ${verdict}.`);
  }
  d3.select("#ovl-preset").on("change", function(){
    const p = PRE[this.value];
    if (p){ d3.select("#ovl-f").property("value", Math.round(p.f*10)); d3.select("#ovl-d").property("value", Math.round(10*p.gain/(p.f/100))); }
    draw();
  });
  d3.select("#ovl-f").on("input", draw);
  d3.select("#ovl-d").on("input", draw);
  draw();
});

/* ───────────────────────── 26 · modality gap toy ───────────────────────── */
CV.safe("gap", function(){
  const {P, txt, norm, dot, clipLoss} = CV;
  const svg = d3.select("#gap-svg"), W = 640, H = 300;
  /* toy: six semantic directions in a plane, captions rotated by a small per-pair offset, gap along a third axis */
  const K = 6, base = d3.range(K).map(k => k*28 + 20), off = [6, -5, 4, -7, 5, -3];   // a 140° cone of meanings
  const rad = a => a*Math.PI/180;
  function build(beta, center){
    const cb = Math.cos(rad(beta)), sb = Math.sin(rad(beta));
    let U = base.map(a => [cb*Math.cos(rad(a)), cb*Math.sin(rad(a)), sb]);
    let V = base.map((a,k) => [cb*Math.cos(rad(a+off[k])), cb*Math.sin(rad(a+off[k])), -sb]);
    const mu = M => [0,1,2].map(d => M.reduce((s,v)=>s+v[d],0)/M.length);
    const gapVec = (mU, mV) => Math.sqrt([0,1,2].reduce((s,d)=>s+Math.pow(mU[d]-mV[d],2),0));
    const g0 = gapVec(mu(U), mu(V));
    if (center){ const mU = mu(U), mV = mu(V); U = U.map(v => norm(v.map((x,d)=>x-mU[d]))); V = V.map(v => norm(v.map((x,d)=>x-mV[d]))); }
    return {U, V, gap: gapVec(mu(U), mu(V)), gap0: g0};
  }
  function draw(){
    svg.selectAll("*").remove();
    const beta = +d3.select("#gap-b").property("value");
    const s = +d3.select("#gap-s").property("value");
    const center = d3.select("#gap-center").property("checked");
    d3.select("#gap-bv").text(beta + "°");
    const {U, V, gap} = build(beta, center);
    const cosM = U.map(u => V.map(v => dot(u,v)));
    const r = clipLoss(cosM.map(row => row.map(c => s*c)));
    let diag = 0, offd = 0, ok = 0;
    cosM.forEach((row,i) => row.forEach((c,j) => { if (i===j) diag += c; else offd += c; }));
    diag /= K; offd /= K*(K-1);
    cosM.forEach((row,i) => { if (row.indexOf(Math.max.apply(null,row)) === i) ok++; });
    // oblique projection of the 3-D points
    const g = svg.append("g"), cx = 170, cy = 150, R = 95;
    const proj = v => [cx + R*v[0], cy - R*0.45*v[1] - R*1.0*v[2]];
    g.append("ellipse").attr("cx",cx).attr("cy",cy).attr("rx",R).attr("ry",R*0.45).attr("fill","none").attr("stroke",P.line);
    g.append("line").attr("x1",cx).attr("y1",cy-R*1.05).attr("x2",cx).attr("y2",cy+R*1.05).attr("stroke",P.line).attr("stroke-dasharray","3 3");
    txt(g, cx+6, cy-R*1.05+10, "gap axis g", {size:9.5, fill:P.muted});
    U.forEach((u,k) => { const [x,y] = proj(u), [x2,y2] = proj(V[k]);
      g.append("line").attr("x1",x).attr("y1",y).attr("x2",x2).attr("y2",y2).attr("stroke",P.muted).attr("stroke-opacity",0.35);
      g.append("rect").attr("x",x-5).attr("y",y-5).attr("width",10).attr("height",10).attr("fill",P.A);
      g.append("circle").attr("cx",x2).attr("cy",y2).attr("r",5).attr("fill","none").attr("stroke",P.B).attr("stroke-width",2);
    });
    txt(g, cx, 18, "images (squares) above, texts (circles) below", {anchor:"middle", size:10, fill:P.muted});
    // stats panel: cos histogram-like strip
    const x = d3.scaleLinear().domain([-1, 1]).range([360, 620]);
    g.append("g").attr("transform","translate(0,230)").attr("class","axis").call(d3.axisBottom(x).ticks(5));
    cosM.forEach((row,i) => row.forEach((c,j) => {
      g.append("circle").attr("cx",x(c)).attr("cy", i===j ? 150 : 195 + ((i*7+j*3)%5)*5).attr("r",3.5).attr("fill", i===j ? P.good : P.bad).attr("opacity",0.8);
    }));
    txt(g, 360, 140, "matched-pair cosines", {size:10, fill:P.good});
    txt(g, 360, 186, "mismatched cosines", {size:10, fill:P.bad});
    txt(g, 490, 262, "cosine", {anchor:"middle", size:10, fill:P.muted});
    d3.select("#gap-read").html(`β = ${beta}°${center ? " (then centred)" : ""}: centre-to-centre gap ‖Δ‖ = <b>${gap.toFixed(2)}</b> · mean matched cosine <b>${diag.toFixed(3)}</b>, mean mismatched <b>${offd.toFixed(3)}</b> · symmetric loss at scale ${s}: <b>${r.loss.toFixed(3)}</b> · rows ranked correctly: <b>${ok}/${K}</b> (the ranking ignores the gap in this toy because the offset is orthogonal to the semantic plane; in real CLIP it is not exactly orthogonal).`);
  }
  d3.select("#gap-b").on("input", draw);
  d3.select("#gap-s").on("change", draw);
  d3.select("#gap-center").on("change", draw);
  draw();
});

/* ───────────────────────── 32 · OpenCLIP scaling laws ───────────────────────── */
CV.safe("scale", function(){
  const {P, txt, sci} = CV;
  const svg = d3.select("#scale-svg"), W = 640, H = 300, m = {l:56, r:20, t:24, b:44};
  /* Source: "Reproducible scaling laws for contrastive language-image learning" (arXiv 2212.07143), Figure 1a
     fits E = β·C^α (error as a fraction; C = GMACs per sample × samples seen), and §4 (ViT-H/14, 78.0%). */
  const FITS = {
    in:  {open:{b:5.86,  a:-0.11}, clip:{b:23.18,  a:-0.16}, label:"zero-shot ImageNet error"},
    rob: {open:{b:11.61, a:-0.13}, clip:{b:211.66, a:-0.24}, label:"ImageNet robustness-set error"}
  };
  const H14 = {C: 190.97e9 * 34e9 / 1e9, acc: 78.0};  // 190.97 GMACs/sample (Table 25) × 34B samples; C in GMAC units
  function E(f, C){ return f.b * Math.pow(C, f.a); }
  function draw(){
    svg.selectAll("*").remove();
    const task = d3.select("#scale-task").property("value"), F = FITS[task];
    const lc = +d3.select("#scale-c").property("value") / 10, C = Math.pow(10, lc);
    d3.select("#scale-cv").text("10^" + lc.toFixed(1));
    const g = svg.append("g");
    const x = d3.scaleLog().domain([1e10, 1e14]).range([m.l, W-m.r]);
    const y = d3.scaleLog().domain([0.1, 0.8]).range([H-m.b, m.t]);
    g.append("g").attr("transform",`translate(0,${H-m.b})`).attr("class","axis").call(d3.axisBottom(x).tickValues([1e10,1e11,1e12,1e13,1e14]).tickFormat(d=>"10^"+Math.round(Math.log10(d))));
    g.append("g").attr("transform",`translate(${m.l},0)`).attr("class","axis").call(d3.axisLeft(y).tickValues([0.1,0.2,0.3,0.4,0.6,0.8]).tickFormat(d3.format(".0%")));
    const cs = d3.range(0, 101).map(i => Math.pow(10, 10 + 4*i/100));
    [["open",P.A,"OpenCLIP / LAION"],["clip",P.B,"OpenAI CLIP / WIT"]].forEach(([k,col,nm],i) => {
      g.append("path").datum(cs).attr("fill","none").attr("stroke",col).attr("stroke-width",2).attr("d", d3.line().x(c=>x(c)).y(c=>y(Math.max(0.1,Math.min(0.8,E(F[k],c))))));
      txt(g, W-m.r-4, m.t+12+14*i, `${nm}: E = ${F[k].b} · C^${F[k].a}`, {anchor:"end", size:10, fill:col});
    });
    if (task === "in"){ g.append("circle").attr("cx",x(H14.C)).attr("cy",y(1-H14.acc/100)).attr("r",5).attr("fill",P.good); txt(g, x(H14.C)-6, y(1-H14.acc/100)+16, "ViT-H/14 measured", {anchor:"end", size:9.5, fill:P.good}); }
    g.append("line").attr("x1",x(C)).attr("x2",x(C)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.muted).attr("stroke-dasharray","4 3");
    txt(g, (m.l+W-m.r)/2, H-8, "total compute C (GMACs per sample × samples seen, log)", {anchor:"middle", size:10, fill:P.muted});
    const eo = E(F.open, C), ec = E(F.clip, C);
    const cross = Math.pow(F.clip.b / F.open.b, 1/(F.open.a - F.clip.a));
    let extra = "";
    if (task === "in") extra = ` · check: ViT-H/14 at C = ${sci(H14.C,2)} → fit ${(100*(1-E(F.open,H14.C))).toFixed(1)}% vs measured ${H14.acc}%`;
    d3.select("#scale-read").html(`At C = ${sci(C,1)}: ${F.label} — OpenCLIP fit <b>${(100*eo).toFixed(1)}%</b> (accuracy ${(100-100*eo).toFixed(1)}%), CLIP fit <b>${(100*ec).toFixed(1)}%</b> · the two fits cross at C ≈ ${sci(cross,1)}; beyond it the steeper WIT curve predicts lower error${extra}. Fits are to each family's Pareto frontier and are only reliable inside the measured range.`);
  }
  d3.select("#scale-task").on("change", draw);
  d3.select("#scale-c").on("input", draw);
  draw();
});

/* ───────────────────────── 33 · softmax vs sigmoid on the same cosines ───────────────────────── */
CV.safe("sig", function(){
  const {P, txt, clipLoss, logSigmoid, comma} = CV;
  const svg = d3.select("#sig-svg"), W = 640, H = 300;
  const S = [[0.30,0.10,0.05],[0.12,0.25,0.20],[0.02,0.15,0.28]];   // the worked example's cosines (§05)
  function draw(){
    svg.selectAll("*").remove();
    const t = +d3.select("#sig-t").property("value");
    const b = +d3.select("#sig-b").property("value");
    const k = +d3.select("#sig-n").property("value"), N = Math.pow(2, k);
    d3.select("#sig-tv").text(t); d3.select("#sig-bv").text(b.toFixed(1)); d3.select("#sig-nv").text(comma(N));
    const sm = clipLoss(S.map(r => r.map(c => t*c)));
    const cell = S.map((r,i) => r.map((c,j) => -logSigmoid((i===j?1:-1) * (t*c + b))));
    const sig = cell.reduce((a,r)=>a+r.reduce((x,y)=>x+y,0),0) / 3;
    const g = svg.append("g");
    function heat(ox, M, title, fmt, maxv){
      const cs = 50, col = d3.scaleSequential(d3.interpolateMagma).domain([0, maxv]);
      txt(g, ox + 1.5*cs, 26, title, {anchor:"middle", size:10.5, fill:P.muted});
      M.forEach((r,i) => r.forEach((v,j) => {
        g.append("rect").attr("x",ox+j*cs).attr("y",40+i*cs).attr("width",cs-3).attr("height",cs-3).attr("rx",3).attr("fill",col(Math.min(v,maxv))).attr("stroke", i===j ? P.good : "none").attr("stroke-width",2);
        txt(g, ox+j*cs+cs/2-1, 40+i*cs+cs/2+4, fmt(v), {anchor:"middle", size:10, fill: v/maxv > 0.6 ? "#111" : "#eee"});
      }));
    }
    heat(40, sm.rowP, "softmax: row probabilities (coupled)", v=>v.toFixed(2), 1);
    const mx = Math.max(1, Math.max.apply(null, [].concat.apply([], cell)));
    heat(250, cell, "sigmoid: per-cell loss (independent)", v=>v<0.01?v.toExponential(0):v.toFixed(2), mx);
    // init comparison at batch N, cos ≈ 0 for every pair
    const smInit = Math.log(N);
    const sgInit = -logSigmoid(b) + (N-1) * (-logSigmoid(-b));
    txt(g, 460, 60, `random init, N = ${comma(N)}`, {size:10.5, fill:P.muted});
    txt(g, 460, 84, `softmax row loss: ln N = ${smInit.toFixed(2)}`, {size:10.5, fill:P.A});
    txt(g, 460, 104, `sigmoid row loss: ${sgInit < 1e4 ? sgInit.toFixed(2) : sgInit.toExponential(2)}`, {size:10.5, fill:P.B});
    txt(g, 460, 124, `  positive term ${(-logSigmoid(b)).toFixed(2)}`, {size:10, fill:P.muted});
    txt(g, 460, 140, `  ${comma(N-1)} negatives × ${(-logSigmoid(-b)).toExponential(1)}`, {size:10, fill:P.muted});
    txt(g, 40, 215, "Cosines from the worked example: diagonal 0.30, 0.25, 0.28.", {size:10, fill:P.muted});
    txt(g, 40, 232, "Softmax changes if any cell in a row changes; each sigmoid cell depends only on its own pair.", {size:10, fill:P.muted});
    d3.select("#sig-read").html(`Scale ${t}, bias ${b.toFixed(1)}: symmetric softmax loss <b>${sm.loss.toFixed(3)}</b> · sigmoid loss (sum of 9 cells ÷ N = 3) <b>${sig.toFixed(3)}</b> · at initialisation with N = ${comma(N)}, a sigmoid row starts at ${sgInit < 1e4 ? sgInit.toFixed(1) : sgInit.toExponential(2)} nats against softmax's ${smInit.toFixed(1)} — the bias decides whether the ${comma(N-1)} negatives or the one positive dominate.`);
  }
  d3.select("#sig-t").on("input", draw);
  d3.select("#sig-b").on("input", draw);
  d3.select("#sig-n").on("input", draw);
  draw();
});
