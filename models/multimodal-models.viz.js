/* multimodal-models.viz.js — every interactive figure on models/multimodal-models.html.
   Loaded after data.js / notes.js (palette C comes from notes.js).
   Each figure is its own IIFE wrapped in try/catch so one failure cannot blank the rest.
   House rule: every number a figure displays is computed here from a spec, never typed into a label. */

const MMV = (function(){
  const P = Object.assign({}, (typeof C !== "undefined" ? C : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"}),
    { panel:"#15181f", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6" });
  function fmtN(x){
    const a = Math.abs(x);
    if (a >= 1e9)  return (x/1e9).toFixed(a>=1e10?1:2)+"B";
    if (a >= 1e6)  return (x/1e6).toFixed(a>=1e8?0:(a>=1e7?1:2))+"M";
    if (a >= 1e3)  return (x/1e3).toFixed(1)+"k";
    return String(Math.round(x));
  }
  const comma = x => Math.round(x).toLocaleString("en-US");
  function txt(g, x, y, s, o){
    o = o || {};
    return g.append("text").attr("x",x).attr("y",y).attr("font-size",o.size||11).attr("fill",o.fill||P.ink)
      .attr("text-anchor",o.anchor||"start").attr("font-family",o.mono?"SF Mono,Menlo,monospace":null)
      .attr("font-weight",o.bold?600:null).text(s);
  }
  function safe(name, fn){ try { fn(); } catch(e){ if (typeof console!=="undefined") console.error("[multimodal.viz] "+name+" failed:", e); } }
  function lcg(seed){ let s = seed >>> 0; return () => { s = (1664525*s + 1013904223) >>> 0; return s/4294967296; }; }
  function fillSelect(sel, items){
    const el = document.getElementById(sel); if (!el) return null;
    el.innerHTML = items.map(d => `<option value="${d.id}">${d.name}</option>`).join("");
    return el;
  }
  return {P, fmtN, comma, txt, safe, lcg, fillSelect};
})();

/* ───────────────────────── 03 · visual-token budget ───────────────────────── */
MMV.safe("tokens", function(){
  const {P, txt, comma, fillSelect} = MMV;
  /* published configurations: resolution r, patch p, connector kind, LM context */
  const PRE = [
    {id:"llava15", name:"LLaVA-1.5 · CLIP ViT-L/14-336 → Vicuna v1.5", kind:"proj", r:336, p:14, ctx:4096, lm:"Vicuna v1.5 (Llama 2 base)"},
    {id:"llava",   name:"LLaVA · CLIP ViT-L/14 → Vicuna-13B",          kind:"proj", r:224, p:14, ctx:2048, lm:"Vicuna-13B (LLaMA base)"},
    {id:"pg224",   name:"PaliGemma 224 · SigLIP So400m/14 → Gemma 2B", kind:"proj", r:224, p:14, ctx:8192, lm:"Gemma 2B"},
    {id:"pg448",   name:"PaliGemma 448",                                kind:"proj", r:448, p:14, ctx:8192, lm:"Gemma 2B"},
    {id:"pg896",   name:"PaliGemma 896",                                kind:"proj", r:896, p:14, ctx:8192, lm:"Gemma 2B"},
    {id:"blip2",   name:"BLIP-2 · Q-Former (32 queries) → OPT-2.7B",   kind:"query", q:32, r:224, p:14, ctx:2048, lm:"OPT-2.7B"},
    {id:"clip",    name:"CLIP ViT-L/14-336 · pooled embedding",        kind:"pool", r:336, p:14, ctx:77, lm:"CLIP text tower"}
  ];
  const sel = fillSelect("tk-preset", PRE);
  const rIn = document.getElementById("tk-r"), pIn = document.getElementById("tk-p");
  const svg = d3.select("#tk-svg"), W = 640, H = 250;
  let cur = PRE[0];
  function load(id){ cur = PRE.find(d=>d.id===id) || PRE[0]; rIn.value = cur.r; pIn.value = cur.p; draw(); }
  function draw(){
    const r = +rIn.value, p = +pIn.value, side = Math.floor(r/p), grid = side*side;
    d3.select("#tk-rv").text(r); d3.select("#tk-pv").text(p);
    const vis = cur.kind === "proj" ? grid : (cur.kind === "query" ? cur.q : 1);
    svg.selectAll("*").remove();
    const g = svg.append("g");
    /* left: patch grid */
    const box = 190, x0 = 20, y0 = 30, cell = box / side;
    txt(g, x0, 18, `encoder grid ${side} × ${side} = ${comma(grid)} patches`, {size:11, fill:P.muted});
    g.append("rect").attr("x",x0).attr("y",y0).attr("width",box).attr("height",box).attr("fill",P.A).attr("fill-opacity",.12).attr("stroke",P.A);
    const step = side > 48 ? Math.ceil(side/48) : 1;
    for (let i=step; i<side; i+=step){
      g.append("line").attr("x1",x0+i*cell).attr("x2",x0+i*cell).attr("y1",y0).attr("y2",y0+box).attr("stroke",P.A).attr("stroke-opacity",.35).attr("stroke-width",.6);
      g.append("line").attr("y1",y0+i*cell).attr("y2",y0+i*cell).attr("x1",x0).attr("x2",x0+box).attr("stroke",P.A).attr("stroke-opacity",.35).attr("stroke-width",.6);
    }
    if (step > 1) txt(g, x0, y0+box+14, `grid lines drawn every ${step} patches`, {size:9.5, fill:P.muted});
    /* right: bars */
    const bx = 260, bw = W - bx - 30, maxV = Math.max(cur.ctx, vis, grid);
    const xs = d3.scaleLinear().domain([0, maxV]).range([0, bw]);
    const rows = [
      {lab:"patches from the encoder", v:grid, col:P.muted},
      {lab:"visual tokens given to the LM", v:vis, col: vis > cur.ctx ? P.bad : P.B},
      {lab:`${cur.lm} context`, v:cur.ctx, col:P.teal}
    ];
    rows.forEach((d,i)=>{
      const y = 44 + i*62;
      txt(g, bx, y-6, d.lab, {size:11, fill:P.ink});
      g.append("rect").attr("x",bx).attr("y",y).attr("width",Math.max(2,xs(d.v))).attr("height",22).attr("fill",d.col).attr("fill-opacity",.8).attr("rx",3);
      txt(g, bx + Math.max(2,xs(d.v)) + 6, y+15, comma(d.v), {size:11, mono:true, fill:P.ink});
    });
    const share = vis / cur.ctx, left = cur.ctx - vis;
    const rule = cur.kind === "proj" ? `floor(${r}/${p})² = ${side}² = ${comma(grid)}` :
                 (cur.kind === "query" ? `fixed ${cur.q} learned queries, whatever the resolution` : "one pooled vector per image");
    txt(g, bx, H-18, cur.kind === "proj" ? "projector: every patch becomes a token" : (cur.kind==="query" ? "query connector: token count fixed" : "dual encoder: no tokens enter a text model"), {size:10.5, fill:P.muted});
    const note = (r % p) ? ` · ${r} is not a multiple of ${p}, so ${r - side*p} px are cropped or padded` : "";
    d3.select("#tk-read").html(`<b>${cur.name}</b> at r = ${r}, p = ${p}: visual tokens = ${rule}. ` +
      (cur.kind === "pool" ? `The image is compared with text by cosine similarity; the ${cur.ctx}-token text context is untouched.` :
      `That is <b>${(100*share).toFixed(1)}%</b> of a ${comma(cur.ctx)}-token context, leaving ${left >= 0 ? comma(left)+" tokens for text" : "<b>no room</b> — the image alone overflows the context by "+comma(-left)+" tokens"}.`) + note);
  }
  if (sel) sel.addEventListener("change", () => load(sel.value));
  rIn.addEventListener("input", draw); pIn.addEventListener("input", draw);
  load(PRE[0].id);
});

/* ───────────────────────── 04 · connector parameters ───────────────────────── */
MMV.safe("connectors", function(){
  const {P, txt, fmtN, comma, fillSelect} = MMV;
  const lin = (a,b) => a*b + b;
  const CN = [
    {id:"pg",   name:"PaliGemma · linear 1152 → 2048",          kind:"linear", dv:1152, dl:2048},
    {id:"ll1",  name:"LLaVA · linear 1024 → 5120 (13B)",        kind:"linear", dv:1024, dl:5120},
    {id:"l157", name:"LLaVA-1.5 7B · MLP 1024 → 4096 → 4096",   kind:"mlp",    dv:1024, dl:4096},
    {id:"l1513",name:"LLaVA-1.5 13B · MLP 1024 → 5120 → 5120",  kind:"mlp",    dv:1024, dl:5120},
    {id:"qf",   name:"BLIP-2 · Q-Former (reported)",            kind:"reported", n:188e6, src:"BLIP-2 paper, total Q-Former parameters incl. 32 × 768 queries"},
    {id:"fl",   name:"Flamingo-80B · resampler + cross-attn (reported)", kind:"reported", n:10.2e9, src:"trainable parameters as listed for Flamingo80B in BLIP-2's comparison table"}
  ];
  CN.forEach(d => {
    if (d.kind === "linear") { d.n = lin(d.dv, d.dl); d.f = `${d.dv}·${d.dl} + ${d.dl}`; }
    if (d.kind === "mlp")    { d.n = lin(d.dv, d.dl) + lin(d.dl, d.dl); d.f = `(${d.dv}·${d.dl} + ${d.dl}) + (${d.dl}·${d.dl} + ${d.dl})`; }
  });
  const sel = fillSelect("cn-preset", CN);
  const svg = d3.select("#cn-svg"), W = 640, H = 240, m = {l:250, r:70, t:14, b:26};
  const x = d3.scaleLog().domain([1e6, 2e10]).range([m.l, W-m.r]);
  const bh = (H - m.t - m.b) / CN.length;
  let cur = CN[2].id;
  function draw(){
    svg.selectAll("*").remove();
    const g = svg.append("g");
    [1e6,1e7,1e8,1e9,1e10].forEach(v=>{
      g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, x(v), H-m.b+14, fmtN(v), {anchor:"middle", size:10, fill:P.muted});
    });
    CN.forEach((d,i)=>{
      const y = m.t + i*bh + 4, on = d.id === cur;
      const row = g.append("g").style("cursor","pointer").on("click", ()=>{ cur = d.id; if (sel) sel.value = cur; draw(); });
      txt(row, m.l-8, y+bh/2, d.name.split(" · ")[0], {anchor:"end", size:11, fill:on?P.B:P.ink, bold:on});
      row.append("rect").attr("x",m.l).attr("y",y).attr("width",Math.max(2, x(d.n)-m.l)).attr("height",bh-10)
        .attr("fill", d.kind==="reported" ? P.purple : P.A).attr("fill-opacity", on ? .95 : .5).attr("rx",3);
      txt(row, x(d.n)+6, y+bh/2, fmtN(d.n), {size:11, mono:true, fill:P.ink});
    });
    const d = CN.find(c=>c.id===cur), ref = CN.find(c=>c.id==="qf");
    const body = d.kind === "reported" ? `${comma(d.n)} (${d.src})` : `${d.f} = <b>${comma(d.n)}</b>`;
    d3.select("#cn-read").html(`<b>${d.name}</b>: ${body}. ` +
      (d.id === "qf" ? "" : `Ratio to the Q-Former: ${(d.n/ref.n) >= 1 ? (d.n/ref.n).toFixed(1)+"× larger" : (ref.n/d.n).toFixed(1)+"× smaller"}.`));
  }
  if (sel){ sel.value = cur; sel.addEventListener("change", ()=>{ cur = sel.value; draw(); }); }
  draw();
});

/* ───────────────────────── 05 · attention wiring ───────────────────────── */
MMV.safe("wiring", function(){
  const {P, txt, fillSelect} = MMV;
  const MODES = [
    {id:"dual",  name:"Dual encoder (CLIP)",               enc:"image tower", conn:"pool + project", lm:"text tower", img:"v"},
    {id:"llava", name:"Projector → causal LM (LLaVA)",      enc:"CLIP ViT (frozen)", conn:"linear / MLP", lm:"decoder LM", img:"v"},
    {id:"blip2", name:"Q-Former → causal LM (BLIP-2, OPT)", enc:"ViT (frozen)", conn:"Q-Former · 32 queries", lm:"frozen LM", img:"q"},
    {id:"prefix",name:"Prefix-LM (PaliGemma)",              enc:"SigLIP So400m", conn:"linear", lm:"Gemma, prefix-LM", img:"v"},
    {id:"flam",  name:"Gated cross-attention (Flamingo)",   enc:"NFNet-F6 (frozen)", conn:"resampler · 64 latents", lm:"frozen LM + xattn", img:"v"}
  ];
  const sel = fillSelect("wr-mode", MODES);
  const niIn = document.getElementById("wr-ni");
  const svg = d3.select("#wr-svg"), W = 640, H = 330;
  let mode = "prefix";
  function build(m, ni){
    const img = d3.range(ni).map(i => ({t:(m.img==="q"?"q":"v")+(i+1), k:"img"}));
    const pre = ["t1","t2","t3"].map(t=>({t, k:"pre"})), suf = ["a1","a2","a3"].map(t=>({t, k:"suf"}));
    const text = pre.concat(suf);
    let rows, cols, ok;
    if (m.id === "dual"){
      rows = cols = img.concat(text.map(d=>({t:d.t, k:"txt"})));
      ok = (i,j) => (i<ni && j<ni) || (i>=ni && j>=ni && j<=i);
    } else if (m.id === "flam"){
      rows = text; cols = img.concat(text);
      ok = (i,j) => j < ni ? true : (j-ni) <= i;
    } else {
      rows = cols = img.concat(text);
      const Pn = ni + 3;
      ok = m.id === "prefix" ? ((i,j) => j < Pn || j <= i) : ((i,j) => j <= i);
    }
    return {rows, cols, ok};
  }
  function draw(){
    const ni = +niIn.value; d3.select("#wr-niv").text(ni);
    const m = MODES.find(d=>d.id===mode) || MODES[0];
    const {rows, cols, ok} = build(m, ni);
    svg.selectAll("*").remove();
    const g = svg.append("g");
    /* block diagram */
    const bx = 16, bw = 170;
    [["image", m.enc, P.A], ["connector", m.conn, P.B], ["language side", m.lm, P.teal]].forEach((b,i)=>{
      const y = 30 + i*92;
      g.append("rect").attr("x",bx).attr("y",y).attr("width",bw).attr("height",54).attr("rx",6).attr("fill",b[2]).attr("fill-opacity",.14).attr("stroke",b[2]);
      txt(g, bx+bw/2, y+22, b[0], {anchor:"middle", size:10.5, fill:P.muted});
      txt(g, bx+bw/2, y+40, b[1], {anchor:"middle", size:11.5, fill:P.ink, bold:true});
      if (i<2) g.append("line").attr("x1",bx+bw/2).attr("x2",bx+bw/2).attr("y1",y+54).attr("y2",y+92).attr("stroke",P.muted).attr("marker-end",null);
    });
    if (m.id === "dual") txt(g, bx+bw/2, 316, "meet only at cos( f(I), g(T) )", {anchor:"middle", size:10.5, fill:P.B});
    if (m.id === "flam") txt(g, bx+bw/2, 316, "visual tokens enter via cross-attention", {anchor:"middle", size:10.5, fill:P.B});
    /* mask */
    const mx = 250, my = 40, cs = Math.min(26, Math.floor(260 / Math.max(rows.length, cols.length)));
    const colOf = k => k==="img" ? P.A : (k==="suf" ? P.teal : P.B);
    cols.forEach((c,j)=> txt(g, mx + j*cs + cs/2, my-6, c.t, {anchor:"middle", size:9, fill:colOf(c.k)}));
    let allowed = 0, imgReadsText = 0, textReadsImg = 0;
    rows.forEach((r,i)=>{
      txt(g, mx-6, my + i*cs + cs*0.65, r.t, {anchor:"end", size:9, fill:colOf(r.k)});
      cols.forEach((c,j)=>{
        const a = ok(i,j);
        if (a){ allowed++; if (r.k==="img" && c.k!=="img") imgReadsText++; if (r.k!=="img" && c.k==="img") textReadsImg++; }
        g.append("rect").attr("x",mx+j*cs+1).attr("y",my+i*cs+1).attr("width",cs-2).attr("height",cs-2).attr("rx",2)
          .attr("fill", a ? (m.id==="flam" && c.k==="img" ? P.purple : colOf(c.k)) : P.panel).attr("fill-opacity", a ? .75 : 1).attr("stroke",P.line);
      });
    });
    txt(g, mx, my + rows.length*cs + 16, "rows = queries · columns = keys", {size:10, fill:P.muted});
    const total = rows.length * cols.length;
    const imgRow = rows.some(r=>r.k==="img");
    d3.select("#wr-read").html(`<b>${m.name}</b>: ${allowed} of ${total} query–key pairs allowed (${(100*allowed/total).toFixed(0)}%). ` +
      `Text rows reading image keys: <b>${textReadsImg}</b>` + (imgRow ? ` · image rows reading text keys: <b>${imgReadsText}</b>` : " · image tokens are not queries in the text sequence") +
      (m.id==="prefix" ? " — the prefix-LM lets the image see the prompt." : (imgReadsText===0 && imgRow && m.id!=="dual" ? " — image tokens are fixed before the prompt is read." : ".")));
  }
  if (sel){ sel.value = mode; sel.addEventListener("change", ()=>{ mode = sel.value; draw(); }); }
  niIn.addEventListener("input", draw);
  draw();
});

/* ───────────────────────── 06 · contrastive batch toy ───────────────────────── */
MMV.safe("contrastive", function(){
  const {P, txt, lcg} = MMV;
  const N = 6, D = 8, labels = ["dog","cat","car","tree","boat","cake"];
  const rnd = lcg(7);
  const gauss = () => { const u = Math.max(1e-9, rnd()), v = rnd(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };
  const unit = a => { const n = Math.hypot(...a) || 1; return a.map(x=>x/n); };
  const U = d3.range(N).map(()=>unit(d3.range(D).map(gauss)));
  const Wn = d3.range(N).map(()=>unit(d3.range(D).map(gauss)));
  const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
  const lossSel = document.getElementById("cs-loss"), tIn = document.getElementById("cs-t"), aIn = document.getElementById("cs-a"), bIn = document.getElementById("cs-b");
  const svg = d3.select("#cs-svg");
  const lse = arr => { const m = Math.max(...arr); return m + Math.log(arr.reduce((s,x)=>s+Math.exp(x-m),0)); };
  const logSig = z => z >= 0 ? -Math.log1p(Math.exp(-z)) : z - Math.log1p(Math.exp(z));
  function draw(){
    const tau = +tIn.value, a = +aIn.value, b = +bIn.value, mode = lossSel.value;
    d3.select("#cs-tv").text(tau.toFixed(2)); d3.select("#cs-av").text(a.toFixed(2)); d3.select("#cs-bv").text(b);
    const V = d3.range(N).map(i => unit(U[i].map((x,k)=> a*x + (1-a)*Wn[i][k])));
    const S = d3.range(N).map(i => d3.range(N).map(j => dot(U[i], V[j])));
    const scale = 1/tau;
    let rowLoss, colLoss, mean, acc = 0;
    d3.range(N).forEach(i => { const r = S[i]; if (r.indexOf(Math.max(...r)) === i) acc++; });
    if (mode === "softmax"){
      rowLoss = d3.range(N).map(i => lse(S[i].map(s=>s*scale)) - S[i][i]*scale);
      colLoss = d3.range(N).map(j => lse(d3.range(N).map(i=>S[i][j]*scale)) - S[j][j]*scale);
      mean = 0.5*(d3.mean(rowLoss) + d3.mean(colLoss));
    } else {
      rowLoss = d3.range(N).map(i => -d3.range(N).reduce((s,j)=> s + logSig((i===j?1:-1)*(scale*S[i][j] + b)), 0));
      colLoss = rowLoss; mean = d3.sum(rowLoss)/N;
    }
    svg.selectAll("*").remove();
    const g = svg.append("g"), x0 = 90, y0 = 40, cs = 38;
    const col = d3.scaleLinear().domain([-1,0,1]).range([P.bad, P.panel, P.good]);
    labels.forEach((l,j)=> txt(g, x0 + j*cs + cs/2, y0-8, "T:"+l, {anchor:"middle", size:9.5, fill:P.muted}));
    S.forEach((row,i)=>{
      txt(g, x0-8, y0 + i*cs + cs*0.6, "I:"+labels[i], {anchor:"end", size:9.5, fill:P.muted});
      row.forEach((s,j)=>{
        g.append("rect").attr("x",x0+j*cs+1).attr("y",y0+i*cs+1).attr("width",cs-2).attr("height",cs-2).attr("rx",3)
          .attr("fill",col(s)).attr("stroke", i===j ? P.B : P.line).attr("stroke-width", i===j ? 2 : 1);
        txt(g, x0+j*cs+cs/2, y0+i*cs+cs*0.62, s.toFixed(2), {anchor:"middle", size:9.5, mono:true, fill:P.ink});
      });
      txt(g, x0 + N*cs + 14, y0 + i*cs + cs*0.62, rowLoss[i].toFixed(3), {size:10.5, mono:true, fill:P.B});
    });
    txt(g, x0 + N*cs + 14, y0-8, mode==="softmax" ? "row loss (nats)" : "row sigmoid loss", {size:9.5, fill:P.muted});
    txt(g, x0, y0 + N*cs + 20, `diagonal = matched pairs · logit = ${mode==="softmax" ? "s / τ" : "t·s + b"}`, {size:10, fill:P.muted});
    const chance = Math.log(N);
    d3.select("#cs-read").html(mode === "softmax" ?
      `logit scale 1/τ = <b>${scale.toFixed(2)}</b> · image→text loss ${d3.mean(rowLoss).toFixed(3)}, text→image ${d3.mean(colLoss).toFixed(3)}, symmetric <b>${mean.toFixed(3)}</b> nats (chance level ln ${N} = ${chance.toFixed(3)}) · image→text top-1 ${acc}/${N}` :
      `t = 1/τ = <b>${scale.toFixed(2)}</b>, b = ${b} · summed sigmoid loss per image <b>${mean.toFixed(3)}</b> over ${N} positives and ${N*(N-1)} negatives · image→text top-1 ${acc}/${N} (the argmax is unchanged by t and b)`);
  }
  [tIn, aIn, bIn].forEach(el => el.addEventListener("input", draw));
  lossSel.addEventListener("change", draw);
  draw();
});

/* ───────────────────────── 08 · lineage timeline ───────────────────────── */
MMV.safe("lineage", function(){
  const {P, txt} = MMV;
  const FAM = [{id:"fusion", name:"fusion encoders", col:P.pink}, {id:"dual", name:"dual encoders", col:P.A}, {id:"gen", name:"generative VLMs", col:P.teal}];
  const MS = [
    {n:"ViLBERT",     d:"2019-08-06", f:"fusion", w:"two-stream co-attention over detector regions and words; pretrained on Conceptual Captions"},
    {n:"LXMERT",      d:"2019-08-20", f:"fusion", w:"object-relationship, language and cross-modality encoders; 9.18M image–sentence pairs on 180K images"},
    {n:"ALIGN",       d:"2021-02-11", f:"dual",   w:"EfficientNet + BERT dual encoder on ≈1.8B noisy alt-text pairs; 76.4% zero-shot ImageNet"},
    {n:"CLIP",        d:"2021-02-26", f:"dual",   w:"ViT / ResNet image tower + text Transformer on 400M pairs; 76.2% zero-shot ImageNet (ViT-L/14@336)"},
    {n:"BLIP",        d:"2022-01-28", f:"gen",    w:"unified understanding and generation; CapFilt bootstraps synthetic captions for noisy web data"},
    {n:"Flamingo",    d:"2022-04-29", f:"gen",    w:"frozen NFNet-F6 + frozen Chinchilla, Perceiver resampler (64 tokens) and gated cross-attention; few-shot prompting"},
    {n:"BLIP-2",      d:"2023-01-30", f:"gen",    w:"Q-Former with 32 queries bridges a frozen image encoder and a frozen LLM"},
    {n:"SigLIP",      d:"2023-03-27", f:"dual",   w:"pairwise sigmoid loss; batch benefits saturate around 32k"},
    {n:"LLaVA",       d:"2023-04-17", f:"gen",    w:"CLIP ViT-L/14 + linear projector + Vicuna; 158K GPT-4-generated instruction samples"},
    {n:"InstructBLIP",d:"2023-05-11", f:"gen",    w:"instruction-aware Q-Former: the instruction is also fed to the queries"},
    {n:"LLaVA-1.5",   d:"2023-10-05", f:"gen",    w:"336 px, two-layer MLP projector, academic VQA data; ≈1 day on one 8-A100 node"},
    {n:"PaliGemma",   d:"2024-07-10", f:"gen",    w:"SigLIP So400m + Gemma 2B, linear projector, prefix-LM, nothing frozen; 224/448/896 px"},
    {n:"PaliGemma 2", d:"2024-12-04", f:"gen",    w:"same encoder with Gemma 2 at 2B, 9B and 27B"}
  ];
  const MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  MS.forEach(m => { const [y,mo,dd] = m.d.split("-").map(Number); m.t = y + (mo-1)/12 + (dd-1)/365; m.label = MON[mo-1]+" "+y; });
  const svg = d3.select("#ln-svg"), W = 640, H = 280, m = {l:110, r:20, t:20, b:34};
  const x = d3.scaleLinear().domain([2019.3, 2025.2]).range([m.l, W-m.r]);
  const rowY = f => m.t + 30 + FAM.findIndex(d=>d.id===f) * 76;
  let sel = MS.findIndex(d=>d.n==="BLIP-2");
  function draw(){
    svg.selectAll("*").remove();
    const g = svg.append("g");
    [2020,2021,2022,2023,2024,2025].forEach(v=>{
      g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, x(v), H-m.b+16, String(v), {anchor:"middle", size:10, fill:P.muted});
    });
    FAM.forEach(f=>{
      const y = rowY(f.id);
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y).attr("y2",y).attr("stroke",f.col).attr("stroke-opacity",.35);
      txt(g, m.l-10, y+4, f.name, {anchor:"end", size:10.5, fill:f.col});
    });
    const seen = {};
    MS.forEach((d,i)=>{
      const f = FAM.find(q=>q.id===d.f), y = rowY(d.f), k = (seen[d.f] = (seen[d.f]||0) + 1), up = k % 2 === 1, on = i === sel;
      const node = g.append("g").style("cursor","pointer").on("click", ()=>{ sel = i; draw(); });
      node.append("circle").attr("cx",x(d.t)).attr("cy",y).attr("r",on?7:5).attr("fill",on?P.B:f.col);
      txt(node, x(d.t), up ? y-11 : y+19, d.n, {anchor:"middle", size:10, fill:on?P.B:P.ink, bold:on});
    });
    const d = MS[sel];
    const prev = MS.slice(0, sel).filter(q=>q.f===d.f).pop();
    const gap = prev ? Math.round((d.t - prev.t)*12) : null;
    d3.select("#ln-read").html(`<b>${d.n}</b> (${d.label}, ${FAM.find(q=>q.id===d.f).name}): ${d.w}.` +
      (prev ? ` About ${gap} month${gap===1?"":"s"} after ${prev.n}.` : " First milestone of its family on this chart."));
  }
  draw();
});

/* ───────────────────────── 10 · prefill cost of the image ───────────────────────── */
MMV.safe("cost", function(){
  const {P, txt, fmtN, comma, fillSelect} = MMV;
  /* published decoder configurations; hk = key/value heads, hd = head dim, gated = SwiGLU/GeGLU (3 matrices) */
  const LM = [
    {id:"v7",  name:"Vicuna v1.5 7B (Llama 2 7B)",  L:32, d:4096, h:32, hk:32, hd:128, f:11008, gated:true,  V:32000,  tied:false, reported:6.74e9, repLabel:"6.74B total"},
    {id:"v13", name:"Vicuna v1.5 13B (Llama 2 13B)",L:40, d:5120, h:40, hk:40, hd:128, f:13824, gated:true,  V:32000,  tied:false, reported:13.0e9, repLabel:"13.0B total"},
    {id:"g2",  name:"Gemma 2B",                      L:18, d:2048, h:8,  hk:1,  hd:256, f:16384, gated:true,  V:256128, tied:true,  reportedNE:1981884416, repLabel:"1,981,884,416 non-embedding"},
    {id:"opt", name:"OPT-2.7B",                      L:32, d:2560, h:32, hk:32, hd:80,  f:10240, gated:false, V:50272,  tied:true,  reported:2.7e9, repLabel:"2.7B total"}
  ];
  const VIS = [
    {id:"q32",  name:"32 · BLIP-2 Q-Former", q:32},
    {id:"r224", name:"224 px / 14 · LLaVA, PaliGemma 224", r:224, p:14},
    {id:"r336", name:"336 px / 14 · LLaVA-1.5", r:336, p:14},
    {id:"r448", name:"448 px / 14 · PaliGemma 448", r:448, p:14},
    {id:"r896", name:"896 px / 14 · PaliGemma 896", r:896, p:14}
  ];
  VIS.forEach(v => { v.T = v.q ? v.q : Math.pow(Math.floor(v.r/v.p), 2); });
  LM.forEach(m => {
    const attn = m.d*m.h*m.hd + 2*m.d*m.hk*m.hd + m.h*m.hd*m.d;
    const ffn = (m.gated ? 3 : 2) * m.d * m.f;
    m.ne = m.L * (attn + ffn);
    m.emb = m.V*m.d*(m.tied ? 1 : 2);
  });
  const lmSel = fillSelect("co-lm", LM), visSel = fillSelect("co-vis", VIS), tIn = document.getElementById("co-txt");
  if (visSel) visSel.value = "r336";
  const svg = d3.select("#co-svg"), W = 640, H = 220;
  function sci(x){ const e = Math.floor(Math.log10(x)); const s = "⁰¹²³⁴⁵⁶⁷⁸⁹"; return (x/Math.pow(10,e)).toFixed(2)+" × 10"+String(e).split("").map(c=>s[+c]).join(""); }
  function draw(){
    const m = LM.find(d=>d.id===lmSel.value) || LM[0], v = VIS.find(d=>d.id===visSel.value) || VIS[2], Tt = +tIn.value;
    d3.select("#co-txtv").text(Tt);
    const T = v.T + Tt;
    const per = n => 2*m.ne*n + 2*m.L*m.d*n*T;   /* each query row attends to all T keys (upper bound) */
    const Ci = per(v.T), Ct = per(Tt), share = Ci/(Ci+Ct);
    svg.selectAll("*").remove();
    const g = svg.append("g"), x0 = 30, bw = W - 60;
    txt(g, x0, 24, `${m.name} · prompt = ${comma(v.T)} image + ${comma(Tt)} text tokens`, {size:11.5, fill:P.ink, bold:true});
    const wi = bw*share;
    g.append("rect").attr("x",x0).attr("y",44).attr("width",Math.max(1,wi)).attr("height",42).attr("fill",P.A).attr("fill-opacity",.8).attr("rx",3);
    g.append("rect").attr("x",x0+wi).attr("y",44).attr("width",Math.max(1,bw-wi)).attr("height",42).attr("fill",P.B).attr("fill-opacity",.8).attr("rx",3);
    txt(g, x0+4, 104, `image ${(100*share).toFixed(1)}% · ${sci(Ci)} FLOPs`, {size:11, fill:P.A});
    txt(g, x0+bw, 104, `text ${(100*(1-share)).toFixed(1)}% · ${sci(Ct)} FLOPs`, {size:11, fill:P.B, anchor:"end"});
    /* parameter check */
    const tot = m.ne + m.emb;
    const chk = m.reportedNE ? `computed non-embedding ${comma(m.ne)} vs reported ${m.repLabel} (${((m.reportedNE-m.ne)/m.reportedNE*100).toFixed(3)}% gap = norm weights)` :
      `computed ${fmtN(m.ne)} non-embedding + ${fmtN(m.emb)} embedding = ${fmtN(tot)} vs reported ${m.repLabel}`;
    txt(g, x0, 140, chk, {size:10.5, fill:P.muted});
    txt(g, x0, 162, `C ≈ 2·N·T + 2·L·d·T² with N = ${fmtN(m.ne)}, L = ${m.L}, d = ${m.d}, T = ${comma(T)}`, {size:10.5, fill:P.muted, mono:true});
    txt(g, x0, 184, `image tokens from ${v.q ? "a fixed "+v.q+" queries" : "floor("+v.r+"/"+v.p+")² = "+comma(v.T)}`, {size:10.5, fill:P.muted, mono:true});
    d3.select("#co-read").html(`<b>${m.name}</b>, ${v.name}: the image accounts for <b>${(100*share).toFixed(1)}%</b> of prefill compute (${sci(Ci)} of ${sci(Ci+Ct)} FLOPs). ` +
      `Equivalent text length: the image costs as much as a ${comma(v.T)}-token passage.`);
  }
  [lmSel, visSel].forEach(el => el && el.addEventListener("change", draw));
  tIn.addEventListener("input", draw);
  draw();
});
