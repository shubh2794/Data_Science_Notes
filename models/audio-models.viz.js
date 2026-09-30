/* audio-models.viz.js — every interactive figure on models/audio-models.html.
   Loaded after data.js / notes.js (palette C comes from notes.js).
   Each figure is its own IIFE wrapped in try/catch so one failure cannot blank the rest.
   House rule: every number a figure displays is computed here from a spec, never typed into a label. */

const AV = (function(){
  const P = Object.assign({}, (typeof C !== "undefined" ? C : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"}),
    { panel:"#15181f", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6" });

  /* wav2vec 2.0 feature encoder, shared by HuBERT / WavLM / XLS-R / MMS (released conv_kernel / conv_stride) */
  const CONV = { kernel:[10,3,3,3,3,2,2], stride:[5,2,2,2,2,2,2], ch:512 };
  /* Whisper log-mel stem: two width-3 convs, second with stride 2 (padding 1) */
  const MEL = { win_ms:25, stem:[{k:3,s:1},{k:3,s:2}], nctx:1500, textctx:448 };

  /* published configurations (paper tables + released config.json) */
  const WHISPER = {
    tiny:  {name:"Whisper tiny",   d:384,  Le:4,  Ld:4,  mels:80,  V:51865, reported:39e6},
    base:  {name:"Whisper base",   d:512,  Le:6,  Ld:6,  mels:80,  V:51865, reported:74e6},
    small: {name:"Whisper small",  d:768,  Le:12, Ld:12, mels:80,  V:51865, reported:244e6},
    medium:{name:"Whisper medium", d:1024, Le:24, Ld:24, mels:80,  V:51865, reported:769e6},
    large: {name:"Whisper large-v2", d:1280, Le:32, Ld:32, mels:80, V:51865, reported:1550e6},
    v3:    {name:"Whisper large-v3", d:1280, Le:32, Ld:32, mels:128, V:51866, reported:1550e6},
    turbo: {name:"Whisper large-v3-turbo", d:1280, Le:32, Ld:4, mels:128, V:51866, reported:809e6}
  };
  const SSL = {
    w2vb: {name:"wav2vec 2.0 BASE",  d:768,  L:12, F:3072, reported:95e6},
    w2vl: {name:"wav2vec 2.0 LARGE", d:1024, L:24, F:4096, reported:317e6},
    hubx: {name:"HuBERT X-LARGE",    d:1280, L:48, F:5120, reported:964e6},
    wlml: {name:"WavLM Large",       d:1024, L:24, F:4096, reported:316.62e6},
    xls1: {name:"XLS-R 1B",          d:1280, L:48, F:5120, reported:965e6},
    xls2: {name:"XLS-R 2B",          d:1920, L:48, F:7680, reported:2.16e9}
  };

  function whisperParams(c, Ld){
    const d = c.d, L = (Ld == null ? c.Ld : Ld);
    const stem = c.mels*d*3 + d + d*d*3 + d;
    const encBlocks = c.Le*(12*d*d + 12*d) + 2*d;       // q,k,v,o (k without bias) + FFN 4d + 2 LNs; final LN
    const encPos = MEL.nctx*d;                           // fixed sinusoidal table, stored with the weights
    const decBlocks = L*(16*d*d + 17*d) + 2*d;           // self-attn + cross-attn + FFN + 3 LNs; final LN
    const tokEmb = c.V*d, decPos = MEL.textctx*d;        // output projection tied to tokEmb
    const parts = [
      {k:"conv stem", v:stem}, {k:"encoder blocks", v:encBlocks}, {k:"encoder positions (fixed)", v:encPos},
      {k:"decoder blocks", v:decBlocks}, {k:"token embedding (tied)", v:tokEmb}, {k:"decoder positions", v:decPos}
    ];
    return {parts, total: parts.reduce((a,p)=>a+p.v,0)};
  }
  function sslParams(c){
    const d = c.d, ch = CONV.ch;
    let fe = 0, cin = 1;
    CONV.kernel.forEach(k=>{ fe += cin*ch*k; cin = ch; });  // conv layers, no bias
    fe += 2*ch;                                             // group norm after the first layer
    const proj = 2*ch + ch*d + d;                           // layer norm + linear to d
    const pos = d*(d/16)*128 + d;                           // grouped positional conv, kernel 128, 16 groups (+ weight-norm gain)
    const blocks = c.L*(4*d*d + 4*d + 2*d + d*c.F + c.F + c.F*d + d + 2*d) + 2*d;
    const parts = [{k:"conv feature encoder", v:fe}, {k:"projection to d", v:proj}, {k:"positional conv", v:pos}, {k:"Transformer blocks", v:blocks}];
    return {parts, total: parts.reduce((a,p)=>a+p.v,0)};
  }
  function convOut(Lin){
    const lens = [Lin]; let L = Lin;
    CONV.kernel.forEach((k,i)=>{ L = Math.floor((L - k)/CONV.stride[i]) + 1; lens.push(Math.max(L,0)); });
    return lens;
  }
  function receptive(){
    let r = 1, jump = 1;
    CONV.kernel.forEach((k,i)=>{ r += (k-1)*jump; jump *= CONV.stride[i]; });
    return {r, hop:jump};
  }
  function fmtN(x){
    const a = Math.abs(x);
    if (a >= 1e9)  return (x/1e9).toFixed(2)+"B";
    if (a >= 1e6)  return String(parseFloat((x/1e6).toFixed(a>=1e8?2:1)))+"M";
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
  function safe(name, fn){ try { fn(); } catch(e){ if (typeof console!=="undefined") console.error("[audio.viz] "+name+" failed:", e); } }
  function lcg(seed){ let s = seed >>> 0; return () => { s = (1664525*s + 1013904223) >>> 0; return s/4294967296; }; }
  return {P, CONV, MEL, WHISPER, SSL, whisperParams, sslParams, convOut, receptive, fmtN, comma, txt, safe, lcg};
})();

/* ───────────────────────── 02 · two front ends ───────────────────────── */
AV.safe("front", function(){
  const {P, CONV, MEL, convOut, receptive, comma, txt} = AV;
  const svg = d3.select("#fe-svg"), W = 640, H = 300, m = {l:56, r:12, t:28, b:62};
  const secEl = document.getElementById("fe-sec"), srEl = document.getElementById("fe-sr"), hopEl = document.getElementById("fe-hop");
  function draw(){
    const sec = +secEl.value, sr = +srEl.value, hopMs = +hopEl.value;
    document.getElementById("fe-secv").textContent = sec;
    const N = sec*sr;
    // log-mel path: frames = N / hop samples; then the stem (k3 s1 p1 keeps length, k3 s2 p1 halves, rounding up)
    const hopS = Math.round(sr*hopMs/1000), winS = Math.round(sr*MEL.win_ms/1000);
    const melFrames = Math.floor(N/hopS);
    let L = melFrames; const stemLens = [];
    MEL.stem.forEach(s=>{ L = Math.floor((L + 2 - s.k)/s.s) + 1; stemLens.push(L); });
    const melPos = stemLens[stemLens.length-1];
    const melBars = [{k:"samples", v:N}, {k:"mel frames", v:melFrames}, {k:"conv k3 s1", v:stemLens[0]}, {k:"conv k3 s2", v:stemLens[1]}];
    // conv path
    const lens = convOut(N);
    const cvBars = lens.map((v,i)=>({k: i===0 ? "samples" : ("k"+CONV.kernel[i-1]+" s"+CONV.stride[i-1]), v}));
    const rf = receptive();
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const all = melBars.concat(cvBars);
    const y = d3.scaleLog().domain([10, d3.max(all, d=>Math.max(d.v,10))*1.3]).range([H-m.b, m.t]);
    const leftW = 190, gap = 40;
    const xa = d3.scaleBand().domain(melBars.map((d,i)=>i)).range([m.l, m.l+leftW]).padding(0.25);
    const xb = d3.scaleBand().domain(cvBars.map((d,i)=>i)).range([m.l+leftW+gap, W-m.r]).padding(0.2);
    [1e1,1e2,1e3,1e4,1e5,1e6].filter(v=>v<=y.domain()[1]).forEach(v=>{
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, m.l-6, y(v)+4, comma(v), {anchor:"end", size:9, fill:P.muted});
    });
    function bars(data, x, col){
      data.forEach((d,i)=>{
        const v = Math.max(d.v, 10);
        g.append("rect").attr("x",x(i)).attr("y",y(v)).attr("width",x.bandwidth()).attr("height",H-m.b-y(v)).attr("fill",col).attr("fill-opacity", i===data.length-1?0.95:0.55);
        txt(g, x(i)+x.bandwidth()/2, y(v)-4, comma(d.v), {anchor:"middle", size:8.5, fill:P.ink});
        g.append("text").attr("transform",`translate(${x(i)+x.bandwidth()/2},${H-m.b+10}) rotate(35)`).attr("font-size",8.5).attr("fill",P.muted).text(d.k);
      });
    }
    bars(melBars, xa, P.B); bars(cvBars, xb, P.A);
    txt(g, m.l, 14, "log-mel + conv stem (Whisper)", {size:11, fill:P.B, bold:true});
    txt(g, m.l+leftW+gap, 14, "learned conv feature encoder (wav2vec 2.0)", {size:11, fill:P.A, bold:true});
    const cvPos = lens[lens.length-1];
    const hopMsConv = rf.hop/sr*1000, rfMs = rf.r/sr*1000;
    const posMel = melPos, msMel = sec*1000/Math.max(posMel,1), msConv = sec*1000/Math.max(cvPos,1);
    d3.select("#fe-read").html(
      `${sec} s at ${sr/1000} kHz = <b>${comma(N)}</b> samples. ` +
      `Log-mel: window ${winS} samples, hop ${hopS} → <b>${comma(melFrames)}</b> frames → <b>${comma(posMel)}</b> encoder positions (${msMel.toFixed(1)} ms each). ` +
      `Conv stack: receptive field <b>${rf.r}</b> samples (${rfMs.toFixed(1)} ms), hop <b>${rf.hop}</b> samples (${hopMsConv.toFixed(1)} ms) → <b>${comma(cvPos)}</b> positions (${msConv.toFixed(1)} ms each). ` +
      `Compression: ${comma(N/Math.max(posMel,1))}× vs ${comma(N/Math.max(cvPos,1))}×.`);
  }
  [secEl, srEl, hopEl].forEach(el=>el.addEventListener(el.tagName==="SELECT"?"change":"input", draw));
  draw();
});

/* ───────────────────────── 03 · design-axis matrix ───────────────────────── */
AV.safe("axes", function(){
  const {P, txt, comma} = AV;
  const svg = d3.select("#ax-svg"), W = 640, H = 330;
  const axes = ["front end","training signal","text output","wiring","pretraining audio"];
  /* hours: pretraining (or training) audio as each paper reports it */
  const M = [
    {name:"Whisper",        v:["log-mel","weakly supervised","attention decoder","encoder–decoder"], hours:680000, note:"680 k h of web audio with transcripts; 30 s windows; task tokens select transcription, translation, language ID"},
    {name:"wav2vec 2.0",    v:["raw waveform","self-supervised","CTC (fine-tune)","encoder-only"], hours:53200, note:"LARGE pretrained on 53.2 k h of Libri-Light; masked contrastive loss with a G = 2, V = 320 quantiser"},
    {name:"HuBERT",         v:["raw waveform","self-supervised","CTC (fine-tune)","encoder-only"], hours:60000, note:"LARGE / X-LARGE on 60 k h Libri-Light; masked prediction of k-means cluster ids"},
    {name:"WavLM",          v:["raw waveform","self-supervised","CTC (fine-tune)","encoder-only"], hours:94000, note:"Base+ / Large on 94 k h (Libri-Light 60 k, GigaSpeech 10 k, VoxPopuli English 24 k); denoising masked prediction"},
    {name:"XLS-R",          v:["raw waveform","self-supervised","CTC (fine-tune)","encoder-only"], hours:436000, note:"436 k h across 128 languages; wav2vec 2.0 objective at 0.3B, 1B and 2B"},
    {name:"MMS",            v:["raw waveform","self-supervised","CTC (fine-tune)","encoder-only"], hours:491000, note:"491 k h across 1,406 languages; CTC recognition for 1,107 languages"},
    {name:"Conformer",      v:["log-mel","supervised","transducer","encoder + prediction net"], hours:960, note:"trained on 960 h of transcribed LibriSpeech; 80-channel filterbanks, single-LSTM decoder"},
    {name:"Deep Speech 2",  v:["spectrogram","supervised","CTC","encoder-only"], hours:11940, note:"11,940 h of transcribed English; 3 conv + 7 bidirectional recurrent layers, CTC loss"}
  ];
  const sel = document.getElementById("ax-axis");
  const pal = [P.A, P.B, P.good, P.purple, P.pink, P.teal];
  let pick = {r:0, c:1};
  const hoursBin = h => h < 5000 ? "< 5 k h" : (h < 100000 ? "5–100 k h" : "≥ 100 k h");
  function val(row, c){ return c < 4 ? row.v[c] : hoursBin(row.hours); }
  function draw(){
    const ax = +sel.value;
    svg.selectAll("*").remove();
    const g = svg.append("g"), x0 = 118, cw = (W - x0 - 10)/5, rh = 34, y0 = 34;
    axes.forEach((a,c)=> txt(g, x0 + c*cw + cw/2, 22, a, {anchor:"middle", size:10.5, fill: c===ax ? P.B : P.muted, bold: c===ax}));
    const cats = [...new Set(M.map(r=>val(r,ax)))];
    const hx = d3.scaleLog().domain([100, 1e6]).range([0, cw-12]);
    M.forEach((row,r)=>{
      const y = y0 + r*rh;
      txt(g, x0-8, y+rh/2+4, row.name, {anchor:"end", size:11});
      axes.forEach((a,c)=>{
        const v = val(row,c), on = (pick.r===r && pick.c===c);
        const col = pal[cats.indexOf(val(row,ax)) % pal.length];
        const cell = g.append("g").style("cursor","pointer").on("click",()=>{ pick = {r, c}; draw(); });
        cell.append("rect").attr("x",x0+c*cw+2).attr("y",y+2).attr("width",cw-4).attr("height",rh-4).attr("rx",5)
          .attr("fill", col).attr("fill-opacity", c===ax ? 0.55 : 0.16).attr("stroke", on ? P.B : "none").attr("stroke-width",2);
        if (c === 4){
          cell.append("rect").attr("x",x0+c*cw+8).attr("y",y+rh-10).attr("width",Math.max(1,hx(Math.max(row.hours,100)))).attr("height",4).attr("fill",P.ink).attr("fill-opacity",.6);
          txt(cell, x0+c*cw+cw/2, y+rh/2+1, comma(row.hours)+" h", {anchor:"middle", size:10});
        } else txt(cell, x0+c*cw+cw/2, y+rh/2+4, v, {anchor:"middle", size:10});
      });
    });
    const row = M[pick.r], same = M.filter(o=>val(o,ax)===val(row,ax)).map(o=>o.name);
    d3.select("#ax-read").html(`<b>${row.name}</b> · ${axes[pick.c]}: <b>${pick.c===4 ? comma(row.hours)+" h" : val(row,pick.c)}</b>. ${row.note}. ` +
      `Coloured by ${axes[ax]}: ${cats.length} distinct values; ${same.length} of ${M.length} models ${same.length===1?"has":"share"} ${row.name}'s value (${same.join(", ")}).`);
  }
  sel.addEventListener("change", draw);
  draw();
});

/* ───────────────────────── 04 · lineage scatter ───────────────────────── */
AV.safe("lineage", function(){
  const {P, fmtN, txt, whisperParams, WHISPER} = AV;
  const svg = d3.select("#lin-svg"), W = 640, H = 300, m = {l:58, r:20, t:20, b:46};
  const col = {sup:P.muted, ssl:P.A, weak:P.B};
  const lab = {sup:"supervised", ssl:"self-supervised", weak:"weakly supervised"};
  const R = [
    {name:"Deep Speech 2", t:2015.95, sizes:[100e6], kind:"sup", what:"~100M-parameter CTC recogniser (3 conv + 7 recurrent layers) on 11,940 h English"},
    {name:"Conformer", t:2020.38, sizes:[10.3e6, 30.7e6, 118.8e6], kind:"sup", what:"convolution-augmented Transformer encoder, S / M / L, supervised on LibriSpeech"},
    {name:"wav2vec 2.0", t:2020.47, sizes:[95e6, 317e6], kind:"ssl", what:"masked contrastive pretraining with a quantiser; BASE and LARGE"},
    {name:"HuBERT", t:2021.46, sizes:[95e6, 317e6, 964e6], kind:"ssl", what:"masked prediction of k-means cluster ids; BASE, LARGE, X-LARGE"},
    {name:"WavLM", t:2021.8, sizes:[94.70e6, 316.62e6], kind:"ssl", what:"denoising masked prediction, gated relative position bias; Base(+) and Large"},
    {name:"XLS-R", t:2021.88, sizes:[317e6, 965e6, 2.16e9], kind:"ssl", what:"wav2vec 2.0 objective on 436 k h in 128 languages"},
    {name:"Whisper", t:2022.72, sizes:["tiny","base","small","medium","large"].map(k=>WHISPER[k].reported), kind:"weak", what:"680 k h weakly supervised, five sizes from tiny to large (large-v2 followed in December 2022)"},
    {name:"MMS", t:2023.39, sizes:[300e6, 1e9], kind:"ssl", what:"wav2vec 2.0-style pretraining on 491 k h in 1,406 languages; ASR for 1,107"},
    {name:"large-v3", t:2023.85, sizes:[WHISPER.v3.reported], kind:"weak", what:"128 mel bins, Cantonese token, 1 M h weak + 4 M h pseudo-labelled audio"},
    {name:"large-v3-turbo", t:2024.75, sizes:[WHISPER.turbo.reported], kind:"weak", what:"large-v3 with the decoder pruned from 32 to 4 layers, then fine-tuned"}
  ];
  const x = d3.scaleLinear().domain([2015.5, 2025.1]).range([m.l, W-m.r]);
  const y = d3.scaleLog().domain([5e6, 5e9]).range([H-m.b, m.t]);
  let sel = 6;
  function draw(){
    svg.selectAll("*").remove();
    const g = svg.append("g");
    [1e7,1e8,1e9].forEach(v=>{
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, m.l-6, y(v)+4, fmtN(v), {anchor:"end", size:10, fill:P.muted});
    });
    d3.range(2016, 2026, 1).forEach(v=> txt(g, x(v), H-m.b+16, String(v), {anchor:"middle", size:10, fill:P.muted}));
    txt(g, 10, m.t+2, "params", {size:10, fill:P.muted});
    Object.keys(lab).forEach((k,i)=>{
      g.append("circle").attr("cx",m.l+8+i*140).attr("cy",H-10).attr("r",5).attr("fill",col[k]);
      txt(g, m.l+18+i*140, H-6, lab[k], {size:10, fill:P.muted});
    });
    R.forEach((d,i)=>{
      const on = i===sel, cx = x(d.t), lo = d3.min(d.sizes), hi = d3.max(d.sizes);
      const node = g.append("g").style("cursor","pointer").on("click",()=>{ sel=i; draw(); });
      if (d.sizes.length>1) node.append("line").attr("x1",cx).attr("x2",cx).attr("y1",y(lo)).attr("y2",y(hi)).attr("stroke",col[d.kind]).attr("stroke-width",on?4:2.5).attr("stroke-opacity",.5);
      d.sizes.forEach(s=> node.append("circle").attr("cx",cx).attr("cy",y(s)).attr("r",on?6:4.5).attr("fill",col[d.kind]).attr("stroke",on?P.ink:"none"));
      const up = i%2===0;
      txt(node, cx, up ? y(hi)-9 : y(lo)+16, d.name, {anchor:"middle", size:10, fill:on?P.B:P.ink, bold:on});
    });
    const d = R[sel], lo = d3.min(d.sizes), hi = d3.max(d.sizes);
    const prev = R.slice(0, sel).filter(r=>r.kind===d.kind).pop();
    d3.select("#lin-read").html(`<b>${d.name}</b> (${Math.floor(d.t)}, ${lab[d.kind]}): ${d.what}. ` +
      (d.sizes.length>1 ? `${d.sizes.length} sizes, <b>${fmtN(lo)}</b> to <b>${fmtN(hi)}</b> (${(hi/lo).toFixed(1)}× range)` : `size <b>${fmtN(hi)}</b>`) +
      (prev ? ` · largest is ${(hi/d3.max(prev.sizes)).toFixed(2)}× the previous ${lab[d.kind]} release's largest (${prev.name}).` : "."));
  }
  draw();
});

/* ───────────────────────── 05 · parameter counts from configs ───────────────────────── */
AV.safe("params", function(){
  const {P, WHISPER, SSL, whisperParams, sslParams, fmtN, comma, txt} = AV;
  const svg = d3.select("#pc-svg"), W = 640, H = 250;
  const sel = document.getElementById("pc-preset"), dec = document.getElementById("pc-dec");
  const opts = Object.keys(WHISPER).map(k=>({id:"w:"+k, name:WHISPER[k].name}))
    .concat(Object.keys(SSL).map(k=>({id:"s:"+k, name:SSL[k].name})));
  opts.forEach(o=>{ const op = document.createElement("option"); op.value = o.id; op.textContent = o.name; sel.appendChild(op); });
  sel.value = "w:v3";
  const pal = [P.muted, P.A, "#3a4a66", P.B, P.purple, P.pink];
  let lastPreset = sel.value;
  function draw(){
    const [kind, key] = sel.value.split(":");
    if (sel.value !== lastPreset){ lastPreset = sel.value; if (kind==="w") dec.value = WHISPER[key].Ld; }
    let res, cfg, note;
    if (kind === "w"){
      cfg = WHISPER[key]; dec.disabled = false;
      const Ld = +dec.value; document.getElementById("pc-decv").textContent = Ld;
      res = whisperParams(cfg, Ld);
      const full = whisperParams(cfg, cfg.Ld).total;
      const decShare = res.parts[3].v / res.total;
      note = `d = ${cfg.d}, ${cfg.Le} encoder / <b>${Ld}</b> decoder layers, ${cfg.mels} mel bins, V = ${comma(cfg.V)}. Decoder blocks are ${(100*decShare).toFixed(1)}% of this total` +
        (Ld !== cfg.Ld ? `; the checkpoint as released (${cfg.Ld} decoder layers) counts ${fmtN(full)}.` : ".");
    } else {
      cfg = SSL[key]; dec.disabled = true; document.getElementById("pc-decv").textContent = "—";
      res = sslParams(cfg);
      note = `d = ${cfg.d}, ${cfg.L} layers, FFN ${cfg.F}; encoder only — the quantiser and projection heads used in pretraining are not counted, so a small shortfall is expected.`;
    }
    svg.selectAll("*").remove();
    const g = svg.append("g"), x0 = 20, x1 = W-20;
    const maxT = Math.max(res.total, cfg.reported) * 1.05;
    const x = d3.scaleLinear().domain([0, maxT]).range([x0, x1]);
    let acc = 0;
    res.parts.forEach((p,i)=>{
      g.append("rect").attr("x",x(acc)).attr("y",40).attr("width",Math.max(0, x(acc+p.v)-x(acc))).attr("height",46).attr("fill",pal[i%pal.length]).attr("fill-opacity",.85);
      acc += p.v;
    });
    g.append("line").attr("x1",x(cfg.reported)).attr("x2",x(cfg.reported)).attr("y1",30).attr("y2",96).attr("stroke",P.good).attr("stroke-width",2).attr("stroke-dasharray","4,3");
    txt(g, x(cfg.reported), 24, "reported " + fmtN(cfg.reported), {anchor: x(cfg.reported) > W-120 ? "end" : "middle", size:10.5, fill:P.good});
    txt(g, x0, 116, cfg.name + " — counted " + fmtN(res.total), {size:12, bold:true});
    res.parts.forEach((p,i)=>{
      const col = i%2, row = Math.floor(i/2), lx = x0 + col*310, ly = 142 + row*24;
      g.append("rect").attr("x",lx).attr("y",ly-9).attr("width",11).attr("height",11).attr("fill",pal[i%pal.length]);
      txt(g, lx+17, ly, `${p.k}: ${fmtN(p.v)} (${(100*p.v/res.total).toFixed(1)}%)`, {size:10.5, fill:P.muted});
    });
    const diff = 100*(res.total - cfg.reported)/cfg.reported;
    const modified = kind === "w" && +dec.value !== cfg.Ld;
    d3.select("#pc-read").html(modified
      ? `<b>${cfg.name}</b> with its decoder cut to ${dec.value} layers: counted <b>${comma(res.total)}</b> parameters. ${note}`
      : `<b>${cfg.name}</b>: counted <b>${comma(res.total)}</b> parameters vs reported <b>${fmtN(cfg.reported)}</b> (${diff>=0?"+":""}${diff.toFixed(2)}%). ${note}`);
  }
  sel.addEventListener("change", draw);
  dec.addEventListener("input", draw);
  draw();
});

/* ───────────────────────── 06 · CTC lattice ───────────────────────── */
AV.safe("ctc", function(){
  const {P, txt, lcg, comma} = AV;
  const svg = d3.select("#ctc-svg"), W = 640, H = 330;
  const ySel = document.getElementById("ctc-y"), TEl = document.getElementById("ctc-T");
  const BL = "∅";
  let rnd = lcg(7), path = null;
  function ext(y){ const e = [BL]; y.split("").forEach(ch=>{ e.push(ch); e.push(BL); }); return e; }
  function minFrames(y){ let r = y.length; for (let i=1;i<y.length;i++) if (y[i]===y[i-1]) r++; return r; }
  function allowedPrev(e, s){ const p = [s]; if (s>=1) p.push(s-1); if (s>=2 && e[s]!==BL && e[s]!==e[s-2]) p.push(s-2); return p; }
  function counts(e, T){
    const S = e.length, a = [], b = [];
    for (let t=0;t<T;t++){ a.push(new Array(S).fill(0)); b.push(new Array(S).fill(0)); }
    a[0][0] = 1; if (S>1) a[0][1] = 1;
    for (let t=1;t<T;t++) for (let s=0;s<S;s++) a[t][s] = allowedPrev(e,s).reduce((acc,q)=>acc+a[t-1][q],0);
    b[T-1][S-1] = 1; if (S>1) b[T-1][S-2] = 1;
    for (let t=T-2;t>=0;t--) for (let s=0;s<S;s++){
      let v = 0; for (let n=s;n<Math.min(S,s+3);n++) if (allowedPrev(e,n).indexOf(s)>=0) v += b[t+1][n];
      b[t][s] = v;
    }
    return {a, b, total: a[T-1][S-1] + (S>1 ? a[T-1][S-2] : 0)};
  }
  function sample(e, T, c){
    const S = e.length, p = [];
    let starts = [0,1].filter(s=>s<S && c.b[0][s]>0), w = starts.map(s=>c.b[0][s]);
    let s = pickW(starts, w); p.push(s);
    for (let t=1;t<T;t++){
      const nxt = []; for (let n=s;n<Math.min(S,s+3);n++) if (allowedPrev(e,n).indexOf(s)>=0 && c.b[t][n]>0) nxt.push(n);
      s = pickW(nxt, nxt.map(n=>c.b[t][n])); p.push(s);
    }
    return p;
  }
  function pickW(items, w){ const tot = w.reduce((x,y)=>x+y,0); let r = rnd()*tot; for (let i=0;i<items.length;i++){ r -= w[i]; if (r < 0) return items[i]; } return items[items.length-1]; }
  function collapse(seq){ const merged = []; seq.forEach((c,i)=>{ if (i===0 || c!==seq[i-1]) merged.push(c); }); return {merged, out: merged.filter(c=>c!==BL)}; }
  function draw(resample){
    const y = ySel.value, e = ext(y), mn = minFrames(y);
    if (+TEl.value < mn) TEl.value = String(mn);
    const T = +TEl.value; document.getElementById("ctc-Tv").textContent = T;
    const c = counts(e, T);
    if (resample || !path || path.length !== T || path.e !== y) { path = sample(e, T, c); path.e = y; }
    svg.selectAll("*").remove();
    const g = svg.append("g"), x0 = 60, y0 = 30, S = e.length;
    const cw = Math.min(40, (W - x0 - 20)/T), rh = Math.min(26, 220/S);
    e.forEach((sym,s)=> txt(g, x0-10, y0 + s*rh + rh/2 + 4, sym, {anchor:"end", size:12, fill: sym===BL ? P.muted : P.ink, mono:true}));
    for (let t=0;t<T;t++) txt(g, x0 + t*cw + cw/2, y0-8, "t"+(t+1), {anchor:"middle", size:9.5, fill:P.muted});
    const maxA = d3.max(c.a.flat());
    for (let t=0;t<T;t++) for (let s=0;s<S;s++){
      const live = c.a[t][s]*c.b[t][s] > 0;
      g.append("rect").attr("x",x0+t*cw+2).attr("y",y0+s*rh+2).attr("width",cw-4).attr("height",rh-4).attr("rx",3)
        .attr("fill", live ? P.A : P.line).attr("fill-opacity", live ? 0.12 + 0.5*Math.log1p(c.a[t][s]*c.b[t][s])/Math.log1p(maxA*maxA) : 0.35);
    }
    const pts = path.map((s,t)=>[x0+t*cw+cw/2, y0+s*rh+rh/2]);
    g.append("path").datum(pts).attr("fill","none").attr("stroke",P.B).attr("stroke-width",2.5).attr("d", d3.line());
    pts.forEach(p=> g.append("circle").attr("cx",p[0]).attr("cy",p[1]).attr("r",4).attr("fill",P.B));
    const frameSeq = path.map(s=>e[s]), col = collapse(frameSeq);
    const yb = y0 + S*rh + 26;
    txt(g, x0-10, yb, "frames", {anchor:"end", size:10, fill:P.muted});
    txt(g, x0, yb, frameSeq.join(" "), {size:12, mono:true});
    txt(g, x0-10, yb+22, "merge", {anchor:"end", size:10, fill:P.muted});
    txt(g, x0, yb+22, col.merged.join(" "), {size:12, mono:true, fill:P.muted});
    txt(g, x0-10, yb+44, "output", {anchor:"end", size:10, fill:P.muted});
    txt(g, x0, yb+44, '"' + col.out.join("") + '"', {size:12, mono:true, fill:P.good, bold:true});
    const ok = col.out.join("") === y;
    d3.select("#ctc-read").html(`target "<b>${y}</b>" (U = ${y.length}) → ${S} lattice rows; minimum frames <b>${mn}</b>. With T = ${T} there ${c.total===1?"is":"are"} <b>${comma(c.total)}</b> valid alignment${c.total===1?"":"s"} (forward DP, all probabilities 1). ` +
      `Sampled path collapses to "${col.out.join("")}" — ${ok ? "matches the target" : "MISMATCH"}; it spends ${frameSeq.filter(v=>v===BL).length} of ${T} frames on blank.`);
  }
  ySel.addEventListener("change", ()=>draw(true));
  TEl.addEventListener("input", ()=>draw(true));
  document.getElementById("ctc-sample").addEventListener("click", ()=>draw(true));
  draw(true);
});

/* ───────────────────────── 08 · span masking ───────────────────────── */
AV.safe("mask", function(){
  const {P, txt, lcg} = AV;
  const svg = d3.select("#mk-svg"), W = 640, H = 220;
  const PRE = { w2v:{p:0.065, M:10, name:"wav2vec 2.0"}, hub:{p:0.08, M:10, name:"HuBERT"} };
  const preEl = document.getElementById("mk-preset"), pEl = document.getElementById("mk-p"), MEl = document.getElementById("mk-M");
  const Tn = 250, frameMs = 20;
  let seed = 3;
  function expected(p, M){ return 1 - Math.pow(1-p, M); }
  function draw(){
    const p = +pEl.value, M = +MEl.value;
    document.getElementById("mk-pv").textContent = p.toFixed(3);
    document.getElementById("mk-Mv").textContent = M;
    const rnd = lcg(seed), mask = new Array(Tn).fill(0);
    let starts = 0;
    for (let t=0;t<Tn;t++) if (rnd() < p){ starts++; for (let j=t;j<Math.min(Tn,t+M);j++) mask[j] = 1; }
    const emp = mask.reduce((a,b)=>a+b,0)/Tn, ex = expected(p, M);
    svg.selectAll("*").remove();
    const g = svg.append("g"), x0 = 40, w = (W - x0 - 20)/Tn;
    mask.forEach((v,t)=> g.append("rect").attr("x",x0+t*w).attr("y",16).attr("width",Math.max(w-0.3,0.5)).attr("height",26).attr("fill", v ? P.B : P.A).attr("fill-opacity", v ? 0.9 : 0.25));
    txt(g, x0, 56, `${Tn} frames = ${(Tn*frameMs/1000).toFixed(1)} s · orange = masked`, {size:10, fill:P.muted});
    const px = d3.scaleLinear().domain([0, 0.2]).range([x0, W-20]), py = d3.scaleLinear().domain([0,1]).range([H-24, 76]);
    [0,0.25,0.5,0.75,1].forEach(v=>{ g.append("line").attr("x1",x0).attr("x2",W-20).attr("y1",py(v)).attr("y2",py(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3"); txt(g, x0-4, py(v)+3, (v*100)+"%", {anchor:"end", size:9, fill:P.muted}); });
    [0,0.05,0.1,0.15,0.2].forEach(v=> txt(g, px(v), H-8, "p = "+v.toFixed(2), {anchor:"middle", size:9, fill:P.muted}));
    const curve = d3.range(0, 0.2001, 0.002).map(q=>[px(q), py(expected(q, M))]);
    g.append("path").datum(curve).attr("fill","none").attr("stroke",P.A).attr("stroke-width",2).attr("d", d3.line());
    g.append("circle").attr("cx",px(p)).attr("cy",py(ex)).attr("r",5).attr("fill",P.B);
    g.append("circle").attr("cx",px(p)).attr("cy",py(emp)).attr("r",4).attr("fill","none").attr("stroke",P.good).attr("stroke-width",2);
    txt(g, px(p)+8, py(ex)-6, `${(100*ex).toFixed(1)}% expected`, {size:10, fill:P.B});
    d3.select("#mk-read").html(`p = ${p.toFixed(3)}, M = ${M}: expected masked fraction 1 − (1 − p)ᴹ = <b>${(100*ex).toFixed(1)}%</b>; this draw started ${starts} spans and masked <b>${(100*emp).toFixed(1)}%</b> of ${Tn} frames (green ring). ` +
      `Span starts are ${(100*p).toFixed(1)}% of frames, yet ${(ex/p).toFixed(1)}× that fraction is hidden.`);
  }
  preEl.addEventListener("change", ()=>{ const q = PRE[preEl.value]; if (q){ pEl.value = String(q.p); MEl.value = String(q.M); } draw(); });
  [pEl, MEl].forEach(el=>el.addEventListener("input", ()=>{ const q = PRE[preEl.value]; if (q && (Math.abs(+pEl.value-q.p)>1e-9 || +MEl.value!==q.M)) preEl.value = "custom"; draw(); }));
  document.getElementById("mk-resample").addEventListener("click", ()=>{ seed = (seed*7 + 11) % 100003; draw(); });
  draw();
});
