/* nlp-models.viz.js — interactive figures for models/nlp-models.html (the NLP Models hub).
   Loaded after data.js / notes.js (palette C comes from notes.js).
   Each figure is its own IIFE-style block wrapped in NV.safe(), so one failure cannot blank the rest.
   House rule: every number a figure displays is computed here from the spec arrays below. */

const NV = (function(){
  const P = Object.assign({}, (typeof C !== "undefined" ? C : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"}),
    { panel:"#15181f", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6" });

  /* ── configuration spec. Values are from each checkpoint's config.json / the paper tables.
     arch: "enc" = encoder-only, "encdec" = encoder–decoder.
     L = layers per stack, d = hidden, f = FFN inner, h = heads, dkv = per-head dim (default d/h),
     V = vocab, E = embedding width (ALBERT/ELECTRA factorise it), P = learned absolute positions
     (rows, per stack), T = token-type rows, bias = linear/LN biases, gated = GLU-style FFN,
     pooler = [CLS] pooler, relRows = DeBERTa relative-position table rows, t5rel = T5 bias buckets,
     share = ALBERT cross-layer sharing, untied = separate output projection,
     embLN = LayerNorm on the embedding sum, finalNorm = final norm per stack,
     skipFirstNorm = ModernBERT's first block has no pre-attention norm.
     reported = the size label in the paper / model card; ckpt = total tensor count of the released
     checkpoint (includes pretraining heads), null where not published in that form. */
  const M = {
    "bert-base":     {name:"BERT-base",        fam:"enc",   L:12, d:768,  f:3072, h:12, V:30522,  P:512, T:2, bias:true, pooler:true, embLN:true, reported:110e6, repLabel:"110M", ckpt:110106428},
    "bert-large":    {name:"BERT-large",       fam:"enc",   L:24, d:1024, f:4096, h:16, V:30522,  P:512, T:2, bias:true, pooler:true, embLN:true, reported:340e6, repLabel:"340M", ckpt:336226108},
    "roberta-base":  {name:"RoBERTa-base",     fam:"enc",   L:12, d:768,  f:3072, h:12, V:50265,  P:514, T:1, bias:true, pooler:true, embLN:true, reported:125e6, repLabel:"125M", ckpt:124697433},
    "roberta-large": {name:"RoBERTa-large",    fam:"enc",   L:24, d:1024, f:4096, h:16, V:50265,  P:514, T:1, bias:true, pooler:true, embLN:true, reported:355e6, repLabel:"355M", ckpt:355412057},
    "albert-base":   {name:"ALBERT-base",      fam:"enc",   L:12, d:768,  f:3072, h:12, V:30000,  E:128, P:512, T:2, bias:true, pooler:true, embLN:true, share:true, reported:12e6, repLabel:"12M", ckpt:null},
    "electra-small": {name:"ELECTRA-Small",    fam:"enc",   L:12, d:256,  f:1024, h:4,  V:30522,  E:128, P:512, T:2, bias:true, embLN:true, reported:14e6, repLabel:"14M", ckpt:null},
    "electra-base":  {name:"ELECTRA-Base",     fam:"enc",   L:12, d:768,  f:3072, h:12, V:30522,  P:512, T:2, bias:true, embLN:true, reported:110e6, repLabel:"110M", ckpt:null},
    "electra-large": {name:"ELECTRA-Large",    fam:"enc",   L:24, d:1024, f:4096, h:16, V:30522,  P:512, T:2, bias:true, embLN:true, reported:335e6, repLabel:"335M", ckpt:null},
    "deberta-v3-base":{name:"DeBERTa-v3-base", fam:"enc",   L:12, d:768,  f:3072, h:12, V:128100, P:0, T:0, bias:true, embLN:true, relRows:512, reported:184e6, repLabel:"86M backbone + 98M embedding", ckpt:null},
    "deberta-v3-large":{name:"DeBERTa-v3-large",fam:"enc",  L:24, d:1024, f:4096, h:16, V:128100, P:0, T:0, bias:true, embLN:true, relRows:512, reported:435e6, repLabel:"304M backbone + 131M embedding", ckpt:null},
    "modernbert-base":{name:"ModernBERT-base", fam:"enc",   L:22, d:768,  f:1152, h:12, V:50368,  P:0, T:0, bias:false, gated:true, embLN:true, finalNorm:true, skipFirstNorm:true, reported:149e6, repLabel:"149M", ckpt:149655232},
    "modernbert-large":{name:"ModernBERT-large",fam:"enc",  L:28, d:1024, f:2624, h:16, V:50368,  P:0, T:0, bias:false, gated:true, embLN:true, finalNorm:true, skipFirstNorm:true, reported:395e6, repLabel:"395M", ckpt:395881664},
    "t5-small":      {name:"T5-Small",         fam:"encdec",L:6,  d:512,  f:2048, h:8,  dkv:64,  V:32128, P:0, bias:false, t5rel:32, finalNorm:true, reported:60e6,  repLabel:"60M",  ckpt:60506880},
    "t5-base":       {name:"T5-Base",          fam:"encdec",L:12, d:768,  f:3072, h:12, dkv:64,  V:32128, P:0, bias:false, t5rel:32, finalNorm:true, reported:220e6, repLabel:"220M", ckpt:222903936},
    "t5-large":      {name:"T5-Large",         fam:"encdec",L:24, d:1024, f:4096, h:16, dkv:64,  V:32128, P:0, bias:false, t5rel:32, finalNorm:true, reported:770e6, repLabel:"770M", ckpt:737668608},
    "t5-3b":         {name:"T5-3B",            fam:"encdec",L:24, d:1024, f:16384,h:32, dkv:128, V:32128, P:0, bias:false, t5rel:32, finalNorm:true, reported:2.8e9, repLabel:"≈2.8B", ckpt:2851599360},
    "t5-11b":        {name:"T5-11B",           fam:"encdec",L:24, d:1024, f:65536,h:128,dkv:128, V:32128, P:0, bias:false, t5rel:32, finalNorm:true, reported:11e9,  repLabel:"≈11B",  ckpt:null},
    "flan-t5-xxl":   {name:"Flan-T5-XXL",      fam:"encdec",L:24, d:4096, f:10240,h:64, dkv:64,  V:32128, P:0, bias:false, gated:true, untied:true, t5rel:32, finalNorm:true, reported:11e9, repLabel:"11B", ckpt:null},
    "mt5-xxl":       {name:"mT5-XXL",          fam:"encdec",L:24, d:4096, f:10240,h:64, dkv:64,  V:250112,P:0, bias:false, gated:true, untied:true, t5rel:32, finalNorm:true, reported:13e9, repLabel:"13B", ckpt:null},
    "bart-base":     {name:"BART-base",        fam:"encdec",L:6,  d:768,  f:3072, h:12, V:50265,  P:1026, bias:true, embLN:true, reported:140e6, repLabel:"140M", ckpt:139420416},
    "bart-large":    {name:"BART-large",       fam:"encdec",L:12, d:1024, f:4096, h:16, V:50265,  P:1026, bias:true, embLN:true, reported:400e6, repLabel:"400M", ckpt:null}
  };

  /* exact parameter count from a config; returns a breakdown */
  function params(c){
    const d = c.d, E = c.E || d, inner = (c.dkv ? c.dkv * c.h : d);
    const nb = w => c.bias ? 2*w : w;                       // a LayerNorm with bias has 2w, without w
    const attn = 4*d*inner + (c.bias ? 3*inner + d : 0);
    const ffn  = c.gated ? 3*d*c.f + (c.bias ? 2*c.f + d : 0) : 2*d*c.f + (c.bias ? c.f + d : 0);
    const encLayer = attn + ffn + 2*nb(d);
    const decLayer = 2*attn + ffn + 3*nb(d);
    const stacks = c.fam === "encdec" ? 2 : 1;
    const nEnc = c.share ? 1 : c.L;
    const tok = c.V * E;
    const pos = stacks * (c.P || 0) * E + (c.T || 0) * E;
    const embNorm = (c.embLN ? stacks * nb(E) : 0) + (E !== d ? E*d + d : 0);
    const enc = nEnc * encLayer - (c.skipFirstNorm ? nb(d) : 0);
    const dec = c.fam === "encdec" ? c.L * decLayer : 0;
    const other = (c.pooler ? d*d + d : 0) + (c.relRows ? c.relRows*d + nb(d) : 0) +
                  (c.t5rel ? stacks * c.t5rel * c.h : 0) + (c.finalNorm ? stacks * nb(d) : 0);
    const head = c.untied ? c.V * d : 0;
    const total = tok + pos + embNorm + enc + dec + other + head;
    return {tok, pos, embNorm, enc, dec, other, head, total, emb: tok + head, nonEmb: total - tok - head, encLayer, decLayer};
  }
  function fmtN(x){
    const a = Math.abs(x);
    if (a >= 1e12) return (x/1e12).toFixed(2)+"T";
    if (a >= 1e9)  return (x/1e9).toFixed(2)+"B";
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
  function safe(name, fn){ try { fn(); } catch(e){ if (typeof console!=="undefined") console.error("[nlp-models.viz] "+name+" failed:", e); } }
  function lcg(seed){ let s = seed >>> 0; return () => { s = (1664525*s + 1013904223) >>> 0; return s/4294967296; }; }
  const FAMC = {enc:P.A, encdec:P.B, dec:P.teal, rnn:P.muted};
  const FAMN = {enc:"encoder-only", encdec:"encoder–decoder", dec:"decoder-only", rnn:"biLSTM"};
  return {P, M, params, fmtN, comma, txt, safe, lcg, FAMC, FAMN};
})();

/* ───────────────────────── 03 · lineage timeline ───────────────────────── */
NV.safe("timeline", function(){
  const {P, M, params, fmtN, txt, FAMC, FAMN} = NV;
  const svg = d3.select("#tl-svg"), W = 640, H = 300, m = {l:58, r:20, t:24, b:40};
  /* date = year + (month−1)/12 from each paper's first arXiv version; cfg links to the spec above */
  const ev = [
    {name:"ELMo",       t:2018 + 1/12,  fam:"rnn",    n:null, what:"deep contextual word vectors from a two-layer bidirectional LSTM language model; features, not a fine-tuned network"},
    {name:"GPT-1",      t:2018 + 5/12,  fam:"dec",    n:117e6, what:"left-to-right Transformer decoder pretrained then fine-tuned whole; the contrast case for this hub"},
    {name:"BERT",       t:2018 + 9/12,  fam:"enc",    cfg:"bert-large", lo:"bert-base", what:"masked language modelling + next-sentence prediction on a bidirectional encoder; fine-tune a small head per task"},
    {name:"RoBERTa",    t:2019 + 6/12,  fam:"enc",    cfg:"roberta-large", lo:"roberta-base", what:"same architecture as BERT, NSP dropped, dynamic masking, byte-level BPE, 160GB of text and far longer training"},
    {name:"ALBERT",     t:2019 + 8/12,  fam:"enc",    n:235e6, lo:"albert-base", what:"factorised embeddings and cross-layer parameter sharing; sentence-order prediction replaces NSP"},
    {name:"T5",         t:2019 + 9/12,  fam:"encdec", cfg:"t5-11b", lo:"t5-small", what:"every task written as text-to-text; span corruption with sentinel tokens; C4 corpus; relative position biases"},
    {name:"BART",       t:2019 + 9.4/12,fam:"encdec", cfg:"bart-large", lo:"bart-base", what:"denoising autoencoder: text infilling with Poisson(3) spans plus sentence permutation, decoded by an autoregressive decoder"},
    {name:"ELECTRA",    t:2020 + 2/12,  fam:"enc",    cfg:"electra-large", lo:"electra-small", what:"replaced-token detection: a small generator corrupts, the discriminator classifies every position"},
    {name:"DeBERTa",    t:2020 + 5/12,  fam:"enc",    n:1.5e9, what:"disentangled content / relative-position attention and an enhanced mask decoder; scaled to 1.5B parameters"},
    {name:"mT5",        t:2020 + 9/12,  fam:"encdec", cfg:"mt5-xxl", lo:{L:8, d:512, f:1024, h:6, dkv:64, V:250112, fam:"encdec", bias:false, gated:true, untied:true, t5rel:32, finalNorm:true} /* mT5-Small config */, what:"T5 recipe on mC4 covering 101 languages; a 250k-entry SentencePiece vocabulary"},
    {name:"DeBERTaV3",  t:2021 + 10/12, fam:"enc",    cfg:"deberta-v3-large", lo:"deberta-v3-base", what:"ELECTRA-style replaced-token detection with gradient-disentangled embedding sharing; 128k vocabulary"},
    {name:"Flan-T5",    t:2022 + 9/12,  fam:"encdec", cfg:"flan-t5-xxl", lo:{L:8, d:512, f:1024, h:6, dkv:64, V:32128, fam:"encdec", bias:false, gated:true, untied:true, t5rel:32, finalNorm:true} /* Flan-T5-Small (T5 v1.1-Small) config */, what:"T5 v1.1-style checkpoints instruction-tuned on roughly 1.8k tasks; strong zero- and few-shot use"},
    {name:"ModernBERT", t:2024 + 11/12, fam:"enc",    cfg:"modernbert-large", lo:"modernbert-base", what:"RoPE, GeGLU, alternating local/global attention, unpadding, 8,192-token context, 2T tokens of text and code"}
  ];
  ev.forEach(e => {
    if (e.cfg) e.n = params(M[e.cfg]).total;
    if (typeof e.lo === "string") e.loN = params(M[e.lo]).total;
    else if (e.lo) e.loN = params(e.lo).total;
  });
  const x = d3.scaleLinear().domain([2017.8, 2025.3]).range([m.l, W-m.r]);
  const y = d3.scaleLog().domain([5e6, 3e10]).range([H-m.b, m.t]);
  let sel = 2, filt = "all";
  const fSel = document.getElementById("tl-filter");
  if (fSel) fSel.addEventListener("change", () => { filt = fSel.value; const vis = ev.map((e,i)=>i).filter(i=>show(ev[i])); if (vis.length && !show(ev[sel])) sel = vis[0]; draw(); });
  function show(e){ return filt === "all" || e.fam === filt; }
  function draw(){
    svg.selectAll("*").remove();
    const g = svg.append("g");
    [1e7,1e8,1e9,1e10].forEach(v=>{
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, m.l-8, y(v)+4, fmtN(v), {anchor:"end", size:10, fill:P.muted});
    });
    for (let yr = 2018; yr <= 2025; yr++){
      txt(g, x(yr), H-m.b+16, String(yr), {anchor:"middle", size:10, fill:P.muted});
      g.append("line").attr("x1",x(yr)).attr("x2",x(yr)).attr("y1",H-m.b).attr("y2",H-m.b+4).attr("stroke",P.muted);
    }
    txt(g, 8, m.t-8, "largest size (params, log)", {size:10, fill:P.muted});
    // legend
    let lx = W-m.r-360;
    ["enc","encdec","dec","rnn"].forEach(f=>{
      g.append("circle").attr("cx",lx).attr("cy",m.t-12).attr("r",4).attr("fill",FAMC[f]);
      txt(g, lx+7, m.t-8, FAMN[f], {size:10, fill:P.muted}); lx += FAMN[f].length*6 + 22;
    });
    ev.forEach((d,i)=>{
      if (!show(d)) return;
      const cx = x(d.t), on = i===sel, cy = d.n ? y(d.n) : H-m.b-10;
      if (d.loN) g.append("line").attr("x1",cx).attr("x2",cx).attr("y1",y(d.loN)).attr("y2",y(d.n)).attr("stroke",FAMC[d.fam]).attr("stroke-width",3).attr("stroke-opacity",on?.7:.3);
      const node = g.append("g").style("cursor","pointer").on("click",()=>{ sel=i; draw(); });
      if (d.n) node.append("circle").attr("cx",cx).attr("cy",cy).attr("r",on?8:5.5).attr("fill",FAMC[d.fam]).attr("stroke",on?P.ink:"none").attr("stroke-width",1.5);
      else node.append("rect").attr("x",cx-6).attr("y",cy-6).attr("width",12).attr("height",12).attr("fill","none").attr("stroke",on?P.ink:P.muted).attr("stroke-dasharray","3,2");
      const up = i % 2 === 0;
      txt(node, cx, cy + (up ? -11 : 19), d.name, {anchor:"middle", size:10.5, fill:on?P.ink:P.muted, bold:on});
    });
    const d = ev[sel];
    const yr = Math.floor(d.t), mo = Math.round((d.t - yr)*12) + 1;
    let size;
    if (d.cfg) size = `largest released size <b>${fmtN(d.n)}</b> recomputed from its configuration (label in the paper or card: ${M[d.cfg].repLabel})` + (d.loN ? `; smallest ${fmtN(d.loN)}, a ${(d.n/d.loN).toFixed(0)}× range` : "");
    else if (d.n) size = `largest reported size <b>${fmtN(d.n)}</b>` + (d.loN ? `; smallest ${fmtN(d.loN)} recomputed from its configuration` : "");
    else size = "not a Transformer; drawn at the floor as a dashed box rather than a size";
    const nShown = ev.filter(show).length;
    d3.select("#tl-read").html(`<b>${d.name}</b> · ${FAMN[d.fam]} · first posted ${yr}-${String(mo).padStart(2,"0")}: ${d.what}. ${size}. <br/>${nShown} of ${ev.length} milestones shown.`);
  }
  draw();
});

/* ───────────────────────── 04 · pretraining objectives on one sentence ───────────────────────── */
NV.safe("objectives", function(){
  const {P, txt, lcg} = NV;
  const svg = d3.select("#obj-svg"), W = 640, H = 300;
  const S1 = "we hide some words and the model has to guess them .".split(" ");
  const S2 = "it sees both sides of each gap .".split(" ");
  const toks = S1.concat(S2), n = toks.length;
  const ALT = {we:"they", hide:"mask", some:"many", words:"tokens", and:"so", the:"a", model:"net", has:"tries", to:"to", guess:"find", them:"those", it:"he", sees:"reads", both:"all", sides:"parts", of:"in", each:"any", gap:"hole", ".":"."};
  /* objective spec: default corruption rate, mean span, and the loss-position formula per length N */
  const OBJ = {
    mlm:  {name:"Masked LM (BERT)",                  rate:0.15, mean:1, perN:(N,r)=>r*N,              note:"loss only on the selected positions; 80% → [M], 10% → random word, 10% unchanged"},
    rtd:  {name:"Replaced-token detection (ELECTRA)",rate:0.15, mean:1, perN:(N,r)=>N,                note:"a small generator fills the masked slots; the discriminator labels every position original / replaced"},
    span: {name:"Span corruption (T5)",              rate:0.15, mean:3, perN:(N,r,mu)=>r*N + r*N/mu + 1, note:"each corrupted span becomes one sentinel; the decoder emits sentinels + the dropped tokens only"},
    bart: {name:"Text infilling + sentence permutation (BART)", rate:0.30, mean:3, perN:(N,r)=>N, note:"Poisson(3) spans each collapse to one [M]; sentences are shuffled; the decoder rebuilds the whole original"},
    clm:  {name:"Causal LM (GPT, for contrast)",     rate:0,    mean:1, perN:(N,r)=>N,                note:"no corruption; every position predicts the next token from the left context only"}
  };
  const modeSel = document.getElementById("obj-mode"), rateR = document.getElementById("obj-rate"), rateV = document.getElementById("obj-ratev");
  const NR = document.getElementById("obj-N"), NV_ = document.getElementById("obj-Nv"), btn = document.getElementById("obj-resample");
  let mode = "mlm", seed = 7, rate = 0.15;
  if (modeSel) modeSel.addEventListener("change", () => { mode = modeSel.value; rate = OBJ[mode].rate; if (rateR) rateR.value = Math.round(rate*100); draw(); });
  if (rateR) rateR.addEventListener("input", () => { rate = (+rateR.value)/100; draw(); });
  if (NR) NR.addEventListener("input", draw);
  if (btn) btn.addEventListener("click", () => { seed = (seed * 31 + 11) % 997; draw(); });

  function poisson(rng, lam){ const L = Math.exp(-lam); let k = 0, p = 1; do { k++; p *= rng(); } while (p > L); return k - 1; }
  /* choose spans covering ≈ rate·n tokens; mean span length mu (1 = independent tokens) */
  function pickSpans(rng, r, mu, poissonLen){
    const target = Math.max(r > 0 ? 1 : 0, Math.round(r*n));
    const hit = new Array(n).fill(false); let covered = 0, guard = 0; const spans = [];
    while (covered < target && guard++ < 200){
      let len = poissonLen ? poisson(rng, mu) : (mu === 1 ? 1 : Math.max(1, Math.round(mu + (rng()-0.5)*2)));
      if (len === 0) len = 1;                      // a zero-length span inserts a mask; keep the demo readable
      len = Math.min(len, target - covered);
      const st = Math.floor(rng()*(n - len + 1));
      let ok = true; for (let k = Math.max(0,st-1); k < Math.min(n, st+len+1); k++) if (hit[k]) ok = false;
      if (!ok) continue;
      for (let k = st; k < st+len; k++) hit[k] = true;
      spans.push([st, st+len]); covered += len;
    }
    spans.sort((a,b)=>a[0]-b[0]);
    return {hit, spans, covered};
  }
  function build(){
    const rng = lcg(seed*101 + mode.length*7);
    const o = OBJ[mode];
    let input = [], target = [], note = "";
    if (mode === "mlm"){
      const {hit, covered} = pickSpans(rng, rate, 1, false);
      toks.forEach((t,i)=>{
        if (!hit[i]){ input.push({t, k:"plain"}); target.push({t:"·", k:"none"}); return; }
        const u = rng(); const shown = u < 0.8 ? "[M]" : (u < 0.9 ? (ALT[t]||t) : t);
        input.push({t:shown, k:"corrupt"}); target.push({t, k:"loss"});
      });
      note = `${covered} of ${n} positions selected`;
    } else if (mode === "rtd"){
      const {hit} = pickSpans(rng, rate, 1, false);
      let rep = 0;
      toks.forEach((t,i)=>{
        if (hit[i] && rng() < 0.75){ input.push({t:ALT[t]||t, k:"corrupt"}); target.push({t:"rep", k:"loss"}); rep++; }
        else { input.push({t, k: hit[i] ? "same" : "plain"}); target.push({t:"orig", k:"loss"}); }
      });
      note = `${rep} replaced by the generator; the rest (including masked slots it guessed correctly) are labelled original`;
    } else if (mode === "span" || mode === "bart"){
      const {hit, spans, covered} = pickSpans(rng, rate, OBJ[mode].mean, mode === "bart");
      const sent = ["<X>","<Y>","<Z>","<W>","<V>","<U>"];
      let order = [[0, S1.length],[S1.length, n]];
      if (mode === "bart") order = [order[1], order[0]];          // sentence permutation: two sentences, swapped
      let si = 0;
      order.forEach(([a,b])=>{
        for (let i = a; i < b; i++){
          const sp = spans.find(s=>s[0]===i);
          if (sp){ input.push({t: mode === "span" ? sent[si % sent.length] : "[M]", k:"corrupt"}); if (mode === "span"){ target.push({t:sent[si % sent.length], k:"loss"}); for (let k = sp[0]; k < sp[1]; k++) target.push({t:toks[k], k:"loss"}); } si++; i = sp[1]-1; }
          else input.push({t:toks[i], k:"plain"});
        }
      });
      if (mode === "span") target.push({t:sent[si % sent.length], k:"loss"});
      else toks.forEach(t=>target.push({t, k:"loss"}));
      note = `${spans.length} span${spans.length===1?"":"s"} covering ${covered} of ${n} tokens` + (mode === "bart" ? "; the two sentences are shown swapped" : "");
    } else {
      input.push({t:"<s>", k:"plain"}); toks.slice(0,-1).forEach(t=>input.push({t, k:"plain"}));
      toks.forEach(t=>target.push({t, k:"loss"}));
      note = "input is the sequence shifted right by one";
    }
    return {input, target, note};
  }
  function row(g, y, label, arr, cw){
    txt(g, 10, y-8, label, {size:10.5, fill:P.muted});
    arr.forEach((c,i)=>{
      const x0 = 10 + i*cw;
      const fill = c.k === "loss" ? P.good : (c.k === "corrupt" ? P.B : (c.k === "same" ? P.purple : "none"));
      g.append("rect").attr("x",x0).attr("y",y).attr("width",cw-2).attr("height",24).attr("rx",3)
        .attr("fill",fill).attr("fill-opacity",c.k === "none" || c.k === "plain" ? 0 : .22).attr("stroke",c.k === "none" ? P.line : (fill === "none" ? P.muted : fill)).attr("stroke-opacity",c.k==="none"?.6:.8);
      txt(g, x0+(cw-2)/2, y+16, c.t, {anchor:"middle", size: c.t.length > 5 ? 8 : 9.5, fill: c.k==="none"?P.line:P.ink, mono:true});
    });
  }
  function draw(){
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const o = OBJ[mode], {input, target, note} = build();
    const cols = Math.max(n, input.length, target.length), cw = Math.min(31, (W-20)/cols);
    row(g, 30, "original text", toks.map(t=>({t, k:"plain"})), cw);
    row(g, 110, mode === "rtd" ? "discriminator input (generator output)" : (mode === "clm" ? "decoder input" : (mode === "span" || mode === "bart" ? "encoder input" : "encoder input")), input, cw);
    row(g, 190, mode === "rtd" ? "per-position label (loss everywhere)" : (mode === "span" || mode === "bart" ? "decoder target (loss on every target token)" : (mode === "clm" ? "target: next token at every position" : "prediction targets (loss only here)")), target, cw);
    const lossK = target.filter(c=>c.k==="loss").length;
    // legend
    const lg = [["loss position",P.good],["corrupted",P.B],["masked but unchanged",P.purple]];
    let lx = 10; lg.forEach(([s,c])=>{ g.append("rect").attr("x",lx).attr("y",H-34).attr("width",12).attr("height",12).attr("fill",c).attr("fill-opacity",.3).attr("stroke",c); txt(g, lx+17, H-24, s, {size:10, fill:P.muted}); lx += s.length*6 + 36; });
    txt(g, 10, H-6, `sentence: ${n} tokens · encoder/decoder input ${input.length} · target ${target.length} · positions with a loss ${lossK}`, {size:10, fill:P.muted, mono:true});
    const N = NR ? +NR.value : 512;
    if (NV_) NV_.textContent = String(N);
    if (rateV) rateV.textContent = Math.round(rate*100) + "%";
    const perN = o.perN(N, rate, o.mean);
    d3.select("#obj-read").html(`<b>${o.name}</b>: ${o.note}. Here: ${note}; <b>${lossK}</b> loss positions out of ${n} tokens (${(100*lossK/n).toFixed(0)}%). ` +
      `At a ${N}-token sequence and ${Math.round(rate*100)}% corruption the expected number of loss positions is <b>${perN.toFixed(1)}</b> (${(100*perN/N).toFixed(1)}% of N).`);
  }
  draw();
});

/* ───────────────────────── 05 · attention visibility for four wirings ───────────────────────── */
NV.safe("masks", function(){
  const {P, txt} = NV;
  const svg = d3.select("#mk-svg"), W = 640, H = 330;
  const wSel = document.getElementById("mk-wiring"), nR = document.getElementById("mk-n"), nV = document.getElementById("mk-nv"), sR = document.getElementById("mk-s"), sV = document.getElementById("mk-sv");
  const WIRE = {
    enc:    {name:"encoder-only (BERT)",      allow:(i,j,s)=>true, params:"one stack"},
    dec:    {name:"decoder-only (GPT)",       allow:(i,j,s)=>j<=i, params:"one stack"},
    prefix: {name:"prefix LM (UniLM-style)",  allow:(i,j,s)=> j < s || j <= i, params:"one stack, shared by source and target"},
    encdec: {name:"encoder–decoder (T5, BART)",allow:(i,j,s)=> (i < s ? j < s : (j < s || j <= i)), params:"two stacks: encoder self-attn, decoder self-attn and cross-attn"}
  };
  let wiring = "encdec";
  [wSel, nR, sR].forEach(el => { if (el) el.addEventListener(el.tagName === "SELECT" ? "change" : "input", () => { if (el === wSel) wiring = wSel.value; draw(); }); });
  function draw(){
    svg.selectAll("*").remove();
    const n = nR ? +nR.value : 8;
    let s = sR ? +sR.value : 4; if (s > n-1) s = n-1; if (s < 1) s = 1;
    if (nV) nV.textContent = String(n); if (sV) sV.textContent = String(s);
    const w = WIRE[wiring], usesS = wiring === "prefix" || wiring === "encdec";
    const cell = Math.min(28, 250/n), x0 = 70, y0 = 40, g = svg.append("g");
    txt(g, x0 + n*cell/2, 20, "keys (positions attended to) →", {anchor:"middle", size:10, fill:P.muted});
    txt(g, 12, y0 + n*cell/2, "queries", {size:10, fill:P.muted});
    let allowed = 0, srcSrc = 0, tgtSrc = 0, tgtTgt = 0;
    for (let i = 0; i < n; i++){
      for (let j = 0; j < n; j++){
        const ok = w.allow(i,j,s);
        if (ok){ allowed++; if (usesS){ if (i < s && j < s) srcSrc++; else if (i >= s && j < s) tgtSrc++; else if (i >= s && j >= s) tgtTgt++; } }
        const src = usesS && j < s, cross = wiring === "encdec" && i >= s && j < s;
        g.append("rect").attr("x", x0 + j*cell).attr("y", y0 + i*cell).attr("width", cell-1.5).attr("height", cell-1.5).attr("rx",2)
          .attr("fill", ok ? (cross ? P.B : (src ? P.purple : P.A)) : "none").attr("fill-opacity", ok ? .75 : 0)
          .attr("stroke", ok ? "none" : P.line);
      }
      const lab = usesS ? (i < s ? "s"+(i+1) : "t"+(i-s+1)) : "x"+(i+1);
      txt(g, x0-6, y0 + i*cell + cell*0.65, lab, {anchor:"end", size:9, fill:P.muted, mono:true});
      txt(g, x0 + i*cell + cell/2, y0 + n*cell + 12, lab, {anchor:"middle", size:9, fill:P.muted, mono:true});
    }
    if (usesS){
      g.append("line").attr("x1",x0+s*cell-0.75).attr("x2",x0+s*cell-0.75).attr("y1",y0-4).attr("y2",y0+n*cell+2).attr("stroke",P.ink).attr("stroke-dasharray","3,2");
      g.append("line").attr("x1",x0-4).attr("x2",x0+n*cell).attr("y1",y0+s*cell-0.75).attr("y2",y0+s*cell-0.75).attr("stroke",P.ink).attr("stroke-dasharray","3,2");
    }
    // side panel
    const px = x0 + n*cell + 40, total = n*n;
    txt(g, px, 50, w.name, {size:12, bold:true});
    txt(g, px, 72, "parameters: " + w.params, {size:10.5, fill:P.muted});
    txt(g, px, 100, `allowed pairs: ${allowed} of ${total} (${(100*allowed/total).toFixed(1)}%)`, {size:11, mono:true});
    const leg = wiring === "encdec" ? [["encoder self-attention (source↔source)",P.purple],["decoder cross-attention (target→source)",P.B],["decoder causal self-attention",P.A]]
              : wiring === "prefix" ? [["bidirectional prefix columns",P.purple],["causal part",P.A]]
              : [["allowed",P.A]];
    leg.forEach(([t,c],k)=>{ g.append("rect").attr("x",px).attr("y",120+k*20).attr("width",12).attr("height",12).attr("fill",c).attr("fill-opacity",.75); txt(g, px+18, 130+k*20, t, {size:10.5, fill:P.muted}); });
    if (usesS){
      txt(g, px, 200, `source length s = ${s}, target length t = ${n-s}`, {size:10.5, fill:P.muted, mono:true});
      txt(g, px, 218, `s² = ${srcSrc} · t·s = ${tgtSrc} · t(t+1)/2 = ${tgtTgt}`, {size:10.5, fill:P.muted, mono:true});
    }
    let msg;
    if (wiring === "enc") msg = `Every token sees all ${n}: ${allowed} = n² pairs. Good for representations, but the model cannot be sampled left to right without a separate decoder.`;
    else if (wiring === "dec") msg = `Causal: ${allowed} = n(n+1)/2 pairs. Row i sees only positions ≤ i, so the same pass trains ${n} next-token predictions.`;
    else {
      const other = wiring === "prefix" ? "encoder–decoder" : "prefix LM";
      msg = `${allowed} allowed pairs = s² + t·s + t(t+1)/2 = ${s*s} + ${(n-s)*s} + ${(n-s)*(n-s+1)/2}. The ${other} wiring has exactly the same visibility pattern; the difference is whether source and target share one set of weights (prefix LM) or use separate stacks joined by cross-attention (encoder–decoder).`;
    }
    d3.select("#mk-read").html(`<b>${w.name}</b>: ${msg}`);
  }
  draw();
});

/* ───────────────────────── 07 · parameter-count explorer ───────────────────────── */
NV.safe("params", function(){
  const {P, M, params, fmtN, comma, txt, FAMC} = NV;
  const svg = d3.select("#pc-svg"), W = 640, H = 380;
  const pre = document.getElementById("pc-preset"), LR = document.getElementById("pc-L"), LV = document.getElementById("pc-Lv"),
        VR = document.getElementById("pc-V"), VV = document.getElementById("pc-Vv"), reset = document.getElementById("pc-reset");
  let key = "bert-base", cur = Object.assign({}, M[key]);
  function load(k){ key = k; cur = Object.assign({}, M[k]); if (LR) LR.value = cur.L; if (VR) VR.value = cur.V; }
  if (pre) pre.addEventListener("change", () => { load(pre.value); draw(); });
  if (LR) LR.addEventListener("input", () => { cur.L = +LR.value; draw(); });
  if (VR) VR.addEventListener("input", () => { cur.V = +VR.value; draw(); });
  if (reset) reset.addEventListener("click", () => { load(key); draw(); });
  const PARTS = [
    ["tok","token embeddings",P.B],["head","untied output projection",P.pink],["pos","position / type embeddings",P.purple],["embNorm","embedding norms + projection",P.muted],
    ["enc","encoder blocks",P.A],["dec","decoder blocks (self + cross + FFN)",P.teal],["other","pooler, rel-pos tables, final norms",P.bad]
  ];
  function draw(){
    svg.selectAll("*").remove();
    const g = svg.append("g"), r = params(cur);
    if (LV) LV.textContent = String(cur.L); if (VV) VV.textContent = comma(cur.V);
    // stacked bar of the selected config
    const bx = 20, bw = W-40, by = 34, bh = 30;
    txt(g, bx, 22, `${cur.name}${edited() ? " (edited)" : ""} — ${fmtN(r.total)} parameters`, {size:12, bold:true});
    let acc = 0;
    PARTS.forEach(([k,lab,col])=>{
      const v = r[k]; if (v <= 0) return;
      const w = bw * v / r.total;
      g.append("rect").attr("x", bx + bw*acc/r.total).attr("y", by).attr("width", Math.max(0.5,w)).attr("height", bh).attr("fill", col).attr("fill-opacity", .8);
      if (w > 60) txt(g, bx + bw*acc/r.total + w/2, by+19, `${(100*v/r.total).toFixed(1)}%`, {anchor:"middle", size:10, fill:"#0f1117", bold:true});
      acc += v;
    });
    let lx = bx, ly = by + bh + 18;
    PARTS.forEach(([k,lab,col])=>{ if (r[k] <= 0) return; const s = `${lab} ${fmtN(r[k])}`; if (lx + s.length*5.8 > W-20){ lx = bx; ly += 16; }
      g.append("rect").attr("x",lx).attr("y",ly-9).attr("width",10).attr("height",10).attr("fill",col); txt(g, lx+14, ly, s, {size:10, fill:P.muted}); lx += s.length*5.8 + 26; });
    // all presets: total vs embedding share
    const keys = Object.keys(M), rows = keys.map(k=>({k, m:M[k], r:params(M[k])}));
    const top = ly + 26, rowH = (H - top - 8) / rows.length;
    const xs = d3.scaleLog().domain([5e6, 2e10]).range([150, W-80]);
    txt(g, 20, top-8, "every preset, recomputed (bar = total, darker part = token embeddings + output projection; log scale)", {size:10, fill:P.muted});
    rows.forEach((o,i)=>{
      const yy = top + i*rowH, on = o.k === key;
      txt(g, 144, yy + rowH*0.72, o.m.name, {anchor:"end", size: Math.min(10, rowH*0.8), fill: on ? P.ink : P.muted, bold:on});
      const x1 = xs(o.r.total), xe = xs(5e6) + (x1 - xs(5e6)) * (o.r.emb / o.r.total);
      g.append("rect").attr("x", xs(5e6)).attr("y", yy+1).attr("width", x1 - xs(5e6)).attr("height", Math.max(2,rowH-3)).attr("fill", FAMC[o.m.fam]).attr("fill-opacity", on ? .55 : .25);
      g.append("rect").attr("x", xs(5e6)).attr("y", yy+1).attr("width", Math.max(0, xe - xs(5e6))).attr("height", Math.max(2,rowH-3)).attr("fill", FAMC[o.m.fam]).attr("fill-opacity", on ? .95 : .55);
      txt(g, x1 + 4, yy + rowH*0.72, `${fmtN(o.r.total)} · ${(100*o.r.emb/o.r.total).toFixed(0)}% emb`, {size: Math.min(9.5, rowH*0.75), fill:P.muted, mono:true});
      g.append("rect").attr("x",0).attr("y",yy).attr("width",W).attr("height",rowH).attr("fill","transparent").style("cursor","pointer")
        .on("click", () => { load(o.k); if (pre) pre.value = o.k; draw(); });
    });
    const base = M[key];
    let cmp = `reported size <b>${base.repLabel}</b>`;
    if (!edited() && base.ckpt) {
      const diff = base.ckpt - r.total;
      cmp += `; the released checkpoint stores ${comma(base.ckpt)} parameters, ${diff >= 0 ? "+" : "−"}${comma(Math.abs(diff))} vs this backbone count (${(100*Math.abs(diff)/base.ckpt).toFixed(2)}%: pretraining heads, or an unused bias table)`;
    }
    if (!edited() && base.reported) cmp += `; formula ÷ reported label = ${(r.total/base.reported).toFixed(3)}`;
    const perLayer = cur.fam === "encdec" ? `encoder block ${fmtN(r.encLayer)}, decoder block ${fmtN(r.decLayer)}` : `block ${fmtN(r.encLayer)}${cur.share ? " (one set shared by all " + cur.L + " layers)" : ""}`;
    d3.select("#pc-read").html(`<b>${cur.name}</b>: total <b>${comma(r.total)}</b> = embeddings ${fmtN(r.emb)} (${(100*r.emb/r.total).toFixed(1)}%) + everything else ${fmtN(r.nonEmb)}. Per ${perLayer}. ${edited() ? "Edited configuration — compare with the preset by pressing reset." : cmp}.`);
  }
  function edited(){ const b = M[key]; return cur.L !== b.L || cur.V !== b.V; }
  draw();
});

/* ───────────────────────── 10 · cost of one answer ───────────────────────── */
NV.safe("cost", function(){
  const {P, M, params, fmtN, txt, FAMC} = NV;
  const svg = d3.select("#cc-svg"), W = 640, H = 250;
  /* decoder-only comparison points: GPT configs (see gpt.html), 12·L·d² non-embedding, tied V·d output */
  const DEC = {
    "gpt2-xl": {name:"GPT-2 XL", L:48, d:1600,  V:50257},
    "gpt3":    {name:"GPT-3 175B", L:96, d:12288, V:50257}
  };
  const SETS = {
    small: {enc:"bert-base",        s2s:"t5-base",     dec:"gpt2-xl"},
    mid:   {enc:"deberta-v3-large", s2s:"bart-large",  dec:"gpt2-xl"},
    large: {enc:"modernbert-large", s2s:"flan-t5-xxl", dec:"gpt3"}
  };
  const setSel = document.getElementById("cc-set"), inR = document.getElementById("cc-in"), inV = document.getElementById("cc-inv"),
        outR = document.getElementById("cc-out"), outV = document.getElementById("cc-outv");
  [setSel, inR, outR].forEach(el => { if (el) el.addEventListener(el.tagName === "SELECT" ? "change" : "input", draw); });
  /* forward FLOPs ≈ 2 × (weights touched per token) × tokens, attention-score term ignored */
  function cost(set, nin, nout){
    const e = M[set.enc], er = params(e);
    const encF = 2 * er.nonEmb * nin;
    const s = M[set.s2s], sr = params(s);
    const encPart = sr.enc + sr.other/2, decPart = sr.dec + sr.other/2, logits = s.V * s.d;
    const s2sF = 2 * encPart * nin + 2 * (decPart + logits) * nout;
    const g = DEC[set.dec], gNon = 12 * g.L * g.d * g.d;
    const decF = 2 * gNon * (nin + nout) + 2 * g.V * g.d * nout;
    return [
      {k:"enc", name:e.name + " (label / tags / span)", f:encF, note:`one pass over ${nin} tokens`},
      {k:"encdec", name:s.name + " (write " + nout + " tokens)", f:s2sF, note:`encoder over ${nin}, decoder over ${nout}`},
      {k:"dec", name:g.name + " (prompt + " + nout + " tokens)", f:decF, note:`${nin + nout} tokens through one stack`}
    ];
  }
  function draw(){
    svg.selectAll("*").remove();
    const set = SETS[setSel ? setSel.value : "small"], nin = inR ? +inR.value : 256, nout = outR ? +outR.value : 32;
    if (inV) inV.textContent = String(nin); if (outV) outV.textContent = String(nout);
    const rows = cost(set, nin, nout), g = svg.append("g");
    const x = d3.scaleLog().domain([1e9, 1e15]).range([200, W-70]);
    [1e9,1e10,1e11,1e12,1e13,1e14,1e15].forEach(v=>{
      g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",20).attr("y2",H-40).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, x(v), H-26, "10" + ["⁹","¹⁰","¹¹","¹²","¹³","¹⁴","¹⁵"][Math.round(Math.log10(v))-9], {anchor:"middle", size:10, fill:P.muted});
    });
    txt(g, (200+W-70)/2, H-8, "forward FLOPs for one request (log scale)", {anchor:"middle", size:10, fill:P.muted});
    rows.forEach((r,i)=>{
      const yy = 34 + i*60;
      txt(g, 192, yy+16, r.name, {anchor:"end", size:10.5});
      txt(g, 192, yy+31, r.note, {anchor:"end", size:9.5, fill:P.muted});
      g.append("rect").attr("x",200).attr("y",yy).attr("width",Math.max(1, x(Math.max(r.f,1e9)) - 200)).attr("height",34).attr("fill",FAMC[r.k]).attr("fill-opacity",.75);
      txt(g, x(Math.max(r.f,1e9)) + 5, yy+22, fmtN(r.f).replace("T"," T").replace("B"," G") + "FLOP", {size:10, fill:P.ink, mono:true});
    });
    const base = rows[0].f;
    d3.select("#cc-read").html(`input ${nin} tokens, output ${nout} tokens: encoder <b>${fmtN(rows[0].f)}</b> FLOPs · encoder–decoder <b>${fmtN(rows[1].f)}</b> (${(rows[1].f/base).toFixed(1)}× the encoder) · decoder-only <b>${fmtN(rows[2].f)}</b> (${(rows[2].f/base).toFixed(0)}× the encoder). ` +
      `Units: B = 10⁹, T = 10¹² FLOPs; attention-score FLOPs, which grow with context length, are ignored (a KV cache is assumed), so long inputs are slightly understated.`);
  }
  draw();
});
