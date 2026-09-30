/* llm-families.viz.js — every interactive figure on models/llm-families.html.
   Loaded after data.js / notes.js (palette C comes from notes.js).
   Each figure is its own IIFE (via LF.safe) wrapped in try/catch so one failure cannot blank the rest.
   House rule: every number a figure displays is computed here from the spec array M, never typed into a label. */

const LF = (function(){
  const P = Object.assign({}, (typeof C !== "undefined" ? C : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"}),
    { panel:"#15181f", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6" });
  const FAM = { GPT:"#5b9cff", Llama:"#f472b6", Mistral:"#ffb454", Qwen:"#c084fc", Gemma:"#4ade80", DeepSeek:"#2dd4bf" };

  /* ── the spec: published configurations (config.json / technical reports) ──
     att: "mha" | "gqa" | "mla";  win: sliding window;  pat: "all" local, "alt" 1:1, "g6" five local then one global
     E/k/fe/shared/kdense: MoE;  tokens: reported training tokens (null = undisclosed)
     rep/repA: report's round numbers;  t: arXiv report date (decimal year; GPT-2 has no arXiv version, so its release month)
     Gemma 1/2 V = 256,128 follows the reports' tables (the Hugging Face configs list 256,000). */
  const M = [
    {id:"gpt2",  name:"GPT-2 XL",        fam:"GPT",   t:2019.12, mon:"Feb 2019", gpt:true, L:48, d:1600, h:25, kv:25, hd:64, f:6400, V:50257, n:1024, tied:true,
     pos:"learned", norm:"pre-LN", act:"GELU", ctx:1024, tokens:null, rep:1.5e9, repLabel:"1.5B"},
    {id:"gpt3",  name:"GPT-3 175B",      fam:"GPT",   t:2020.37, mon:"May 2020", gpt:true, L:96, d:12288, h:96, kv:96, hd:128, f:49152, V:50257, n:2048, tied:true,
     pos:"learned", norm:"pre-LN", act:"GELU", ctx:2048, tokens:300e9, rep:175e9, repLabel:"175B"},
    {id:"llama1",name:"LLaMA 65B",       fam:"Llama", t:2023.15, mon:"Feb 2023", L:80, d:8192, h:64, kv:64, hd:128, f:22016, V:32000,
     pos:"RoPE 10k", norm:"RMSNorm pre", act:"SwiGLU", ctx:2048, tokens:1.4e12, rep:65.2e9, repLabel:"65.2B"},
    {id:"llama2",name:"Llama 2 70B",     fam:"Llama", t:2023.54, mon:"Jul 2023", L:80, d:8192, h:64, kv:8, hd:128, f:28672, V:32000,
     pos:"RoPE 10k", norm:"RMSNorm pre", act:"SwiGLU", ctx:4096, tokens:2.0e12, rep:70e9, repLabel:"70B"},
    {id:"llama3",name:"Llama 3 8B",      fam:"Llama", t:2024.30, mon:"Apr 2024", L:32, d:4096, h:32, kv:8, hd:128, f:14336, V:128256, noTimeline:true,
     pos:"RoPE 500k", norm:"RMSNorm pre", act:"SwiGLU", ctx:8192, tokens:15e12, rep:8e9, repLabel:"8B"},
    {id:"llama31",name:"Llama 3.1 405B", fam:"Llama", t:2024.58, mon:"Jul 2024", L:126, d:16384, h:128, kv:8, hd:128, f:53248, V:128256,
     pos:"RoPE 500k", norm:"RMSNorm pre", act:"SwiGLU", ctx:131072, tokens:15.6e12, rep:405e9, repLabel:"405B"},
    {id:"mistral",name:"Mistral 7B",     fam:"Mistral", t:2023.78, mon:"Oct 2023", L:32, d:4096, h:32, kv:8, hd:128, f:14336, V:32000, win:4096, pat:"all",
     pos:"RoPE 10k", norm:"RMSNorm pre", act:"SwiGLU", ctx:8192, tokens:null, rep:7e9, repLabel:"7B"},
    {id:"mixtral",name:"Mixtral 8x7B",   fam:"Mistral", t:2024.02, mon:"Jan 2024", L:32, d:4096, h:32, kv:8, hd:128, f:14336, V:32000, E:8, k:2, fe:14336,
     pos:"RoPE 1M", norm:"RMSNorm pre", act:"SwiGLU", ctx:32768, tokens:null, rep:47e9, repA:13e9, repLabel:"47B / 13B"},
    {id:"gemma1",name:"Gemma 7B",        fam:"Gemma", t:2024.20, mon:"Mar 2024", L:28, d:3072, h:16, kv:16, hd:256, f:24576, V:256128, tied:true,
     pos:"RoPE 10k", norm:"RMSNorm pre", act:"GeGLU", ctx:8192, tokens:6e12, rep:8.538e9, repLabel:"7B (report table: 8.54B)"},
    {id:"dsv2",  name:"DeepSeek-V2",     fam:"DeepSeek", t:2024.35, mon:"May 2024", L:60, d:5120, h:128, att:"mla", f:12288, V:102400, E:160, k:6, fe:1536, shared:2, kdense:1,
     mla:{qr:1536, kvr:512, rope:64, nope:128, v:128}, pos:"RoPE 10k", norm:"RMSNorm pre", act:"SwiGLU", ctx:131072, tokens:8.1e12, rep:236e9, repA:21e9, repLabel:"236B / 21B"},
    {id:"qwen2", name:"Qwen2 72B",       fam:"Qwen", t:2024.54, mon:"Jul 2024", L:80, d:8192, h:64, kv:8, hd:128, f:29568, V:152064, bias:true,
     pos:"RoPE 1M", norm:"RMSNorm pre", act:"SwiGLU", ctx:32768, tokens:7e12, rep:72e9, repLabel:"72B"},
    {id:"gemma2",name:"Gemma 2 27B",     fam:"Gemma", t:2024.58, mon:"Jul 2024", L:46, d:4608, h:32, kv:16, hd:128, f:36864, V:256128, tied:true, win:4096, pat:"alt",
     pos:"RoPE 10k", norm:"RMSNorm pre+post", act:"GeGLU", ctx:8192, tokens:13e12, rep:27.227e9, repLabel:"27B (report table: 27.23B)"},
    {id:"qwen25",name:"Qwen2.5 72B",     fam:"Qwen", t:2024.97, mon:"Dec 2024", L:80, d:8192, h:64, kv:8, hd:128, f:29568, V:152064, bias:true,
     pos:"RoPE 1M", norm:"RMSNorm pre", act:"SwiGLU", ctx:131072, tokens:18e12, rep:72e9, repLabel:"72B"},
    {id:"dsv3",  name:"DeepSeek-V3",     fam:"DeepSeek", t:2024.99, mon:"Dec 2024", L:61, d:7168, h:128, att:"mla", f:18432, V:129280, E:256, k:8, fe:2048, shared:1, kdense:3,
     mla:{qr:1536, kvr:512, rope:64, nope:128, v:128}, pos:"RoPE 10k", norm:"RMSNorm pre", act:"SwiGLU", ctx:131072, tokens:14.8e12, rep:671e9, repA:37e9, repLabel:"671B / 37B"},
    {id:"gemma3",name:"Gemma 3 27B",     fam:"Gemma", t:2025.20, mon:"Mar 2025", L:62, d:5376, h:32, kv:16, hd:128, f:21504, V:262208, tied:true, win:1024, pat:"g6",
     pos:"RoPE 1M / 10k", norm:"RMSNorm pre+post", act:"GeGLU", ctx:131072, tokens:14e12, rep:27.016e9, repLabel:"27B (report table: 27.0B text)"},
    {id:"qwen3d",name:"Qwen3 32B",       fam:"Qwen", t:2025.37, mon:"May 2025", L:64, d:5120, h:64, kv:8, hd:128, f:25600, V:151936, noTimeline:true,
     pos:"RoPE 1M", norm:"RMSNorm pre", act:"SwiGLU", ctx:32768, tokens:36e12, rep:32.8e9, repLabel:"32.8B"},
    {id:"qwen3m",name:"Qwen3 235B-A22B", fam:"Qwen", t:2025.37, mon:"May 2025", L:94, d:4096, h:64, kv:4, hd:128, f:12288, V:151936, E:128, k:8, fe:1536,
     pos:"RoPE 1M", norm:"RMSNorm pre", act:"SwiGLU", ctx:32768, tokens:36e12, rep:235e9, repA:22e9, repLabel:"235B / 22B"}
  ];
  const byId = {}; M.forEach(m => byId[m.id] = m);

  /* ── parameter accounting ── */
  function attnParams(m, kvOverride){
    const d = m.d, h = m.h;
    if (m.att === "mla" && !kvOverride){
      const a = m.mla, qk = a.nope + a.rope;
      const q = a.qr ? d*a.qr + a.qr + a.qr*h*qk : d*h*qk;
      const kv = d*(a.kvr + a.rope) + a.kvr + a.kvr*h*(a.nope + a.v);
      return q + kv + h*a.v*d;
    }
    const hd = m.hd || (m.mla ? m.mla.v : d/h);
    const kv = kvOverride === "mha" ? h : kvOverride === "mqa" ? 1 : (m.kv || h);
    return d*h*hd + 2*d*kv*hd + h*hd*d + (m.bias ? h*hd + 2*kv*hd : 0);
  }
  function params(m, opt){
    opt = opt || {};
    const d = m.d, L = m.L, V = m.V;
    if (m.gpt){
      const attn = L*(4*d*d + 4*d), ffn = L*(8*d*d + 5*d), ln = L*4*d + 2*d, emb = V*d + m.n*d;
      const total = attn + ffn + ln + emb;
      return {emb, attn, dense:ffn, expT:0, expA:0, norms:ln, total, active:total};
    }
    const emb = V*d*(m.tied ? 1 : 2);
    const aL = attnParams(m, opt.kv && opt.kv !== "cfg" ? opt.kv : null);
    const attn = L*aL, norms = L*2*d + d;
    let dense = 0, expT = 0, expA = 0;
    const k = m.E ? Math.min(m.E, opt.k || m.k) : 0;
    for (let i = 0; i < L; i++){
      if (m.E && i >= (m.kdense || 0)){
        const one = 3*d*m.fe, sh = (m.shared || 0)*one, router = d*m.E;
        expT += m.E*one + sh + router;
        expA += k*one + sh + router;
      } else dense += 3*d*m.f;
    }
    const total = emb + attn + norms + dense + expT;
    const active = emb + attn + norms + dense + expA;
    return {emb, attn, dense, expT, expA, norms, total, active, k};
  }
  /* KV-cache elements stored per cached token slot, per layer */
  function kvElemsPerLayer(m, kvOverride){
    if (m.att === "mla" && !kvOverride) return m.mla.kvr + m.mla.rope;
    const hd = m.hd || (m.mla ? m.mla.v : m.d/m.h);
    const kv = kvOverride === "mha" ? m.h : kvOverride === "mqa" ? 1 : (m.kv || m.h);
    return 2*kv*hd;
  }
  function isLocal(m, i){
    if (!m.win) return false;
    if (m.pat === "all") return true;
    if (m.pat === "alt") return i % 2 === 0;
    if (m.pat === "g6") return (i + 1) % 6 !== 0;
    return false;
  }
  /* KV-cache bytes for one sequence of length T (bf16 = 2 bytes) */
  function kvBytes(m, T, swa, kvOverride){
    const e = kvElemsPerLayer(m, kvOverride);
    let slots = 0;
    for (let i = 0; i < m.L; i++) slots += (swa && isLocal(m, i)) ? Math.min(T, m.win) : T;
    return slots * e * 2;
  }
  function nLocal(m){ let c = 0; for (let i = 0; i < m.L; i++) if (isLocal(m, i)) c++; return c; }

  function fmtN(x){
    const a = Math.abs(x);
    if (a >= 1e12) return (x/1e12).toFixed(a >= 1e13 ? 1 : 2)+"T";
    if (a >= 1e9)  return (x/1e9).toFixed(a >= 1e11 ? 1 : 2)+"B";
    if (a >= 1e6)  return (x/1e6).toFixed(a >= 1e8 ? 0 : 1)+"M";
    if (a >= 1e3)  return (x/1e3).toFixed(1)+"k";
    return String(Math.round(x));
  }
  function bytes(b){
    const u = ["B","KiB","MiB","GiB","TiB"]; let i = 0, v = b;
    while (v >= 1024 && i < u.length-1){ v /= 1024; i++; }
    return v.toFixed(v >= 100 ? 0 : (v >= 10 ? 1 : 2))+" "+u[i];
  }
  function sci(x, p){ if (!isFinite(x) || x === 0) return "0"; const e = Math.floor(Math.log10(Math.abs(x))); return (x/Math.pow(10,e)).toFixed(p == null ? 2 : p)+" × 10"+sup(e); }
  function sup(e){ const m = {"-":"⁻","0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"}; return String(e).split("").map(c => m[c]).join(""); }
  function ctxLabel(T){ return T >= 1024 ? (T/1024 % 1 === 0 ? (T/1024)+"K" : (T/1024).toFixed(1)+"K") : String(T); }
  function txt(g, x, y, s, o){
    o = o || {};
    return g.append("text").attr("x",x).attr("y",y).attr("font-size",o.size||11).attr("fill",o.fill||P.ink)
      .attr("text-anchor",o.anchor||"start").attr("font-family",o.mono?"SF Mono,Menlo,monospace":null)
      .attr("font-weight",o.bold?600:null).text(s);
  }
  function safe(name, fn){ try { fn(); } catch(e){ if (typeof console !== "undefined") console.error("[llm-families.viz] "+name+" failed:", e); } }
  return {P, FAM, M, byId, params, kvBytes, kvElemsPerLayer, nLocal, isLocal, fmtN, bytes, sci, ctxLabel, txt, safe};
})();

/* ───────────────────────── 02 · lineage timeline ───────────────────────── */
LF.safe("lineage", function(){
  const {P, FAM, M, params, fmtN, txt} = LF;
  const svg = d3.select("#lin-svg"), W = 660, H = 330, m = {l:62, r:20, t:26, b:44};
  const pts = M.filter(d => !d.noTimeline).map(d => { const p = params(d); return Object.assign({}, d, {tot:p.total, act:p.active}); });
  const x = d3.scaleLinear().domain([2018.8, 2025.6]).range([m.l, W-m.r]);
  const y = d3.scaleLog().domain([8e8, 1.2e12]).range([H-m.b, m.t]);
  let sel = pts.findIndex(d => d.id === "dsv3"), fam = "all";
  const selEl = document.getElementById("lin-fam");
  if (selEl) selEl.addEventListener("change", () => { fam = selEl.value; const vis = pts.filter(d => fam === "all" || d.fam === fam); if (vis.length && (fam !== "all") && pts[sel].fam !== fam) sel = pts.indexOf(vis[vis.length-1]); draw(); });
  function draw(){
    svg.selectAll("*").remove();
    const g = svg.append("g");
    [1e9,1e10,1e11,1e12].forEach(v => {
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, m.l-8, y(v)+4, fmtN(v), {anchor:"end", size:10, fill:P.muted});
    });
    for (let yr = 2019; yr <= 2025; yr++){
      txt(g, x(yr), H-m.b+16, String(yr), {anchor:"middle", size:10, fill:P.muted});
      g.append("line").attr("x1",x(yr)).attr("x2",x(yr)).attr("y1",H-m.b).attr("y2",H-m.b+4).attr("stroke",P.muted);
    }
    txt(g, 10, m.t-8, "parameters (log)", {size:10, fill:P.muted});
    // legend
    Object.keys(FAM).forEach((f, i) => {
      const lx = m.l + i*96, ly = H-10;
      g.append("circle").attr("cx",lx).attr("cy",ly-4).attr("r",4).attr("fill",FAM[f]).attr("opacity", fam === "all" || fam === f ? 1 : .3);
      txt(g, lx+8, ly, f === "Mistral" ? "Mistral/Mixtral" : f, {size:10, fill:P.muted});
    });
    // per-family lines through dense totals
    Object.keys(FAM).forEach(f => {
      const s = pts.filter(d => d.fam === f).sort((a,b) => a.t-b.t);
      if (s.length < 2) return;
      g.append("path").datum(s).attr("fill","none").attr("stroke",FAM[f]).attr("stroke-opacity", fam === "all" || fam === f ? .35 : .08)
        .attr("d", d3.line().x(d => x(d.t)).y(d => y(d.tot)));
    });
    pts.forEach((d, i) => {
      const on = i === sel, vis = fam === "all" || d.fam === fam, cx = x(d.t), cy = y(d.tot);
      const node = g.append("g").style("cursor","pointer").attr("opacity", vis ? 1 : .15).on("click", () => { sel = i; draw(); });
      if (d.E){
        node.append("line").attr("x1",cx).attr("x2",cx).attr("y1",y(d.act)).attr("y2",cy).attr("stroke",FAM[d.fam]).attr("stroke-width",on?4:3).attr("stroke-opacity",.6);
        node.append("rect").attr("x",cx-5).attr("y",y(d.act)-5).attr("width",10).attr("height",10).attr("fill","none").attr("stroke",FAM[d.fam]).attr("stroke-width",1.5);
        node.append("path").attr("d", d3.symbol().type(d3.symbolDiamond).size(on?150:90)()).attr("transform",`translate(${cx},${cy})`).attr("fill",FAM[d.fam]).attr("stroke",on?P.ink:"none");
      } else {
        node.append("circle").attr("cx",cx).attr("cy",cy).attr("r",on?8:5.5).attr("fill",FAM[d.fam]).attr("stroke",on?P.ink:"none");
      }
      if (on || vis && fam !== "all") txt(node, cx, cy-12, d.name, {anchor: cx > W-90 ? "end" : "middle", size:10.5, fill:on?P.ink:P.muted, bold:on});
    });
    const d = pts[sel];
    const ratio = d.tot / d.rep;
    const tok = d.tokens ? `${fmtN(d.tokens)} training tokens (${(d.tokens/d.tot).toFixed(1)} per parameter)` : "training tokens not disclosed";
    const moe = d.E ? ` · MoE: ${d.E} routed experts${d.shared?` + ${d.shared} shared`:""}, ${d.k} active → <b>${fmtN(d.act)}</b> active of <b>${fmtN(d.tot)}</b> total (${(100*d.act/d.tot).toFixed(1)}% per token)` : ` · dense: <b>${fmtN(d.tot)}</b> parameters`;
    const prevF = pts.filter(p => p.fam === d.fam && p.t < d.t).sort((a,b) => b.t-a.t)[0];
    const grow = prevF ? ` · ${(d.tot/prevF.tot).toFixed(1)}× the total of ${prevF.name}` : "";
    d3.select("#lin-read").html(`<b>${d.name}</b> (${d.mon})${moe}; report label ${d.repLabel}, config count is ${(100*ratio).toFixed(1)}% of it · ${tok} · native context ${LF.ctxLabel(d.ctx)}${grow}.`);
  }
  draw();
});

/* ───────────────────────── 03 · design-axis matrix ───────────────────────── */
LF.safe("axes", function(){
  const {P, FAM, M, txt} = LF;
  const svg = d3.select("#axes-svg"), W = 680, H = 430;
  const rows = M.filter(d => d.id !== "qwen3d");
  const attnOf = d => (d.att === "mla" ? "MLA" : d.kv === d.h ? "MHA" : d.kv === 1 ? "MQA" : "GQA") + (d.win ? (d.pat === "all" ? " + SWA" : " + local/global") : "");
  const vocabOf = d => d.V <= 60000 ? "≤ 50k" : d.V < 200000 ? "100k–160k" : "≥ 256k";
  const ctxOf = d => d.ctx <= 4096 ? "≤ 4K" : d.ctx <= 8192 ? "8K" : d.ctx <= 32768 ? "32K" : "128K";
  const cols = [
    {key:"Attention", f:attnOf, why:{
      "MHA":"one K/V head per query head — the largest KV cache per token",
      "GQA":"query heads share a smaller set of K/V heads, cutting the cache by n_h / n_kv",
      "GQA + SWA":"grouped K/V heads and every layer limited to a sliding window, with a rolling-buffer cache",
      "GQA + local/global":"grouped K/V heads; local layers see only a window, global layers see everything",
      "MLA":"keys and values reconstructed from a cached low-rank latent plus a small RoPE key"}},
    {key:"Positions", f:d => d.pos, why:{
      "learned":"a learned absolute position vector per slot — context is capped by the table size",
      "RoPE 10k":"rotary embeddings with the original base θ = 10,000",
      "RoPE 500k":"rotary embeddings with θ raised to 500,000 for longer contexts",
      "RoPE 1M":"rotary embeddings with θ = 1,000,000",
      "RoPE 1M / 10k":"θ = 1,000,000 on global layers, 10,000 on local layers"}},
    {key:"Norm", f:d => d.norm, why:{
      "pre-LN":"LayerNorm at the input of each sub-layer",
      "RMSNorm pre":"RMSNorm (no mean subtraction, no bias) at the input of each sub-layer",
      "RMSNorm pre+post":"RMSNorm both before and after each sub-layer"}},
    {key:"Activation", f:d => d.act, why:{
      "GELU":"a two-matrix MLP with GELU, width 4·d",
      "SwiGLU":"a gated three-matrix FFN with a Swish (SiLU) gate",
      "GeGLU":"a gated three-matrix FFN with an approximate-GELU gate"}},
    {key:"FFN", f:d => d.E ? (d.shared ? "MoE + shared" : "MoE") : "dense", why:{
      "dense":"one FFN per layer; every parameter is used for every token",
      "MoE":"routed experts; only the top-k experts run for each token",
      "MoE + shared":"routed experts plus always-on shared experts"}},
    {key:"Tied emb.", f:d => d.tied ? "tied" : "untied", why:{
      "tied":"the output projection reuses the input embedding matrix (V·d parameters once)",
      "untied":"separate input and output matrices (2·V·d parameters)"}},
    {key:"Vocab", f:vocabOf, why:{
      "≤ 50k":"a small vocabulary; cheap embeddings, more tokens per non-English text",
      "100k–160k":"a mid-sized multilingual byte-level BPE",
      "≥ 256k":"a very large SentencePiece vocabulary shared with Gemini"}},
    {key:"Native ctx", f:ctxOf, why:{
      "≤ 4K":"trained with at most 4,096 positions",
      "8K":"trained at 8,192 positions",
      "32K":"trained or extended to 32,768 positions",
      "128K":"advertised 128K context after a long-context stage"}}
  ];
  const pal = [P.A, P.B, P.good, P.purple, P.teal, P.pink, P.bad, "#fcd34d"];
  cols.forEach(c => { c.opts = []; rows.forEach(r => { const v = c.f(r); if (c.opts.indexOf(v) < 0) c.opts.push(v); }); });
  const colSel = document.getElementById("ax-col");
  if (colSel && colSel.options.length < 2) cols.forEach((c, i) => { const o = document.createElement("option"); o.value = String(i); o.textContent = c.key; colSel.appendChild(o); });
  const lx = 128, top = 34, cw = (W - lx - 8)/cols.length, rh = (H - top - 10)/rows.length;
  let selR = rows.findIndex(r => r.id === "dsv3"), selC = 0, hi = -1;
  if (colSel) colSel.addEventListener("change", () => { hi = +colSel.value; if (hi >= 0) selC = hi; draw(); });
  function draw(){
    svg.selectAll("*").remove();
    const g = svg.append("g");
    cols.forEach((c, j) => txt(g, lx + j*cw + cw/2, top-12, c.key, {anchor:"middle", size:10.5, fill: j === hi ? P.ink : P.muted, bold: j === hi}));
    rows.forEach((r, i) => {
      txt(g, lx-8, top + i*rh + rh/2 + 4, r.name, {anchor:"end", size:10.5, fill: i === selR ? P.ink : P.muted, bold: i === selR});
      g.append("circle").attr("cx",6).attr("cy",top + i*rh + rh/2).attr("r",3.5).attr("fill",FAM[r.fam]);
      cols.forEach((c, j) => {
        const v = c.f(r), oi = c.opts.indexOf(v), on = i === selR && j === selC;
        const cell = g.append("g").style("cursor","pointer").on("click", () => { selR = i; selC = j; draw(); });
        cell.append("rect").attr("x", lx + j*cw + 1).attr("y", top + i*rh + 1).attr("width", cw-2).attr("height", rh-2).attr("rx",3)
          .attr("fill", pal[oi % pal.length]).attr("fill-opacity", hi < 0 || hi === j ? .55 : .14)
          .attr("stroke", on ? P.ink : "none").attr("stroke-width", 2);
        const lab = v.length > 12 ? v.replace(" + local/global"," + L/G").replace("RMSNorm ","RMS ") : v;
        txt(cell, lx + j*cw + cw/2, top + i*rh + rh/2 + 3.5, lab, {anchor:"middle", size:9, fill:P.ink});
      });
    });
    const c = cols[selC], r = rows[selR], v = c.f(r);
    const counts = c.opts.map(o => `${o}: ${rows.filter(q => c.f(q) === o).length}`).join(" · ");
    d3.select("#axes-read").html(`<b>${r.name} — ${c.key}: ${v}</b> — ${c.why[v] || ""}. Across the ${rows.length} models listed, ${c.key.toLowerCase()} splits as ${counts} (${c.opts.length === 1 ? "converged" : c.opts.length + " distinct choices"}).`);
  }
  draw();
});

/* ───────────────────────── 04 · KV cache vs context ───────────────────────── */
LF.safe("kv", function(){
  const {P, FAM, byId, kvBytes, kvElemsPerLayer, nLocal, bytes, ctxLabel, txt} = LF;
  const svg = d3.select("#kv-svg"), W = 660, H = 330, m = {l:64, r:150, t:20, b:40};
  const ids = ["gpt3","llama2","llama31","mistral","qwen3m","gemma2","gemma3","dsv3"];
  const ms = ids.map(id => byId[id]);
  const Tin = document.getElementById("kv-T"), Tv = document.getElementById("kv-Tv"),
        swaEl = document.getElementById("kv-swa"), scEl = document.getElementById("kv-scale");
  const grid = d3.range(10, 17.01, 0.25).map(e => Math.round(Math.pow(2, e)));
  function draw(){
    const T = Math.round(Math.pow(2, +(Tin ? Tin.value : 15))), swa = swaEl ? swaEl.checked : true, lin = scEl && scEl.value === "lin";
    if (Tv) Tv.textContent = ctxLabel(T);
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const GiB = Math.pow(1024, 3);
    const series = ms.map(mm => ({m:mm, pts:grid.map(t => [t, kvBytes(mm, t, swa)/GiB])}));
    const ymax = d3.max(series, s => d3.max(s.pts, p => p[1]));
    const ymin = d3.min(series, s => d3.min(s.pts, p => p[1]));
    const x = d3.scaleLog().base(2).domain([1024, 131072]).range([m.l, W-m.r]);
    const y = lin ? d3.scaleLinear().domain([0, ymax*1.05]).range([H-m.b, m.t])
                  : d3.scaleLog().domain([Math.max(ymin*0.8, 1e-3), ymax*1.3]).range([H-m.b, m.t]);
    const yt = lin ? y.ticks(5) : y.ticks(5).filter(v => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9);
    yt.forEach(v => {
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, m.l-6, y(v)+4, v >= 1 ? v.toFixed(0) : String(v), {anchor:"end", size:10, fill:P.muted});
    });
    [1024, 4096, 16384, 65536, 131072].forEach(t => txt(g, x(t), H-m.b+16, ctxLabel(t), {anchor:"middle", size:10, fill:P.muted}));
    txt(g, 8, m.t-6, "GiB per sequence", {size:10, fill:P.muted});
    txt(g, (m.l + W - m.r)/2, H-6, "context length T (tokens, log scale)", {anchor:"middle", size:10, fill:P.muted});
    g.append("line").attr("x1",x(T)).attr("x2",x(T)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.B).attr("stroke-dasharray","4,3");
    const endY = [];
    series.forEach(s => {
      g.append("path").datum(s.pts).attr("fill","none").attr("stroke",FAM[s.m.fam]).attr("stroke-width",1.8)
        .attr("stroke-dasharray", s.m.fam === "Gemma" && s.m.id === "gemma2" ? "5,3" : null)
        .attr("d", d3.line().x(p => x(p[0])).y(p => y(Math.max(p[1], y.domain()[0]))));
      const v = kvBytes(s.m, T, swa)/GiB;
      g.append("circle").attr("cx",x(T)).attr("cy",y(Math.max(v, y.domain()[0]))).attr("r",3.5).attr("fill",FAM[s.m.fam]);
      endY.push({m:s.m, y:y(Math.max(s.pts[s.pts.length-1][1], y.domain()[0]))});
    });
    endY.sort((a,b) => a.y-b.y);
    let last = -1e9; endY.forEach(e => { const yy = Math.max(e.y, last + 12); last = yy; txt(g, W-m.r+6, yy+4, e.m.name, {size:10, fill:FAM[e.m.fam]}); });
    const rank = ms.map(mm => ({mm, b:kvBytes(mm, T, swa), per:kvElemsPerLayer(mm)*2*mm.L})).sort((a,b) => a.b-b.b);
    const lo = rank[0], hi = rank[rank.length-1];
    const notes = ms.filter(mm => mm.win).map(mm => `${mm.name}: ${nLocal(mm)} of ${mm.L} layers local (window ${mm.win})`).join("; ");
    d3.select("#kv-read").html(`At T = <b>${ctxLabel(T)}</b>${swa ? "" : " (sliding windows ignored)"}: smallest cache <b>${lo.mm.name}</b> ${bytes(lo.b)}, largest <b>${hi.mm.name}</b> ${bytes(hi.b)} — a ${(hi.b/lo.b).toFixed(0)}× spread. ` +
      `Full-attention bytes per token: ` + rank.slice().sort((a,b) => a.per-b.per).map(r => `${r.mm.name} ${bytes(r.per)}`).join(", ") + `. ${swa ? notes + "." : ""}`);
  }
  [Tin, swaEl, scEl].forEach(el => { if (el) el.addEventListener(el.type === "range" ? "input" : "change", draw); });
  draw();
});

/* ───────────────────────── 08 · tokens per parameter ───────────────────────── */
LF.safe("tpp", function(){
  const {P, FAM, M, params, fmtN, sci, txt} = LF;
  const svg = d3.select("#tpp-svg"), W = 660, H = 360, m = {l:56, r:16, t:20, b:96};
  const denEl = document.getElementById("tpp-den"), sortEl = document.getElementById("tpp-sort");
  const rows = M.filter(d => d.tokens).map(d => { const p = params(d); return Object.assign({}, d, {tot:p.total, act:p.active}); });
  function draw(){
    const act = denEl && denEl.value === "active", byRatio = sortEl && sortEl.value === "ratio";
    const data = rows.map(r => ({r, v: r.tokens / (act ? r.act : r.tot)}));
    data.sort(byRatio ? (a,b) => a.v-b.v : (a,b) => a.r.t-b.r.t);
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const x = d3.scaleBand().domain(data.map(d => d.r.id)).range([m.l, W-m.r]).padding(0.22);
    const y = d3.scaleLog().domain([1, 2000]).range([H-m.b, m.t]);
    [1, 10, 100, 1000].forEach(v => {
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, m.l-6, y(v)+4, String(v), {anchor:"end", size:10, fill:P.muted});
    });
    txt(g, 8, m.t-6, "tokens per parameter (log)", {size:10, fill:P.muted});
    data.forEach(d => {
      g.append("rect").attr("x",x(d.r.id)).attr("width",x.bandwidth()).attr("y",y(Math.max(d.v,1))).attr("height",(H-m.b)-y(Math.max(d.v,1)))
        .attr("fill",FAM[d.r.fam]).attr("fill-opacity",.75);
      txt(g, x(d.r.id)+x.bandwidth()/2, y(Math.max(d.v,1))-4, d.v >= 100 ? d.v.toFixed(0) : d.v.toFixed(1), {anchor:"middle", size:9, fill:P.ink});
      g.append("text").attr("transform",`translate(${x(d.r.id)+x.bandwidth()/2},${H-m.b+8}) rotate(-50)`).attr("text-anchor","end")
        .attr("font-size",9.5).attr("fill",P.muted).text(d.r.name);
    });
    const ref = 20;
    g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(ref)).attr("y2",y(ref)).attr("stroke",P.B).attr("stroke-dasharray","6,4").attr("stroke-width",1.5);
    txt(g, W-m.r-4, y(ref)-5, `≈ ${ref} tokens / parameter`, {anchor:"end", size:10, fill:P.B});
    const vals = data.map(d => d.v).sort((a,b) => a-b);
    const med = vals.length % 2 ? vals[(vals.length-1)/2] : (vals[vals.length/2-1] + vals[vals.length/2])/2;
    const above = data.filter(d => d.v > ref).length;
    const big = rows.slice().sort((a,b) => b.act*b.tokens - a.act*a.tokens)[0];
    d3.select("#tpp-read").html(`Dividing by <b>${act ? "active" : "total"}</b> parameters: median <b>${med.toFixed(0)}</b> tokens per parameter across ${data.length} models; ${above} of ${data.length} exceed ≈ ${ref}. ` +
      `Lowest: ${data.slice().sort((a,b)=>a.v-b.v)[0].r.name} (${data.slice().sort((a,b)=>a.v-b.v)[0].v.toFixed(1)}); highest: ${data.slice().sort((a,b)=>b.v-a.v)[0].r.name} (${data.slice().sort((a,b)=>b.v-a.v)[0].v.toFixed(0)}). ` +
      `Largest training compute here, 6 · N_active · D: ${big.name}, 6 × ${fmtN(big.act)} × ${fmtN(big.tokens)} ≈ ${sci(6*big.act*big.tokens, 1)} FLOPs.`);
  }
  [denEl, sortEl].forEach(el => { if (el) el.addEventListener("change", draw); });
  draw();
});

/* ───────────────────────── 09 · config explorer ───────────────────────── */
LF.safe("cfg", function(){
  const {P, M, byId, params, kvElemsPerLayer, fmtN, bytes, txt} = LF;
  const svg = d3.select("#cfg-svg"), W = 660, H = 300, m = {l:70, r:20, t:30, b:70};
  const modelEl = document.getElementById("cfg-model"), kvEl = document.getElementById("cfg-kv"),
        kEl = document.getElementById("cfg-k"), kOut = document.getElementById("cfg-kv-k");
  if (modelEl && modelEl.options.length === 0) M.forEach(d => { const o = document.createElement("option"); o.value = d.id; o.textContent = d.name; modelEl.appendChild(o); });
  if (modelEl) modelEl.value = "dsv3";
  let cur = byId[modelEl ? modelEl.value : "dsv3"] || M[0];
  function syncK(){ if (kEl && cur.E){ kEl.max = String(Math.min(16, cur.E)); kEl.value = String(cur.k); } }
  syncK();
  const parts = [
    {key:"emb",   label:"embeddings", col:P.muted},
    {key:"attn",  label:"attention",  col:P.A},
    {key:"dense", label:"dense FFN",  col:P.B},
    {key:"exp",   label:"experts",    col:P.purple},
    {key:"norms", label:"norms",      col:P.good}
  ];
  function draw(){
    const kvMode = kvEl ? kvEl.value : "cfg";
    const k = cur.E ? Math.max(1, Math.min(cur.E, +(kEl ? kEl.value : cur.k))) : 0;
    if (kOut) kOut.textContent = cur.E ? String(k) : "n/a";
    const p = params(cur, {kv: kvMode, k: k || undefined});
    const base = params(cur);
    svg.selectAll("*").remove();
    const g = svg.append("g");
    const bars = [
      {name:"total", vals:{emb:p.emb, attn:p.attn, dense:p.dense, exp:p.expT, norms:p.norms}, sum:p.total},
      {name:"active / token", vals:{emb:p.emb, attn:p.attn, dense:p.dense, exp:p.expA, norms:p.norms}, sum:p.active}
    ];
    const x = d3.scaleLinear().domain([0, p.total]).range([m.l, W-m.r]);
    bars.forEach((b, i) => {
      const yy = m.t + i*70;
      txt(g, m.l-8, yy+26, b.name, {anchor:"end", size:11, fill:P.ink});
      let acc = 0;
      parts.forEach(pt => {
        const v = b.vals[pt.key]; if (!v) return;
        g.append("rect").attr("x",x(acc)).attr("y",yy+8).attr("width",Math.max(0.5, x(acc+v)-x(acc))).attr("height",30).attr("fill",pt.col).attr("fill-opacity",.8);
        if (x(acc+v)-x(acc) > 46) txt(g, (x(acc)+x(acc+v))/2, yy+27, fmtN(v), {anchor:"middle", size:10, fill:"#0f1117", bold:true});
        acc += v;
      });
      txt(g, Math.min(x(acc)+6, W-m.r-2), yy+52, fmtN(b.sum), {anchor: x(acc) > W-80 ? "end" : "start", size:11, fill:P.ink, bold:true});
    });
    parts.forEach((pt, i) => {
      const lx = m.l + i*112, ly = H-24;
      g.append("rect").attr("x",lx).attr("y",ly-9).attr("width",10).attr("height",10).attr("fill",pt.col);
      txt(g, lx+14, ly, pt.label, {size:10, fill:P.muted});
    });
    const kvPer = kvElemsPerLayer(cur, kvMode !== "cfg" ? kvMode : null) * 2 * cur.L;
    const kvBase = kvElemsPerLayer(cur) * 2 * cur.L;
    const attnKind = cur.att === "mla" && kvMode === "cfg" ? `MLA (latent ${cur.mla.kvr} + RoPE key ${cur.mla.rope} per layer)` :
      `${kvMode === "mha" ? cur.h : kvMode === "mqa" ? 1 : (cur.kv || cur.h)} KV heads of width ${cur.hd || (cur.mla ? cur.mla.v : cur.d/cur.h)}`;
    const repTxt = cur.repA ? `report: ${cur.repLabel} → config gives ${fmtN(base.total)} / ${fmtN(base.active)} (${(100*base.total/cur.rep).toFixed(1)}% / ${(100*base.active/cur.repA).toFixed(1)}%)`
                            : `report: ${cur.repLabel} → config gives ${fmtN(base.total)} (${(100*base.total/cur.rep).toFixed(1)}%)`;
    const over = kvMode !== "cfg" || (cur.E && k !== cur.k)
      ? ` With your overrides: ${fmtN(p.total)} total, ${fmtN(p.active)} active; KV cache ${bytes(kvPer)} per token (${(kvPer/kvBase).toFixed(2)}× the configured ${bytes(kvBase)}).` : "";
    d3.select("#cfg-read").html(`<b>${cur.name}</b> — L = ${cur.L}, d = ${cur.d}, ${cur.h} query heads, ${attnKind}, vocab ${cur.V.toLocaleString("en-US")}${cur.tied ? " (tied)" : ""}` +
      (cur.E ? `, ${cur.E} experts × width ${cur.fe}${cur.shared ? ` + ${cur.shared} shared` : ""}, top-${cur.k}${cur.kdense ? `, first ${cur.kdense} layer${cur.kdense>1?"s":""} dense` : ""}` : `, FFN width ${cur.f}`) +
      `. ${repTxt}. Embeddings are ${(100*base.emb/base.total).toFixed(1)}% of the total. KV cache ${bytes(kvBase)} per token in bf16 (${bytes(kvBase*32768)} at 32K context, full attention).${over}`);
  }
  if (modelEl) modelEl.addEventListener("change", () => { cur = byId[modelEl.value] || M[0]; syncK(); draw(); });
  if (kvEl) kvEl.addEventListener("change", draw);
  if (kEl) kEl.addEventListener("input", draw);
  draw();
});
