/* llama.viz.js — every interactive figure on models/llama.html.
   Loaded after data.js / notes.js (palette C comes from notes.js).
   Each figure is its own block wrapped in LV.safe so one failure cannot blank the rest.
   House rule: every number a figure displays is computed here from the released configs
   (the spec below), never typed into a label. */

const LV = (function(){
  const P = Object.assign({}, (typeof C !== "undefined" ? C : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"}),
    { panel:"#15181f", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6" });

  /* ── the spec: released Hugging Face configs (L, d, heads, KV heads, head dim, FFN width, vocab, tying),
        plus what the papers / model cards report (tokens, GPU-hours, the size label). ── */
  const A100 = 312e12, H100 = 989e12;   /* dense bf16 peak FLOP/s per GPU, used only for the implied-MFU view */
  const CFG = {
    l1_7:  {name:"LLaMA 7B",       rel:"l1",  L:32,  d:4096,  h:32,  kv:32, hd:128, f:11008, V:32000,  tied:false, ctx:2048,   tok:1.0e12,  tokLabel:"1.0T", rep:6.7e9,  repLabel:"6.7B",  gpuh:82432,    peak:A100, mult:null, mo:256},
    l1_13: {name:"LLaMA 13B",      rel:"l1",  L:40,  d:5120,  h:40,  kv:40, hd:128, f:13824, V:32000,  tied:false, ctx:2048,   tok:1.0e12,  tokLabel:"1.0T", rep:13.0e9, repLabel:"13.0B", gpuh:135168,   peak:A100, mult:null, mo:256},
    l1_33: {name:"LLaMA 33B",      rel:"l1",  L:60,  d:6656,  h:52,  kv:52, hd:128, f:17920, V:32000,  tied:false, ctx:2048,   tok:1.4e12,  tokLabel:"1.4T", rep:32.5e9, repLabel:"32.5B", gpuh:530432,   peak:A100, mult:null, mo:256},
    l1_65: {name:"LLaMA 65B",      rel:"l1",  L:80,  d:8192,  h:64,  kv:64, hd:128, f:22016, V:32000,  tied:false, ctx:2048,   tok:1.4e12,  tokLabel:"1.4T", rep:65.2e9, repLabel:"65.2B", gpuh:1022362,  peak:A100, mult:null, mo:256},
    l2_7:  {name:"Llama 2 7B",     rel:"l2",  L:32,  d:4096,  h:32,  kv:32, hd:128, f:11008, V:32000,  tied:false, ctx:4096,   tok:2.0e12,  tokLabel:"2.0T", rep:7e9,    repLabel:"7B",    gpuh:184320,   peak:A100, mult:null, mo:256},
    l2_13: {name:"Llama 2 13B",    rel:"l2",  L:40,  d:5120,  h:40,  kv:40, hd:128, f:13824, V:32000,  tied:false, ctx:4096,   tok:2.0e12,  tokLabel:"2.0T", rep:13e9,   repLabel:"13B",   gpuh:368640,   peak:A100, mult:null, mo:256},
    l2_70: {name:"Llama 2 70B",    rel:"l2",  L:80,  d:8192,  h:64,  kv:8,  hd:128, f:28672, V:32000,  tied:false, ctx:4096,   tok:2.0e12,  tokLabel:"2.0T", rep:70e9,   repLabel:"70B",   gpuh:1720320,  peak:A100, mult:1.3,  mo:4096},
    l3_8:  {name:"Llama 3 8B",     rel:"l3",  L:32,  d:4096,  h:32,  kv:8,  hd:128, f:14336, V:128256, tied:false, ctx:8192,   tok:15e12,   tokLabel:"15T+", rep:8e9,    repLabel:"8B",    gpuh:null,     peak:H100, mult:1.3,  mo:1024},
    l3_70: {name:"Llama 3 70B",    rel:"l3",  L:80,  d:8192,  h:64,  kv:8,  hd:128, f:28672, V:128256, tied:false, ctx:8192,   tok:15e12,   tokLabel:"15T+", rep:70e9,   repLabel:"70B",   gpuh:null,     peak:H100, mult:1.3,  mo:4096},
    l31_8: {name:"Llama 3.1 8B",   rel:"l31", L:32,  d:4096,  h:32,  kv:8,  hd:128, f:14336, V:128256, tied:false, ctx:131072, tok:15e12,   tokLabel:"~15T", rep:8e9,    repLabel:"8B",    gpuh:1.46e6,   peak:H100, mult:1.3,  mo:1024},
    l31_70:{name:"Llama 3.1 70B",  rel:"l31", L:80,  d:8192,  h:64,  kv:8,  hd:128, f:28672, V:128256, tied:false, ctx:131072, tok:15e12,   tokLabel:"~15T", rep:70e9,   repLabel:"70B",   gpuh:7.0e6,    peak:H100, mult:1.3,  mo:4096},
    l31_405:{name:"Llama 3.1 405B",rel:"l31", L:126, d:16384, h:128, kv:8,  hd:128, f:53248, V:128256, tied:false, ctx:131072, tok:15.6e12, tokLabel:"15.6T",rep:405e9,  repLabel:"405B",  gpuh:30.84e6,  peak:H100, mult:1.2,  mo:4096},
    l32_1: {name:"Llama 3.2 1B",   rel:"l32", L:16,  d:2048,  h:32,  kv:8,  hd:64,  f:8192,  V:128256, tied:true,  ctx:131072, tok:null,    tokLabel:"not in the sources used", rep:1e9, repLabel:"1B", gpuh:null, peak:H100, mult:1.5, mo:256},
    l32_3: {name:"Llama 3.2 3B",   rel:"l32", L:28,  d:3072,  h:24,  kv:8,  hd:128, f:8192,  V:128256, tied:true,  ctx:131072, tok:null,    tokLabel:"not in the sources used", rep:3e9, repLabel:"3B", gpuh:null, peak:H100, mult:1.0, mo:256},
    l33_70:{name:"Llama 3.3 70B",  rel:"l33", L:80,  d:8192,  h:64,  kv:8,  hd:128, f:28672, V:128256, tied:false, ctx:131072, tok:15e12,   tokLabel:"15T+", rep:70e9,   repLabel:"70B",   gpuh:7.0e6,    peak:H100, mult:1.3,  mo:4096},
    l4_s:  {name:"Llama 4 Scout",  rel:"l4",  L:48,  d:5120,  h:40,  kv:8,  hd:128, f:16384, V:202048, tied:false, ctx:10485760,tok:40e12,  tokLabel:"~40T", rep:109e9,  repLabel:"109B total / 17B active", gpuh:5.0e6, peak:H100,
            E:16,  top:1, fe:8192, moeStep:1, chunk:8192, nopeEvery:4},
    l4_m:  {name:"Llama 4 Maverick",rel:"l4", L:48,  d:5120,  h:40,  kv:8,  hd:128, f:16384, V:202048, tied:false, ctx:1048576, tok:22e12,  tokLabel:"~22T", rep:400e9,  repLabel:"400B total / 17B active", gpuh:2.38e6, peak:H100,
            E:128, top:1, fe:8192, moeStep:2, chunk:8192, nopeEvery:4}
  };
  const REL = {
    l1:  {name:"LLaMA (1)", t:2023.15, mon:"Feb 2023", ids:["l1_7","l1_13","l1_33","l1_65"], what:"public-data-only pretraining on 1.0–1.4T tokens; pre-norm RMSNorm, SwiGLU, RoPE; 2,048-token context; research licence"},
    l2:  {name:"Llama 2",   t:2023.55, mon:"Jul 2023", ids:["l2_7","l2_13","l2_70"], what:"2.0T tokens, 4,096-token context, GQA on the 34B (unreleased) and 70B, and Llama 2-Chat (SFT → rejection sampling → PPO, Ghost Attention); community licence with commercial use"},
    l3:  {name:"Llama 3",   t:2024.30, mon:"Apr 2024", ids:["l3_8","l3_70"], what:"128K-entry tiktoken-based vocabulary, GQA with 8 KV heads at every size, RoPE base 500,000, 15T+ tokens, 8,192-token context"},
    l31: {name:"Llama 3.1", t:2024.56, mon:"Jul 2024", ids:["l31_8","l31_70","l31_405"], what:"dense 405B on 15.6T tokens (3.8 × 10²⁵ FLOPs), 128K context via a six-stage long-context phase and the llama3 RoPE rescaling, multilingual and tool-use post-training"},
    l32: {name:"Llama 3.2", t:2024.73, mon:"Sep 2024", ids:["l32_1","l32_3"], what:"1B and 3B text models pruned from 3.1 8B and distilled from 8B/70B logits, tied embeddings, 128K context; 11B and 90B vision models built from 3.1 8B/70B plus cross-attention image adapters (not counted here)"},
    l33: {name:"Llama 3.3", t:2024.93, mon:"Dec 2024", ids:["l33_70"], what:"a text-only 70B instruction model on the 3.1 70B architecture with newer post-training"},
    l4:  {name:"Llama 4",   t:2025.26, mon:"Apr 2025", ids:["l4_s","l4_m"], what:"mixture-of-experts (one routed expert + one shared expert per token), early-fusion image input, interleaved NoPE layers (iRoPE), 202,048-row vocabulary"}
  };

  function isMoe(c, i){ if (!c.E) return false; return c.moeStep === 1 ? true : (i % c.moeStep === c.moeStep - 1); }
  function isNope(c, i){ return !!c.nopeEvery && (i % c.nopeEvery === c.nopeEvery - 1); }

  /* exact count for a Llama-style decoder: bias-free q/k/v/o with GQA, SwiGLU (3 matrices), two RMSNorm gains per block,
     final RMSNorm, input embedding, and an output head unless tied. MoE layers: E routed experts + 1 shared + router. */
  function params(c){
    const d = c.d, q = c.h*c.hd, kvw = c.kv*c.hd;
    const attnL = d*q + 2*d*kvw + q*d;
    let mlp = 0, mlpAct = 0, router = 0;
    for (let i=0;i<c.L;i++){
      if (isMoe(c,i)){ const one = 3*d*c.fe; mlp += c.E*one + one; mlpAct += c.top*one + one; router += d*c.E; }
      else { const one = 3*d*c.f; mlp += one; mlpAct += one; }
    }
    const attn = c.L*attnL, norm = c.L*2*d + d, emb = c.V*d, head = c.tied ? 0 : c.V*d;
    const total = emb + head + attn + mlp + router + norm;
    const active = emb + head + attn + mlpAct + router + norm;
    return {attn, attnL, mlp, mlpAct, router, norm, emb, head, total, active, nonEmb: attn + mlp + router + norm};
  }
  /* KV cache bytes for one sequence of n tokens; chunked-attention layers (Llama 4 RoPE layers) keep at most `chunk` tokens */
  function kvBytes(c, n, bytes, hybrid){
    let tok = 0;
    for (let i=0;i<c.L;i++){
      const chunked = hybrid && c.chunk && !isNope(c,i);
      tok += chunked ? Math.min(n, c.chunk) : n;
    }
    return 2 * tok * c.kv * c.hd * bytes;
  }
  function kvPerToken(c, bytes){ return 2 * c.L * c.kv * c.hd * bytes; }
  /* the reference implementation's FFN width rule */
  function ffnHidden(d, mult, mo){
    const s1 = 4*d, s2 = Math.floor(2*s1/3), s3 = mult ? Math.floor(mult*s2) : s2, f = mo*Math.ceil(s3/mo);
    return {s1, s2, s3, f};
  }
  /* RoPE inverse frequencies and the llama3 rescaling rule (as implemented for rope_type "llama3") */
  function invFreq(theta, hd){ const r=[]; for (let i=0;i<hd/2;i++) r.push(Math.pow(theta, -2*i/hd)); return r; }
  function llama3Scale(inv, sc){
    if (!sc) return inv.map(v=>({v, band:"none"}));
    const lowW = sc.orig/sc.lo, highW = sc.orig/sc.hi;
    return inv.map(f=>{
      const w = 2*Math.PI/f;
      if (w < highW) return {v:f, band:"keep"};
      if (w > lowW)  return {v:f/sc.factor, band:"divide"};
      const s = (sc.orig/w - sc.lo)/(sc.hi - sc.lo);
      return {v:(1-s)*f/sc.factor + s*f, band:"blend"};
    });
  }
  function fmtN(x){
    const a = Math.abs(x);
    if (a >= 1e12) return (x/1e12).toFixed(2)+"T";
    if (a >= 1e9)  return (x/1e9).toFixed(a>=1e11?1:2)+"B";
    if (a >= 1e6)  return (x/1e6).toFixed(a>=1e8?1:2)+"M";
    if (a >= 1e3)  return (x/1e3).toFixed(1)+"k";
    return String(Math.round(x));
  }
  function sup(e){ const m={"-":"⁻","0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"}; return String(e).split("").map(ch=>m[ch]).join(""); }
  function sci(x, p){ if (!isFinite(x) || x===0) return "0"; const e = Math.floor(Math.log10(Math.abs(x))); return (x/Math.pow(10,e)).toFixed(p==null?2:p)+" × 10"+sup(e); }
  function bytes(b){ const u=["B","KiB","MiB","GiB","TiB"]; let i=0, v=b; while (v >= 1024 && i < u.length-1){ v/=1024; i++; } return v.toFixed(v>=100?0:(v>=10?1:2))+" "+u[i]; }
  function gb(b){ return (b/1e9).toFixed(b>=1e11?0:1)+" GB"; }
  function ctxLabel(n){ return n >= 1048576 ? (n/1048576)+"M" : (n >= 1024 ? (n/1024)+"K" : String(n)); }
  function txt(g, x, y, s, o){
    o = o || {};
    return g.append("text").attr("x",x).attr("y",y).attr("font-size",o.size||11).attr("fill",o.fill||P.ink)
      .attr("text-anchor",o.anchor||"start").attr("font-family",o.mono?"SF Mono,Menlo,monospace":null)
      .attr("font-weight",o.bold?600:null).text(s);
  }
  function safe(name, fn){ try { fn(); } catch(e){ if (typeof console!=="undefined") console.error("[llama.viz] "+name+" failed:", e); } }
  function lcg(seed){ let s = seed >>> 0; return () => { s = (1664525*s + 1013904223) >>> 0; return (s+0.5)/4294967296; }; }
  function gauss(r){ return Math.sqrt(-2*Math.log(r())) * Math.cos(2*Math.PI*r()); }
  return {P, CFG, REL, isMoe, isNope, params, kvBytes, kvPerToken, ffnHidden, invFreq, llama3Scale, fmtN, sci, bytes, gb, ctxLabel, txt, safe, lcg, gauss};
})();

/* ───────────────────────── 02 · lineage ───────────────────────── */
LV.safe("lineage", function(){
  const {P, CFG, REL, params, fmtN, ctxLabel, txt} = LV;
  const svg = d3.select("#lin-svg"), W = 640, H = 290, m = {l:60, r:20, t:20, b:40};
  const keys = Object.keys(REL);
  const x = d3.scaleLinear().domain([2022.95, 2025.5]).range([m.l, W-m.r]);
  const y = d3.scaleLog().domain([5e8, 1e12]).range([H-m.b, m.t]);
  let sel = "l31";
  function draw(){
    svg.selectAll("*").remove();
    const g = svg.append("g");
    [1e9,1e10,1e11,1e12].forEach(v=>{
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, m.l-8, y(v)+4, fmtN(v), {anchor:"end", size:10, fill:P.muted});
    });
    [2023,2024,2025].forEach(v=>{ txt(g, x(v), H-m.b+16, String(v), {anchor:"middle", size:10, fill:P.muted});
      g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",H-m.b).attr("y2",H-m.b+4).attr("stroke",P.muted); });
    txt(g, 12, m.t+2, "params", {size:10, fill:P.muted});
    keys.forEach((k,j)=>{
      const R = REL[k], on = k===sel, cx = x(R.t);
      const node = g.append("g").style("cursor","pointer").on("click",()=>{ sel=k; draw(); });
      const ps = R.ids.map(id=>params(CFG[id]));
      const tops = ps.map(p=>p.total);
      node.append("line").attr("x1",cx).attr("x2",cx).attr("y1",y(d3.min(tops))).attr("y2",y(d3.max(tops))).attr("stroke",on?P.B:P.muted).attr("stroke-width",3).attr("stroke-opacity",.4);
      ps.forEach((p,i)=>{
        node.append("circle").attr("cx",cx).attr("cy",y(p.total)).attr("r",on?6:4.5).attr("fill",on?P.B:(k==="l4"?P.purple:P.A));
        if (p.active < p.total*0.99) node.append("circle").attr("cx",cx).attr("cy",y(p.active)).attr("r",on?6:4.5).attr("fill","none").attr("stroke",on?P.B:P.purple).attr("stroke-width",1.5);
      });
      txt(node, cx, y(d3.max(tops)) - 10 - (j%2)*12, R.name, {anchor:"middle", size:10.5, fill:on?P.B:P.ink, bold:on});
      node.append("rect").attr("x",cx-14).attr("y",m.t).attr("width",28).attr("height",H-m.b-m.t).attr("fill","transparent");
    });
    txt(g, W-m.r, H-6, "filled = total parameters · hollow = active per token (MoE)", {anchor:"end", size:9.5, fill:P.muted});
    const R = REL[sel];
    const rows = R.ids.map(id=>{ const c = CFG[id], p = params(c); return `${c.name}: <b>${fmtN(p.total)}</b>${p.active < p.total*0.99 ? " total, <b>"+fmtN(p.active)+"</b> active" : ""}`; }).join(" · ");
    const c0 = CFG[R.ids[0]];
    d3.select("#lin-read").html(`<b>${R.name}</b> (${R.mon}): ${R.what}. Sizes recomputed from the configs — ${rows}. Vocabulary ${c0.V.toLocaleString("en-US")} rows; context ${ctxLabel(c0.ctx)} tokens; pretraining tokens ${c0.tokLabel}.`);
  }
  draw();
});

/* ───────────────────────── 03 · tokens per parameter and lifetime compute ───────────────────────── */
LV.safe("tpp", function(){
  const {P, CFG, params, fmtN, sci, txt} = LV;
  const svg = d3.select("#tpp-svg"), W = 640, H = 330, m = {l:120, r:30, t:18, b:34};
  const ids = ["l1_7","l1_13","l1_33","l1_65","l2_7","l2_70","l31_8","l31_70","l31_405","l4_s","l4_m"];
  const refs = [{name:"GPT-3 175B (ref.)", N:174.6e9, D:300e9}, {name:"70B, 1.4T (compute-optimal paper)", N:70e9, D:1.4e12}];
  function rows(){
    return ids.map(id=>{ const c = CFG[id], p = params(c); const N = c.E ? p.active : p.total; return {id, name:c.name, N, Ntot:p.total, D:c.tok, ref:false}; })
      .concat(refs.map(r=>({id:r.name, name:r.name, N:r.N, Ntot:r.N, D:r.D, ref:true})));
  }
  function draw(){
    svg.selectAll("*").remove();
    const R = rows(), g = svg.append("g"), key = d3.select("#tpp-model").property("value"), lq = +d3.select("#tpp-q").property("value");
    d3.select("#tpp-qv").text(fmtN(Math.pow(10,lq)));
    const xs = d3.scaleLog().domain([1, 5000]).range([m.l, W-m.r]), bh = (H-m.t-m.b)/R.length;
    [1,10,100,1000].forEach(v=>{ g.append("line").attr("x1",xs(v)).attr("x2",xs(v)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, xs(v), H-m.b+14, String(v), {anchor:"middle", size:10, fill:P.muted}); });
    txt(g, (m.l+W-m.r)/2, H-4, "training tokens per parameter (log; MoE rows use active parameters)", {anchor:"middle", size:10, fill:P.muted});
    R.forEach((r,i)=>{
      const tpp = r.D/r.N, yy = m.t + i*bh, on = r.id===key;
      txt(g, m.l-6, yy+bh/2+4, r.name, {anchor:"end", size:10, fill:on?P.B:(r.ref?P.muted:P.ink), bold:on});
      g.append("rect").attr("x",m.l).attr("y",yy+3).attr("width",Math.max(1,xs(tpp)-m.l)).attr("height",bh-6).attr("rx",3)
        .attr("fill", r.ref ? P.muted : (on ? P.B : P.A)).attr("fill-opacity", on ? .9 : .55);
      txt(g, xs(tpp)+5, yy+bh/2+4, tpp.toFixed(tpp<10?1:0), {size:9.5, fill:P.muted});
    });
    g.append("line").attr("x1",xs(20)).attr("x2",xs(20)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.good).attr("stroke-width",1.5);
    txt(g, xs(20)+4, m.t+10, "≈ 20 tokens / param", {size:9.5, fill:P.good});
    const c = CFG[key], p = params(c), N = c.E ? p.active : p.total, D = c.tok, Q = Math.pow(10,lq);
    const Ctr = 6*N*D, Cinf = 2*N*Q, share = Cinf/(Ctr+Cinf), Nopt = Math.sqrt(Ctr/120);
    d3.select("#tpp-read").html(`<b>${c.name}</b>: ${(D/N).toFixed(0)} tokens per ${c.E?"active ":""}parameter. Training ≈ 6·N·D = <b>${sci(Ctr)}</b> FLOPs; serving ${fmtN(Q)} tokens ≈ 2·N·Q = <b>${sci(Cinf)}</b> FLOPs, so inference is <b>${(100*share).toFixed(1)}%</b> of lifetime compute. A 20-tokens-per-parameter model with the same training compute would have N ≈ ${fmtN(Nopt)} (${(Nopt/N).toFixed(1)}× larger) and cost ${(Nopt/N).toFixed(1)}× more per served token.`);
  }
  d3.select("#tpp-model").on("change", draw);
  d3.select("#tpp-q").on("input", draw);
  draw();
});

/* ───────────────────────── 05 · data mixes ───────────────────────── */
LV.safe("mix", function(){
  const {P, fmtN, txt} = LV;
  const svg = d3.select("#mix-svg"), W = 640, H = 270, m = {l:120, r:90, t:16, b:30};
  const MIX = {
    l1: {total:1.4e12, label:"LLaMA 1 (Table 1, 1.4T tokens)", rows:[
      {s:"CommonCrawl",   p:67.0, ep:1.10, disk:"3.3 TB"}, {s:"C4", p:15.0, ep:1.06, disk:"783 GB"},
      {s:"GitHub",        p:4.5,  ep:0.64, disk:"328 GB"}, {s:"Wikipedia", p:4.5, ep:2.45, disk:"83 GB"},
      {s:"Books",         p:4.5,  ep:2.23, disk:"85 GB"},  {s:"ArXiv", p:2.5, ep:1.06, disk:"92 GB"},
      {s:"StackExchange", p:2.0,  ep:1.03, disk:"78 GB"}]},
    l3: {total:15.6e12, label:"Llama 3 (§3.1.2, final mix, 15.6T tokens)", rows:[
      {s:"general knowledge", p:50}, {s:"math & reasoning", p:25}, {s:"code", p:17}, {s:"multilingual", p:8}]}
  };
  function draw(){
    svg.selectAll("*").remove();
    const k = d3.select("#mix-set").property("value"), metric = d3.select("#mix-metric").property("value"), M = MIX[k];
    const hasEp = M.rows.every(r=>r.ep);
    const val = r => metric==="share" ? r.p : metric==="seen" ? r.p/100*M.total : (hasEp ? (metric==="epochs" ? r.ep : r.p/100*M.total/r.ep) : r.p/100*M.total);
    const g = svg.append("g"), vmax = d3.max(M.rows, val), xs = d3.scaleLinear().domain([0, vmax*1.05]).range([m.l, W-m.r]), bh = (H-m.t-m.b)/M.rows.length;
    M.rows.forEach((r,i)=>{
      const yy = m.t + i*bh, v = val(r);
      txt(g, m.l-6, yy+bh/2+4, r.s, {anchor:"end", size:10.5});
      g.append("rect").attr("x",m.l).attr("y",yy+3).attr("width",Math.max(1,xs(v)-m.l)).attr("height",bh-6).attr("rx",3).attr("fill",P.A).attr("fill-opacity",.7);
      const lab = metric==="share" ? v.toFixed(1)+"%" : metric==="epochs" && hasEp ? v.toFixed(2)+" ep" : fmtN(v)+" tok";
      txt(g, xs(v)+5, yy+bh/2+4, lab, {size:10, fill:P.muted});
      if (metric==="epochs" && hasEp && v > 1) g.append("rect").attr("x",xs(1)).attr("y",yy+3).attr("width",xs(v)-xs(1)).attr("height",bh-6).attr("fill",P.B).attr("fill-opacity",.6);
    });
    if (metric==="epochs" && hasEp){ g.append("line").attr("x1",xs(1)).attr("x2",xs(1)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.B).attr("stroke-dasharray","3,3"); txt(g, xs(1), H-m.b+14, "1 epoch", {anchor:"middle", size:10, fill:P.B}); }
    txt(g, m.l, 12, M.label, {size:10, fill:P.muted});
    const sum = d3.sum(M.rows, r=>r.p);
    let extra;
    if (hasEp){
      const uniq = d3.sum(M.rows, r=>r.p/100*M.total/r.ep), rep = M.rows.filter(r=>r.ep>1.5).map(r=>r.s).join(" and ");
      extra = `Unique tokens implied by share ÷ epochs: <b>${fmtN(uniq)}</b> of the ${fmtN(M.total)} seen, so ${(100*(1-uniq/M.total)).toFixed(1)}% of training tokens are repeats — mostly ${rep}, the only sources seen about twice. GitHub is the only source seen for less than one epoch.`;
    } else {
      extra = `The report gives only these four category shares, not per-source epochs or sizes; tokens seen per category are the shares applied to the ${fmtN(M.total)} total. "Epochs" and "unique" views fall back to tokens seen.`;
    }
    d3.select("#mix-read").html(`Shares sum to <b>${sum.toFixed(1)}%</b>. Largest source: <b>${M.rows[0].s}</b> at ${M.rows[0].p}% ≈ ${fmtN(M.rows[0].p/100*M.total)} tokens. ${extra}`);
  }
  d3.select("#mix-set").on("change", draw); d3.select("#mix-metric").on("change", draw);
  draw();
});

/* ───────────────────────── 06 · block diff ───────────────────────── */
LV.safe("block", function(){
  const {P, ffnHidden, fmtN, txt} = LV;
  const svg = d3.select("#blk-svg");
  const ROWS = [
    {k:"emb",  gpt:"token emb + learned positions", ll:"token embedding only", diff:true,  why:"LLaMA removed the absolute position table and injects position inside attention with RoPE, so there is no hard row limit at the embedding."},
    {k:"n1",   gpt:"LayerNorm (γ, β)",              ll:"RMSNorm (gain g)",     diff:true,  why:"RMSNorm drops mean-centring and the bias: one d-vector of gains instead of two, and one reduction instead of two."},
    {k:"qkv",  gpt:"Q, K, V with biases",           ll:"Q, K, V, no biases · RoPE on Q, K", diff:true, why:"No bias terms anywhere. Q and K are rotated by position (RoPE) just before the dot product; V is not rotated. From Llama 2 70B on, K and V have fewer heads (GQA)."},
    {k:"att",  gpt:"causal softmax attention",      ll:"causal softmax attention", diff:false, why:"Unchanged: scaled dot-product attention under a causal mask (the mechanics are in the GPT and Attention pages)."},
    {k:"o",    gpt:"output proj + bias, residual add", ll:"output proj (no bias), residual add", diff:true, why:"Same residual structure; only the bias is gone."},
    {k:"n2",   gpt:"LayerNorm (γ, β)",              ll:"RMSNorm (gain g)",     diff:true,  why:"Second pre-norm, again RMSNorm."},
    {k:"ffn",  gpt:"W₂ GELU(W₁x + b₁) + b₂, width 4d", ll:"W₂ (SiLU(W₁x) ⊙ W₃x), width ≈ 8d/3", diff:true, why:"SwiGLU: a gated FFN with three matrices. The width is cut to about two-thirds of 4d so the parameter count stays near 8d², then rounded up."},
    {k:"fin",  gpt:"final LayerNorm",               ll:"final RMSNorm",        diff:true,  why:"Both are pre-norm stacks, so both need a norm before the output."},
    {k:"head", gpt:"output = embedding matrixᵀ (tied)", ll:"separate output matrix (untied)", diff:true, why:"Llama 1–3.1 and Llama 4 keep a separate V×d output head; only Llama 3.2 1B/3B tie it (§10)."}
  ];
  let sel = "ffn";
  function draw(){
    svg.selectAll("*").remove();
    const d = +d3.select("#blk-d").property("value"), only = d3.select("#blk-only").property("checked"), g = svg.append("g");
    txt(g, 150, 16, "GPT-2 / GPT-3 block", {anchor:"middle", size:11, bold:true, fill:P.muted});
    txt(g, 470, 16, "LLaMA block", {anchor:"middle", size:11, bold:true, fill:P.B});
    ROWS.forEach((r,i)=>{
      const yy = 28 + i*28, on = r.k===sel, dim = only && !r.diff;
      const row = g.append("g").style("cursor","pointer").on("click",()=>{ sel=r.k; draw(); });
      row.append("rect").attr("x",20).attr("y",yy).attr("width",260).attr("height",22).attr("rx",5).attr("fill",P.panel).attr("stroke",on?P.B:P.line).attr("opacity",dim?.35:1);
      row.append("rect").attr("x",340).attr("y",yy).attr("width",270).attr("height",22).attr("rx",5).attr("fill",r.diff?"rgba(255,180,84,.12)":P.panel).attr("stroke",on?P.B:(r.diff?P.B:P.line)).attr("opacity",dim?.35:1);
      txt(row, 150, yy+15, r.gpt, {anchor:"middle", size:10, fill:dim?P.muted:P.ink});
      txt(row, 475, yy+15, r.ll, {anchor:"middle", size:10, fill:dim?P.muted:(r.diff?P.B:P.ink)});
      txt(row, 310, yy+15, r.diff ? "≠" : "=", {anchor:"middle", size:12, fill:r.diff?P.B:P.muted});
    });
    const f = ffnHidden(d, null, 256).f;
    const gpt = 12*d*d + 13*d, ll = 4*d*d + 3*d*f + 2*d;
    const r = ROWS.find(z=>z.k===sel), nd = ROWS.filter(z=>z.diff).length;
    d3.select("#blk-read").html(`<b>${r.ll}</b> — ${r.why} <br/>Per block at d = ${d}: GPT-style 12d² + 13d = <b>${fmtN(gpt)}</b>; LLaMA-style 4d² + 3·d·f + 2d with f = ${f} (the LLaMA 1 rounding rule) = <b>${fmtN(ll)}</b>, ${(100*(ll/gpt-1)).toFixed(2)}% ${ll>gpt?"more":"less"}. ${nd} of ${ROWS.length} rows differ.`);
  }
  d3.select("#blk-d").on("change", draw); d3.select("#blk-only").on("change", draw);
  draw();
});

/* ───────────────────────── 07 · RMSNorm vs LayerNorm ───────────────────────── */
LV.safe("rmsnorm", function(){
  const {P, txt} = LV;
  const svg = d3.select("#rms-svg"), x0 = [2,-1,3,0,1,-2];
  function draw(){
    svg.selectAll("*").remove();
    const c = +d3.select("#rms-c").property("value"), s = +d3.select("#rms-s").property("value");
    d3.select("#rms-cv").text(c.toFixed(1)); d3.select("#rms-sv").text(s.toFixed(2));
    const x = x0.map(v=>s*v + c), n = x.length;
    const mean = d3.mean(x), vr = d3.mean(x.map(v=>(v-mean)*(v-mean))), rms = Math.sqrt(d3.mean(x.map(v=>v*v)));
    const ln = x.map(v=>(v-mean)/Math.sqrt(vr + 1e-6)), rn = x.map(v=>v/Math.sqrt(rms*rms + 1e-6));
    const base0 = x0.map(v=>v), m0 = d3.mean(base0), rms0 = Math.sqrt(d3.mean(base0.map(v=>v*v))), rn0 = base0.map(v=>v/rms0);
    const g = svg.append("g"), panels = [["input x = s·x₀ + c", x, P.muted],["LayerNorm(x)", ln, P.A],["RMSNorm(x)", rn, P.B]];
    panels.forEach(([lab, arr, col], k)=>{
      const px = 20 + k*208, w = 190, mid = 150, sc = 26;
      txt(g, px + w/2, 18, lab, {anchor:"middle", size:11, bold:true, fill:col});
      g.append("line").attr("x1",px).attr("x2",px+w).attr("y1",mid).attr("y2",mid).attr("stroke",P.line);
      const lim = 4.6;
      arr.forEach((v,i)=>{
        const vv = Math.max(-lim, Math.min(lim, v)), bw = w/n - 6, bx = px + i*(w/n) + 3;
        g.append("rect").attr("x",bx).attr("y", vv>=0 ? mid - vv*sc : mid).attr("width",bw).attr("height",Math.abs(vv)*sc).attr("rx",2).attr("fill",col).attr("fill-opacity",.75);
        txt(g, bx + bw/2, vv>=0 ? mid - vv*sc - 4 : mid + Math.abs(vv)*sc + 11, v.toFixed(2), {anchor:"middle", size:8.5, fill:P.muted});
        if (Math.abs(v) > lim) txt(g, bx + bw/2, v>0 ? 34 : 272, "↕", {anchor:"middle", size:9, fill:P.bad});
      });
    });
    const shiftInv = ln.every((v,i)=>Math.abs(v - (base0[i]-m0)/Math.sqrt(d3.mean(base0.map(z=>(z-m0)*(z-m0))))) < 1e-3);
    const rmsSame = rn.every((v,i)=>Math.abs(v - rn0[i]) < 1e-3);
    d3.select("#rms-read").html(`mean(x) = <b>${mean.toFixed(3)}</b>, std(x) = ${Math.sqrt(vr).toFixed(3)}, rms(x) = <b>${rms.toFixed(3)}</b>. LayerNorm output ${shiftInv ? "is <b>identical</b> to the c = 0, s = 1 case — it removes both shift and scale" : "differs from the base case"}. RMSNorm output ${rmsSame ? "is <b>identical</b> to the base case — it removes scale" : "<b>changes</b> with the shift c — it removes scale only, not the mean"}. Reductions per token: LayerNorm 2 (mean, variance), RMSNorm 1 (mean of squares); learned vectors: LayerNorm 2 (γ, β), RMSNorm 1 (g).`);
  }
  d3.select("#rms-c").on("input", draw); d3.select("#rms-s").on("input", draw);
  draw();
});

/* ───────────────────────── 08 · SwiGLU hidden size ───────────────────────── */
LV.safe("ffn", function(){
  const {P, CFG, ffnHidden, fmtN, txt} = LV;
  const svg = d3.select("#ffn-svg");
  const pre = ["l1_7","l1_13","l1_33","l1_65","l2_70","l3_8","l3_70","l31_405","l32_1","l32_3"];
  function setPreset(){
    const c = CFG[d3.select("#ffn-preset").property("value")];
    d3.select("#ffn-d").property("value", c.d); d3.select("#ffn-mult").property("value", c.mult==null ? "none" : String(c.mult)); d3.select("#ffn-mo").property("value", String(c.mo));
  }
  function draw(){
    svg.selectAll("*").remove();
    const d = +d3.select("#ffn-d").property("value"), ms = d3.select("#ffn-mult").property("value"), mult = ms==="none" ? null : +ms, mo = +d3.select("#ffn-mo").property("value");
    d3.select("#ffn-dv").text(d);
    const r = ffnHidden(d, mult, mo), g = svg.append("g");
    const steps = [["4·d", r.s1, P.muted], ["⌊2/3 · 4d⌋", r.s2, P.A], [mult ? "⌊"+mult+" × ·⌋" : "(no multiplier)", r.s3, P.A], ["round up to ×"+mo, r.f, P.B]];
    steps.forEach((s,i)=>{
      const bx = 18 + i*155;
      g.append("rect").attr("x",bx).attr("y",26).attr("width",136).attr("height",52).attr("rx",7).attr("fill",P.panel).attr("stroke",s[2]);
      txt(g, bx+68, 46, s[0], {anchor:"middle", size:10.5, fill:P.muted});
      txt(g, bx+68, 67, s[1].toLocaleString("en-US"), {anchor:"middle", size:14, bold:true, fill:s[2], mono:true});
      if (i) txt(g, bx-10, 58, "→", {anchor:"middle", size:14, fill:P.muted});
    });
    /* parameter comparison bars */
    const gelu = 8*d*d, sw = 3*d*r.f, ideal = 3*d*(8*d/3), mx = Math.max(gelu, sw)*1.08, xs = d3.scaleLinear().domain([0,mx]).range([0, 430]);
    [["GELU FFN, 2 × d × 4d", gelu, P.muted],["SwiGLU at exactly 8d/3", ideal, P.A],["SwiGLU as built, 3 × d × f", sw, P.B]].forEach((b,i)=>{
      const yy = 110 + i*40;
      txt(g, 18, yy+14, b[0], {size:10.5});
      g.append("rect").attr("x",190).attr("y",yy).attr("width",xs(b[1])).attr("height",20).attr("rx",3).attr("fill",b[2]).attr("fill-opacity",.75);
      txt(g, 196 + xs(b[1]), yy+14, fmtN(b[1]) + " per layer", {size:10, fill:P.muted});
    });
    const hit = pre.map(k=>CFG[k]).filter(c=>c.d===d && (c.mult==null ? mult==null : c.mult===mult) && c.mo===mo);
    const check = hit.length ? `Released config${hit.length>1?"s":""} with these inputs (${hit.map(c=>c.name).join(", ")}): intermediate_size = <b>${hit[0].f.toLocaleString("en-US")}</b> — ${hit.every(c=>c.f===r.f) ? "<b style='color:"+P.good+"'>matches</b> the rule" : "<b style='color:"+P.bad+"'>does not match</b>"}.` : "No released model uses exactly these three inputs.";
    d3.select("#ffn-read").html(`f = <b>${r.f.toLocaleString("en-US")}</b> = ${(r.f/d).toFixed(3)}·d (8/3 = 2.667). SwiGLU parameters per layer are ${(100*sw/gelu).toFixed(1)}% of a 4d GELU FFN; rounding added ${(r.f - Math.round(8*d/3)).toLocaleString("en-US")} columns over 8d/3. ${check}`);
  }
  d3.select("#ffn-preset").on("change", ()=>{ setPreset(); draw(); });
  ["#ffn-d"].forEach(s=>d3.select(s).on("input", draw));
  ["#ffn-mult","#ffn-mo"].forEach(s=>d3.select(s).on("change", draw));
  setPreset(); draw();
});

/* ───────────────────────── 09 · RoPE by hand ───────────────────────── */
LV.safe("rope", function(){
  const {P, txt} = LV;
  const svg = d3.select("#rope-svg"), q0 = [1.0, 0.3], k0 = [0.8, -0.5];
  let off = 0;
  function rot(v, a){ return [v[0]*Math.cos(a) - v[1]*Math.sin(a), v[0]*Math.sin(a) + v[1]*Math.cos(a)]; }
  function draw(){
    svg.selectAll("*").remove();
    const i = +d3.select("#rope-i").property("value"), base = +d3.select("#rope-b").property("value"), hd = 128;
    const m = +d3.select("#rope-m").property("value") + off, n = +d3.select("#rope-n").property("value") + off;
    d3.select("#rope-iv").text(i); d3.select("#rope-mv").text(m); d3.select("#rope-nv").text(n);
    const w = Math.pow(base, -2*i/hd), q = rot(q0, m*w), k = rot(k0, n*w);
    const dot = q[0]*k[0] + q[1]*k[1];
    const rel = rot(q0, (m-n)*w), dotRel = rel[0]*k0[0] + rel[1]*k0[1];
    const g = svg.append("g"), cx = 170, cy = 150, R = 110;
    g.append("circle").attr("cx",cx).attr("cy",cy).attr("r",R).attr("fill","none").attr("stroke",P.line);
    g.append("line").attr("x1",cx-R-8).attr("x2",cx+R+8).attr("y1",cy).attr("y2",cy).attr("stroke",P.line);
    g.append("line").attr("x1",cx).attr("x2",cx).attr("y1",cy-R-8).attr("y2",cy+R+8).attr("stroke",P.line);
    function arrow(v, col, lab, dash){
      const sc = R/1.2;
      g.append("line").attr("x1",cx).attr("y1",cy).attr("x2",cx+v[0]*sc).attr("y2",cy-v[1]*sc).attr("stroke",col).attr("stroke-width",2.5).attr("stroke-dasharray",dash||null);
      g.append("circle").attr("cx",cx+v[0]*sc).attr("cy",cy-v[1]*sc).attr("r",3.5).attr("fill",col);
      txt(g, cx+v[0]*sc*1.12, cy-v[1]*sc*1.12+4, lab, {anchor:"middle", size:10.5, fill:col, bold:true});
    }
    arrow(q0, P.muted, "q", "3,3"); arrow(k0, P.muted, "k", "3,3");
    arrow(q, P.A, "R(mω)q"); arrow(k, P.B, "R(nω)k");
    txt(g, cx, 18, `pair i = ${i} of ${hd/2} (head dim ${hd})`, {anchor:"middle", size:10.5, fill:P.muted});
    /* right: the dot product as a function of offset m − n, for this pair */
    const x0 = 330, x1 = 620, y0 = 250, y1 = 40, span = 64;
    const xs = d3.scaleLinear().domain([-span, span]).range([x0, x1]), ys = d3.scaleLinear().domain([-1.2, 1.2]).range([y0, y1]);
    g.append("line").attr("x1",x0).attr("x2",x1).attr("y1",ys(0)).attr("y2",ys(0)).attr("stroke",P.line);
    g.append("line").attr("x1",xs(0)).attr("x2",xs(0)).attr("y1",y0).attr("y2",y1).attr("stroke",P.line);
    const pts = d3.range(-span, span+0.5, 0.5).map(o=>{ const r2 = rot(q0, o*w); return [o, r2[0]*k0[0] + r2[1]*k0[1]]; });
    g.append("path").datum(pts).attr("fill","none").attr("stroke",P.A).attr("stroke-width",1.8).attr("d", d3.line().x(p=>xs(p[0])).y(p=>ys(p[1])));
    const o = Math.max(-span, Math.min(span, m-n));
    g.append("circle").attr("cx",xs(o)).attr("cy",ys(dotRel)).attr("r",5).attr("fill",P.B);
    txt(g, (x0+x1)/2, y0+22, "relative offset m − n", {anchor:"middle", size:10, fill:P.muted});
    txt(g, x0, y1-12, "q·k after rotation, this pair only", {size:10, fill:P.muted});
    [-span,-32,0,32,span].forEach(v=>txt(g, xs(v), y0+11, String(v), {anchor:"middle", size:9, fill:P.muted}));
    const wl = 2*Math.PI/w;
    d3.select("#rope-read").html(`ω<sub>${i}</sub> = ${base.toLocaleString("en-US")}<sup>−2·${i}/${hd}</sup> = <b>${w.toExponential(3)}</b> rad/token, wavelength 2π/ω = <b>${wl < 1e4 ? wl.toFixed(1) : wl.toExponential(2)}</b> tokens. Rotated dot product computed from absolute positions (m = ${m}, n = ${n}): <b>${dot.toFixed(6)}</b>; computed from the offset alone (m − n = ${m-n}): <b>${dotRel.toFixed(6)}</b> — ${Math.abs(dot-dotRel)<1e-9 ? "equal" : "different"}. ${off ? "Both positions shifted by +"+off+"; the score did not change." : ""}`);
  }
  ["#rope-i","#rope-m","#rope-n"].forEach(s=>d3.select(s).on("input", ()=>{ draw(); }));
  d3.select("#rope-b").on("change", draw);
  d3.select("#rope-shift").on("click", ()=>{ off = off ? 0 : 10; draw(); });
  draw();
});

/* ───────────────────────── 20 · llama3 RoPE scaling ───────────────────────── */
LV.safe("ropescale", function(){
  const {P, invFreq, llama3Scale, txt} = LV;
  const svg = d3.select("#rsc-svg"), W = 640, H = 300, m = {l:62, r:20, t:22, b:40};
  const SC = {
    l31:  {factor:8,  lo:1, hi:4, orig:8192, label:"Llama 3.1 (factor 8, low 1, high 4)"},
    l32:  {factor:32, lo:1, hi:4, orig:8192, label:"Llama 3.2 1B/3B (factor 32)"},
    scout:{factor:16, lo:1, hi:1, orig:8192, label:"Llama 4 Scout (factor 16, low = high = 1)"},
    none: null
  };
  function draw(){
    svg.selectAll("*").remove();
    const key = d3.select("#rsc-set").property("value"), theta = +d3.select("#rsc-theta").property("value"), hd = 128, sc = SC[key];
    const inv = invFreq(theta, hd), out = llama3Scale(inv, sc), g = svg.append("g");
    const wl = inv.map(f=>2*Math.PI/f), wl2 = out.map(o=>2*Math.PI/o.v);
    const xs = d3.scaleLinear().domain([0, hd/2-1]).range([m.l, W-m.r]), ys = d3.scaleLog().domain([1, 1e8]).range([H-m.b, m.t]);
    [1,1e2,1e4,1e6,1e8].forEach(v=>{ g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",ys(v)).attr("y2",ys(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, m.l-6, ys(v)+4, v>=1e4 ? "10"+LV.sci(v,0).split("10")[1] : String(v), {anchor:"end", size:9.5, fill:P.muted}); });
    [0,16,32,48,63].forEach(v=>txt(g, xs(v), H-m.b+14, String(v), {anchor:"middle", size:9.5, fill:P.muted}));
    txt(g, (m.l+W-m.r)/2, H-8, "frequency pair index i (0 = fastest rotation)", {anchor:"middle", size:10, fill:P.muted});
    txt(g, 8, m.t-6, "wavelength (tokens, log)", {size:10, fill:P.muted});
    if (sc){
      const hiW = sc.orig/sc.hi, loW = sc.orig/sc.lo;
      [[hiW, "high-freq cutoff "+hiW], [loW, "low-freq cutoff "+loW]].forEach(([v,lab],j)=>{
        g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",ys(v)).attr("y2",ys(v)).attr("stroke",P.good).attr("stroke-dasharray","5,3");
        txt(g, W-m.r-4, ys(v) + (j ? -4 : 12), lab, {anchor:"end", size:9.5, fill:P.good});
      });
    }
    g.append("path").datum(wl).attr("fill","none").attr("stroke",P.muted).attr("stroke-width",2).attr("stroke-dasharray","4,3").attr("d", d3.line().x((v,i)=>xs(i)).y(v=>ys(v)));
    out.forEach((o,i)=>{ g.append("circle").attr("cx",xs(i)).attr("cy",ys(wl2[i])).attr("r",2.8)
      .attr("fill", o.band==="keep" ? P.A : o.band==="blend" ? P.purple : o.band==="divide" ? P.B : P.muted); });
    const cnt = b => out.filter(o=>o.band===b).length;
    const legend = [["kept",P.A],["blended",P.purple],["÷ factor",P.B],["original",P.muted]];
    legend.forEach((l,j)=>{ g.append("circle").attr("cx",m.l+10+j*90).attr("cy",m.t+6).attr("r",4).attr("fill",l[1]); txt(g, m.l+18+j*90, m.t+10, l[0], {size:9.5, fill:P.muted}); });
    const maxBefore = d3.max(wl), maxAfter = d3.max(wl2);
    d3.select("#rsc-read").html(sc
      ? `<b>${sc.label}</b>, base ${theta.toLocaleString("en-US")}: of ${hd/2} frequency pairs, <b>${cnt("keep")}</b> are left unchanged (wavelength &lt; ${sc.orig/sc.hi}), <b>${cnt("blend")}</b> are blended, and <b>${cnt("divide")}</b> are divided by ${sc.factor} (wavelength &gt; ${sc.orig/sc.lo}). Longest wavelength ${Math.round(maxBefore).toLocaleString("en-US")} → <b>${Math.round(maxAfter).toLocaleString("en-US")}</b> tokens. Local, fast-rotating pairs keep their trained resolution; only the slow pairs are stretched to cover ${sc.factor}× the original ${sc.orig.toLocaleString("en-US")}-token window.`
      : `No rescaling, base ${theta.toLocaleString("en-US")}: wavelengths run from ${wl[0].toFixed(2)} to <b>${Math.round(maxBefore).toLocaleString("en-US")}</b> tokens. ${cnt("none")} pairs, each rotating at its trained speed; ${wl.filter(v=>v>8192).length} of them have a wavelength longer than 8,192 tokens.`);
  }
  d3.select("#rsc-set").on("change", draw); d3.select("#rsc-theta").on("change", draw);
  draw();
});

/* ───────────────────────── 13 · parameter calculator ───────────────────────── */
LV.safe("params", function(){
  const {P, CFG, params, fmtN, txt} = LV;
  const svg = d3.select("#pc-svg");
  function draw(){
    svg.selectAll("*").remove();
    const c = CFG[d3.select("#pc-preset").property("value")], p = params(c), g = svg.append("g");
    const parts = [["input embedding", p.emb, P.muted], ["output head", p.head, P.pink], ["attention", p.attn, P.A], [c.E ? "FFN + experts + router" : "SwiGLU FFN", p.mlp + p.router, P.B], ["RMSNorm gains", p.norm, P.good]];
    const xs = d3.scaleLinear().domain([0, p.total]).range([20, 620]);
    let acc = 0;
    txt(g, 20, 20, `${c.name}: L = ${c.L}, d = ${c.d}, heads ${c.h} / KV ${c.kv}, head dim ${c.hd}, FFN ${c.E ? c.fe+" per expert ("+c.E+" experts)" : c.f}, V = ${c.V.toLocaleString("en-US")}${c.tied?", tied":""}`, {size:10.5, fill:P.muted});
    parts.forEach((q,i)=>{
      const x = xs(acc), w = xs(acc+q[1]) - x;
      g.append("rect").attr("x",x).attr("y",36).attr("width",Math.max(0,w)).attr("height",34).attr("fill",q[2]).attr("fill-opacity",.8).attr("stroke","#0b0d12");
      acc += q[1];
      const ly = 96 + i*22;
      g.append("rect").attr("x",20).attr("y",ly-10).attr("width",12).attr("height",12).attr("fill",q[2]);
      txt(g, 40, ly, q[0], {size:10.5});
      txt(g, 230, ly, fmtN(q[1]), {size:10.5, mono:true, anchor:"end"});
      txt(g, 290, ly, (100*q[1]/p.total).toFixed(q[1]/p.total<0.001?3:1)+"%", {size:10.5, mono:true, anchor:"end", fill:P.muted});
    });
    if (c.E){
      const xa = xs(p.active);
      g.append("line").attr("x1",xa).attr("x2",xa).attr("y1",30).attr("y2",76).attr("stroke",P.ink).attr("stroke-width",2).attr("stroke-dasharray","3,2");
      txt(g, Math.min(xa+4, 540), 84, "active ≈ " + fmtN(p.active), {size:10, fill:P.ink});
    }
    txt(g, 340, 110, "per layer", {size:10.5, bold:true, fill:P.muted});
    txt(g, 340, 130, `attention: ${fmtN(p.attnL)} (Q,O ${fmtN(2*c.d*c.h*c.hd)} · K,V ${fmtN(2*c.d*c.kv*c.hd)})`, {size:10.5});
    txt(g, 340, 152, c.E ? `routed expert: ${fmtN(3*c.d*c.fe)} each · dense FFN: ${fmtN(3*c.d*c.f)}` : `SwiGLU FFN: ${fmtN(3*c.d*c.f)}`, {size:10.5});
    txt(g, 340, 174, `norm gains: ${(2*c.d).toLocaleString("en-US")}`, {size:10.5});
    const diff = 100*(p.total/c.rep - 1);
    d3.select("#pc-read").html(`total <b>${fmtN(p.total)}</b> (${p.total.toLocaleString("en-US")}) against the label <b>${c.repLabel}</b> — ${diff>=0?"+":""}${diff.toFixed(1)}%. Non-embedding ${fmtN(p.nonEmb)}; embeddings (input${c.tied?"":" + head"}) are ${(100*(p.emb+p.head)/p.total).toFixed(1)}% of the model.${c.E ? " Active per token (all attention, the shared expert and "+c.top+" routed expert per MoE layer, both embedding matrices): <b>"+fmtN(p.active)+"</b>; without the input lookup "+fmtN(p.active - p.emb)+"." : ""}`);
  }
  d3.select("#pc-preset").on("change", draw);
  draw();
});

/* ───────────────────────── 14 · GQA grouping ───────────────────────── */
LV.safe("gqa", function(){
  const {P, CFG, kvPerToken, bytes, fmtN, txt} = LV;
  const svg = d3.select("#gqa-svg");
  function divisors(h){ const r=[]; for (let k=1;k<=h;k++) if (h%k===0) r.push(k); return r; }
  function sync(){
    const c = CFG[d3.select("#gqa-model").property("value")], dv = divisors(c.h), idx = dv.indexOf(c.kv);
    d3.select("#gqa-k").attr("max", dv.length-1).property("value", idx);
  }
  function draw(){
    svg.selectAll("*").remove();
    const c = CFG[d3.select("#gqa-model").property("value")], dv = divisors(c.h);
    const idx = Math.max(0, Math.min(dv.length-1, +d3.select("#gqa-k").property("value"))), kv = dv[idx], grp = c.h/kv;
    d3.select("#gqa-kv").text(kv);
    const g = svg.append("g"), W = 600, x0 = 20, qw = W/c.h, kw = W/kv;
    txt(g, x0, 18, `${c.h} query heads`, {size:10.5, fill:P.muted});
    txt(g, x0, 168, `${kv} key/value head${kv>1?"s":""} — each shared by ${grp} query head${grp>1?"s":""}`, {size:10.5, fill:P.muted});
    const pal = [P.A, P.B, P.good, P.purple, P.pink, P.teal];
    for (let i=0;i<c.h;i++){
      const grpIdx = Math.floor(i/grp), col = pal[grpIdx % pal.length];
      g.append("rect").attr("x",x0 + i*qw + 0.5).attr("y",26).attr("width",Math.max(1,qw-1)).attr("height",30).attr("fill",col).attr("fill-opacity",.75);
      g.append("line").attr("x1",x0 + i*qw + qw/2).attr("y1",56).attr("x2",x0 + grpIdx*kw + kw/2).attr("y2",136).attr("stroke",col).attr("stroke-opacity",.35);
    }
    for (let j=0;j<kv;j++){
      g.append("rect").attr("x",x0 + j*kw + 1).attr("y",136).attr("width",Math.max(1,kw-2)).attr("height",18).attr("fill",pal[j % pal.length]).attr("stroke","#0b0d12");
    }
    const cc = Object.assign({}, c, {kv}), per = kvPerToken(cc, 2), mha = kvPerToken(Object.assign({}, c, {kv:c.h}), 2);
    const attnL = 2*c.d*c.h*c.hd + 2*c.d*kv*c.hd, attnMHA = 4*c.d*c.h*c.hd;
    const at4k = per*4096, at128k = per*131072;
    d3.select("#gqa-read").html(`<b>${c.name}</b> with ${kv} KV head${kv>1?"s":""}${kv===c.kv?" (as released)":""}: KV cache <b>${bytes(per)}</b> per token at 16 bits (2 · L · n_kv · d_head · 2 bytes), ${(mha/per).toFixed(0)}× smaller than full multi-head (${bytes(mha)}); one 4,096-token sequence ${bytes(at4k)}, one 131,072-token sequence ${bytes(at128k)}. Attention weights per layer: ${fmtN(attnL)} vs ${fmtN(attnMHA)} for MHA (${(100*(1-attnL/attnMHA)).toFixed(1)}% fewer).`);
  }
  d3.select("#gqa-model").on("change", ()=>{ sync(); draw(); });
  d3.select("#gqa-k").on("input", draw);
  sync(); draw();
});

/* ───────────────────────── 15 · KV cache vs context ───────────────────────── */
LV.safe("kvcache", function(){
  const {P, CFG, kvBytes, kvPerToken, bytes, ctxLabel, txt} = LV;
  const svg = d3.select("#kv-svg"), W = 640, H = 290, m = {l:62, r:130, t:18, b:38};
  const ids = ["l1_65","l2_70","l31_8","l31_70","l31_405","l32_1","l4_s"], cols = [P.bad, P.B, P.good, P.A, P.purple, P.teal, P.pink];
  const CT = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
  function draw(){
    svg.selectAll("*").remove();
    const by = +d3.select("#kv-bytes").property("value"), b = +d3.select("#kv-b").property("value"), n = CT[+d3.select("#kv-n").property("value")], hyb = d3.select("#kv-hyb").property("checked");
    d3.select("#kv-bv").text(b); d3.select("#kv-nv").text(ctxLabel(n));
    const g = svg.append("g"), xs = d3.scaleLog().base(2).domain([2048, 1048576]).range([m.l, W-m.r]), ys = d3.scaleLog().domain([1e8, 2e14]).range([H-m.b, m.t]);
    [1e9,1e10,1e11,1e12,1e13,1e14].forEach(v=>{ g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",ys(v)).attr("y2",ys(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, m.l-6, ys(v)+4, bytes(v), {anchor:"end", size:9.5, fill:P.muted}); });
    CT.forEach(v=>txt(g, xs(v), H-m.b+14, ctxLabel(v), {anchor:"middle", size:9, fill:P.muted}));
    txt(g, (m.l+W-m.r)/2, H-6, "context length (tokens)", {anchor:"middle", size:10, fill:P.muted});
    g.append("line").attr("x1",xs(n)).attr("x2",xs(n)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.ink).attr("stroke-dasharray","3,3");
    const at = [];
    ids.forEach((id,j)=>{
      const c = CFG[id], pts = CT.map(t=>[t, b*kvBytes(c, t, by, hyb)]);
      g.append("path").datum(pts).attr("fill","none").attr("stroke",cols[j]).attr("stroke-width",1.8).attr("d", d3.line().x(p=>xs(p[0])).y(p=>ys(Math.max(1e8,p[1]))));
      pts.forEach(p=>{ if (p[0] > c.ctx) g.append("circle").attr("cx",xs(p[0])).attr("cy",ys(Math.max(1e8,p[1]))).attr("r",2).attr("fill","#0b0d12").attr("stroke",cols[j]); });
      txt(g, W-m.r+6, m.t+12+j*16, c.name, {size:9.5, fill:cols[j]});
      at.push({c, v:b*kvBytes(c, n, by, hyb), per:kvPerToken(c, by)});
    });
    at.sort((a,z)=>z.v-a.v);
    d3.select("#kv-read").html(`At ${n.toLocaleString("en-US")} tokens × batch ${b}, ${by===2?"16-bit":"8-bit"} cache: ` + at.map(a=>`${a.c.name} <b>${bytes(a.v)}</b>${n > a.c.ctx ? " (beyond its "+ctxLabel(a.c.ctx)+" context)" : ""}`).join(" · ") + `. Per token: ${at.map(a=>a.c.name.replace("Llama ","")+" "+bytes(a.per)).join(", ")}${hyb ? "; Scout's 36 chunked-attention layers stop growing at 8,192 tokens" : ""}.`);
  }
  ["#kv-bytes"].forEach(s=>d3.select(s).on("change", draw));
  ["#kv-b","#kv-n"].forEach(s=>d3.select(s).on("input", draw));
  d3.select("#kv-hyb").on("change", draw);
  draw();
});

/* ───────────────────────── 16 · training compute ───────────────────────── */
LV.safe("compute", function(){
  const {P, CFG, params, fmtN, sci, txt} = LV;
  const svg = d3.select("#cmp-svg"), W = 640, H = 320, m = {l:118, r:90, t:14, b:34};
  const ids = ["l1_7","l1_13","l1_33","l1_65","l2_7","l2_13","l2_70","l31_8","l31_70","l31_405","l4_s","l4_m"];
  function draw(){
    svg.selectAll("*").remove();
    const metric = d3.select("#cmp-metric").property("value"), g = svg.append("g");
    const R = ids.map(id=>{ const c = CFG[id], p = params(c), N = c.E ? p.active : p.total, F = 6*N*c.tok;
      const tfl = c.gpuh ? F/(c.gpuh*3600) : null; return {c, N, F, gpuh:c.gpuh, tfl, mfu: tfl ? tfl/c.peak : null}; });
    const val = r => metric==="flops" ? r.F : metric==="gpuh" ? r.gpuh : metric==="tfl" ? r.tfl : r.mfu;
    const vals = R.map(val).filter(v=>v!=null), lin = metric==="tfl" || metric==="mfu";
    const xs = lin ? d3.scaleLinear().domain([0, d3.max(vals)*1.1]).range([m.l, W-m.r]) : d3.scaleLog().domain([d3.min(vals)/2, d3.max(vals)*2]).range([m.l, W-m.r]);
    const bh = (H-m.t-m.b)/R.length;
    R.forEach((r,i)=>{
      const yy = m.t + i*bh, v = val(r);
      txt(g, m.l-6, yy+bh/2+4, r.c.name, {anchor:"end", size:10});
      if (v==null){ txt(g, m.l+4, yy+bh/2+4, "not reported", {size:9.5, fill:P.muted}); return; }
      const x0 = lin ? xs(0) : m.l;
      g.append("rect").attr("x",x0).attr("y",yy+3).attr("width",Math.max(1, xs(v)-x0)).attr("height",bh-6).attr("rx",3).attr("fill", r.c.peak===312e12 ? P.A : P.B).attr("fill-opacity",.75);
      const lab = metric==="flops" ? sci(v,2) : metric==="gpuh" ? fmtN(v)+" h" : metric==="tfl" ? (v/1e12).toFixed(0)+" TFLOP/s" : (100*v).toFixed(0)+"%";
      txt(g, xs(v)+5, yy+bh/2+4, lab, {size:9.5, fill:P.muted});
    });
    txt(g, W-m.r+4, H-m.b+22, "blue: A100 · orange: H100", {size:9.5, fill:P.muted, anchor:"end"});
    const b = R.find(r=>r.c===CFG.l31_405), l2 = R.find(r=>r.c===CFG.l2_70);
    d3.select("#cmp-read").html(`6·N·D for Llama 3.1 405B = 6 × ${fmtN(b.N)} × ${fmtN(b.c.tok)} = <b>${sci(b.F)}</b> FLOPs (the report: 3.8 × 10²⁵), ${(b.F/l2.F).toFixed(0)}× Llama 2 70B's ${sci(l2.F)}. Implied sustained throughput from reported GPU-hours: Llama 2 70B ${(l2.tfl/1e12).toFixed(0)} TFLOP/s per A100 (${(100*l2.mfu).toFixed(0)}% of the 312 TFLOP/s bf16 peak); 405B ${(b.tfl/1e12).toFixed(0)} TFLOP/s per H100 (${(100*b.mfu).toFixed(0)}% of 989). GPU-hours include restarts, evaluation and the long-context stage, so these are lower bounds on the achieved rate during steady training. MoE rows use active parameters.`);
  }
  d3.select("#cmp-metric").on("change", draw);
  draw();
});

/* ───────────────────────── 18 · rejection sampling (toy) ───────────────────────── */
LV.safe("rejection", function(){
  const {P, lcg, gauss, txt} = LV;
  const svg = d3.select("#rs-svg"), W = 640, H = 280, m = {l:56, r:20, t:20, b:40}, NP = 300, KMAX = 100;
  /* toy reward model: prompt difficulty b_p; a sample's reward = b_p + mean(T) + spread(T)·z  (illustrative only) */
  const r0 = lcg(7), base = d3.range(NP).map(()=>0.55 + 0.04*gauss(r0));
  const Z = d3.range(NP).map((_,p)=>{ const r = lcg(1000+p); return d3.range(KMAX).map(()=>gauss(r)); });
  function curves(T){
    const mu = -0.12*(T-0.8)*(T-0.8), sd = 0.02 + 0.05*T, mx = [], md = [];
    for (let K=1; K<=KMAX; K++){
      let sMax = 0, sMed = 0;
      for (let p=0;p<NP;p++){
        const a = Z[p].slice(0,K).map(z=>base[p] + mu + sd*z).sort((x,y)=>x-y);
        sMax += a[K-1]; sMed += (K%2 ? a[(K-1)/2] : (a[K/2-1]+a[K/2])/2);
      }
      mx.push(sMax/NP); md.push(sMed/NP);
    }
    return {mx, md};
  }
  const cache = {};
  function draw(){
    svg.selectAll("*").remove();
    const K = +d3.select("#rs-k").property("value"), T = +d3.select("#rs-t").property("value");
    d3.select("#rs-kv").text(K); d3.select("#rs-tv").text(T.toFixed(1));
    const cv = cache[T] || (cache[T] = curves(T)), g = svg.append("g");
    const xs = d3.scaleLog().domain([1, KMAX]).range([m.l, W-m.r]), ys = d3.scaleLinear().domain([0.4, 0.85]).range([H-m.b, m.t]);
    [1,2,5,10,20,50,100].forEach(v=>txt(g, xs(v), H-m.b+14, String(v), {anchor:"middle", size:9.5, fill:P.muted}));
    ys.ticks(5).forEach(v=>{ g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",ys(v)).attr("y2",ys(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3"); txt(g, m.l-6, ys(v)+4, v.toFixed(2), {anchor:"end", size:9.5, fill:P.muted}); });
    txt(g, (m.l+W-m.r)/2, H-8, "samples per prompt K (log)", {anchor:"middle", size:10, fill:P.muted});
    const line = d3.line().x((v,i)=>xs(i+1)).y(v=>ys(v));
    g.append("path").datum(cv.mx).attr("fill","none").attr("stroke",P.B).attr("stroke-width",2).attr("d", line);
    g.append("path").datum(cv.md).attr("fill","none").attr("stroke",P.A).attr("stroke-width",2).attr("d", line);
    g.append("line").attr("x1",xs(K)).attr("x2",xs(K)).attr("y1",ys(cv.md[K-1])).attr("y2",ys(cv.mx[K-1])).attr("stroke",P.good).attr("stroke-width",3);
    txt(g, m.l+8, m.t+10, "max reward over K (kept for fine-tuning)", {size:10, fill:P.B});
    txt(g, m.l+8, m.t+24, "median reward over K", {size:10, fill:P.A});
    const gain = cv.mx[K-1] - cv.md[K-1];
    d3.select("#rs-read").html(`Toy reward model, ${NP} prompts, temperature ${T.toFixed(1)}: at K = ${K} the best-of-K sample averages <b>${cv.mx[K-1].toFixed(3)}</b>, the median <b>${cv.md[K-1].toFixed(3)}</b> — a potential gain of <b>${gain.toFixed(3)}</b> for rejection-sampling fine-tuning. In this toy the median peaks at temperature 0.8 while the spread keeps growing with temperature, so for large K the best-of-K reward is higher at a temperature above 0.8 — the qualitative pattern in the Llama 2 paper's Figure 8 (its numbers are not reproduced here).`);
  }
  d3.select("#rs-k").on("input", draw); d3.select("#rs-t").on("input", draw);
  draw();
});

/* ───────────────────────── 24 · Llama 4 layer map ───────────────────────── */
LV.safe("llama4", function(){
  const {P, CFG, params, isMoe, isNope, kvBytes, fmtN, bytes, ctxLabel, txt} = LV;
  const svg = d3.select("#l4-svg");
  const CT = [8192, 32768, 131072, 262144, 1048576, 10485760];
  function draw(){
    svg.selectAll("*").remove();
    const c = CFG[d3.select("#l4-model").property("value")], n = CT[+d3.select("#l4-n").property("value")], p = params(c), g = svg.append("g");
    d3.select("#l4-nv").text(ctxLabel(n));
    const cw = 600/c.L;
    txt(g, 20, 16, `${c.name}: ${c.L} layers, d = ${c.d}, ${c.h} query / ${c.kv} KV heads`, {size:10.5, fill:P.muted});
    let nMoe = 0, nNope = 0;
    for (let i=0;i<c.L;i++){
      const moe = isMoe(c,i), nope = isNope(c,i); if (moe) nMoe++; if (nope) nNope++;
      const x = 20 + i*cw;
      g.append("rect").attr("x",x+0.5).attr("y",30).attr("width",cw-1).attr("height",22).attr("fill", nope ? P.good : P.A).attr("fill-opacity",.8);
      g.append("rect").attr("x",x+0.5).attr("y",56).attr("width",cw-1).attr("height",34).attr("fill", moe ? P.B : P.muted).attr("fill-opacity", moe ? .85 : .5);
    }
    txt(g, 20, 106, "top row: attention — blue = RoPE, chunked to 8,192 tokens · green = NoPE, global (every 4th layer)", {size:9.5, fill:P.muted});
    txt(g, 20, 121, "bottom row: feed-forward — orange = MoE (routed + shared expert) · grey = dense SwiGLU", {size:9.5, fill:P.muted});
    /* parameter bars */
    const bars = [["total", p.total, P.B], ["active / token", p.active, P.A]], xs = d3.scaleLog().domain([1e9, 5e11]).range([120, 600]);
    bars.forEach((b,i)=>{ const yy = 140 + i*28; txt(g, 112, yy+14, b[0], {anchor:"end", size:10.5});
      g.append("rect").attr("x",120).attr("y",yy).attr("width",xs(b[1])-120).attr("height",20).attr("rx",3).attr("fill",b[2]).attr("fill-opacity",.75);
      txt(g, xs(b[1])+5, yy+14, fmtN(b[1]), {size:10, fill:P.muted}); });
    const full = kvBytes(c, n, 2, false), hyb = kvBytes(c, n, 2, true);
    const kb = [["KV, no chunking", full, P.bad], ["KV, hybrid", hyb, P.good]], ks = d3.scaleLog().domain([1e8, 1e13]).range([120, 600]);
    kb.forEach((b,i)=>{ const yy = 206 + i*28; txt(g, 112, yy+14, b[0], {anchor:"end", size:10.5});
      g.append("rect").attr("x",120).attr("y",yy).attr("width",Math.max(1, ks(Math.max(1e8,b[1]))-120)).attr("height",20).attr("rx",3).attr("fill",b[2]).attr("fill-opacity",.75);
      txt(g, ks(Math.max(1e8,b[1]))+5, yy+14, bytes(b[1]), {size:10, fill:P.muted}); });
    d3.select("#l4-read").html(`<b>${c.name}</b>: ${nMoe} MoE layers (${c.E} routed experts, 1 active, plus 1 shared) and ${c.L-nMoe} dense layers; ${nNope} NoPE layers. Recomputed: <b>${fmtN(p.total)}</b> total, <b>${fmtN(p.active)}</b> active (text decoder only; the vision encoder is extra) against the published ${c.repLabel}. At ${n.toLocaleString("en-US")} tokens one sequence's 16-bit KV cache is ${bytes(full)} if every layer caches everything, <b>${bytes(hyb)}</b> when the ${c.L-nNope} RoPE layers keep only the last 8,192 tokens${n > c.ctx ? " — beyond this model's "+ctxLabel(c.ctx)+" limit" : ""}.`);
  }
  d3.select("#l4-model").on("change", draw); d3.select("#l4-n").on("input", draw);
  draw();
});

/* ───────────────────────── 26 · quantised memory ───────────────────────── */
LV.safe("quant", function(){
  const {P, CFG, params, kvBytes, gb, txt} = LV;
  const svg = d3.select("#q-svg"), W = 640, H = 250;
  /* bits per weight from the storage layout: block of 32 weights + one fp16 scale (Q8_0, Q4_0); int4 with a bf16 scale per 128 */
  const FMT = { bf16:{bpw:16, label:"bf16"}, fp8:{bpw:8, label:"fp8 (per-tensor scale)"}, q8:{bpw:(32*8+16)/32, label:"GGUF Q8_0"}, q4g:{bpw:4+16/128, label:"int4, group 128"}, q4:{bpw:(32*4+16)/32, label:"GGUF Q4_0"} };
  function draw(){
    svg.selectAll("*").remove();
    const c = CFG[d3.select("#q-model").property("value")], f = FMT[d3.select("#q-fmt").property("value")], n = +d3.select("#q-n").property("value");
    d3.select("#q-nv").text(n.toLocaleString("en-US"));
    const p = params(c), wB = p.total*f.bpw/8, kB = kvBytes(c, n, 2, !!c.E), tot = wB + kB, g = svg.append("g");
    const xmax = Math.max(tot, 90e9)*1.08, xs = d3.scaleLinear().domain([0, xmax]).range([30, 610]);
    g.append("rect").attr("x",xs(0)).attr("y",60).attr("width",xs(wB)-xs(0)).attr("height",40).attr("fill",P.A).attr("fill-opacity",.8);
    g.append("rect").attr("x",xs(wB)).attr("y",60).attr("width",Math.max(0,xs(tot)-xs(wB))).attr("height",40).attr("fill",P.B).attr("fill-opacity",.8);
    txt(g, 30, 44, `${c.name} · ${f.label} = ${f.bpw.toFixed(3)} bits per weight`, {size:11, bold:true});
    [[24e9,"24 GB"],[48e9,"48 GB"],[80e9,"80 GB"]].forEach(([v,lab])=>{ if (v < xmax){ g.append("line").attr("x1",xs(v)).attr("x2",xs(v)).attr("y1",52).attr("y2",112).attr("stroke",P.good).attr("stroke-dasharray","4,3"); txt(g, xs(v), 126, lab, {anchor:"middle", size:9.5, fill:P.good}); } });
    txt(g, 30, 156, "weights", {size:10.5, fill:P.A}); txt(g, 100, 156, gb(wB), {size:10.5, mono:true});
    txt(g, 30, 176, "KV cache", {size:10.5, fill:P.B}); txt(g, 100, 176, gb(kB) + " (16-bit, one sequence)", {size:10.5, mono:true});
    txt(g, 30, 196, "total", {size:10.5, bold:true}); txt(g, 100, 196, gb(tot), {size:10.5, mono:true, bold:true});
    txt(g, 30, 226, "activations, framework overhead and any vision encoder are not included", {size:9.5, fill:P.muted});
    const fits = [24e9,48e9,80e9].filter(v=>tot<=v).map(v=>(v/1e9)+" GB");
    d3.select("#q-read").html(`${p.total.toLocaleString("en-US")} parameters × ${f.bpw.toFixed(3)} bits = <b>${gb(wB)}</b> of weights, plus <b>${gb(kB)}</b> of KV cache at ${n.toLocaleString("en-US")} tokens: <b>${gb(tot)}</b>. ${fits.length ? "Fits in a "+fits[0]+" device by this count." : "Needs more than one 80 GB device by this count."} Relative to bf16 the weights shrink ${(16/f.bpw).toFixed(2)}×.`);
  }
  d3.select("#q-model").on("change", draw); d3.select("#q-fmt").on("change", draw); d3.select("#q-n").on("input", draw);
  draw();
});
