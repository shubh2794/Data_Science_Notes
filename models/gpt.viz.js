/* gpt.viz.js — every interactive figure on models/gpt.html.
   Loaded after data.js / notes.js (palette C comes from notes.js).
   Each figure is its own IIFE wrapped in try/catch so one failure cannot blank the rest.
   House rule: every number a figure displays is computed here from a spec, never typed into a label. */

const GV = (function(){
  const P = Object.assign({}, (typeof C !== "undefined" ? C : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"}),
    { panel:"#15181f", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6" });

  /* exact GPT parameter count: L blocks of (12d² + 13d), tied embeddings, learned positions, optional final LN */
  function params(cfg){
    const {L,d,V,n} = cfg, fin = cfg.finalLN === false ? 0 : 2*d;
    const attn = L*(4*d*d + 4*d), ffn = L*(8*d*d + 5*d), ln = L*4*d + fin, emb = V*d, pos = n*d;
    return {attn, ffn, ln, emb, pos, total: attn+ffn+ln+emb+pos, nonEmb: attn+ffn+ln, approx: 12*L*d*d + V*d};
  }
  /* published configurations (spec) */
  const CFG = {
    gpt1:{name:"GPT-1",       L:12, d:768,   V:40478, n:512,  finalLN:false, reported:null,    reportedLabel:"≈117M (commonly cited)"},
    g2s: {name:"GPT-2 small", L:12, d:768,   V:50257, n:1024, reported:117e6,   reportedLabel:"117M"},
    g2m: {name:"GPT-2 medium",L:24, d:1024,  V:50257, n:1024, reported:345e6,   reportedLabel:"345M"},
    g2l: {name:"GPT-2 large", L:36, d:1280,  V:50257, n:1024, reported:762e6,   reportedLabel:"762M"},
    g2xl:{name:"GPT-2 XL",    L:48, d:1600,  V:50257, n:1024, reported:1542e6,  reportedLabel:"1542M"},
    g3s: {name:"GPT-3 Small", L:12, d:768,   V:50257, n:2048, reported:125e6,   reportedLabel:"125M"},
    g3m: {name:"GPT-3 Medium",L:24, d:1024,  V:50257, n:2048, reported:350e6,   reportedLabel:"350M"},
    g3lg:{name:"GPT-3 Large", L:24, d:1536,  V:50257, n:2048, reported:760e6,   reportedLabel:"760M"},
    g3xl:{name:"GPT-3 XL",    L:24, d:2048,  V:50257, n:2048, reported:1.3e9,   reportedLabel:"1.3B"},
    g327:{name:"GPT-3 2.7B",  L:32, d:2560,  V:50257, n:2048, reported:2.7e9,   reportedLabel:"2.7B"},
    g367:{name:"GPT-3 6.7B",  L:32, d:4096,  V:50257, n:2048, reported:6.7e9,   reportedLabel:"6.7B"},
    g313:{name:"GPT-3 13B",   L:40, d:5140,  V:50257, n:2048, reported:13.0e9,  reportedLabel:"13.0B"},
    g3:  {name:"GPT-3 175B",  L:96, d:12288, V:50257, n:2048, reported:175.0e9, reportedLabel:"175.0B"}
  };
  function fmtN(x){
    const a = Math.abs(x);
    if (a >= 1e12) return (x/1e12).toFixed(2)+"T";
    if (a >= 1e9)  return (x/1e9).toFixed(a>=1e11?1:2)+"B";
    if (a >= 1e6)  return (x/1e6).toFixed(a>=1e8?1:2)+"M";
    if (a >= 1e3)  return (x/1e3).toFixed(1)+"k";
    return String(Math.round(x));
  }
  function sci(x, p){ if (!isFinite(x) || x===0) return "0"; const e = Math.floor(Math.log10(Math.abs(x))); return (x/Math.pow(10,e)).toFixed(p==null?2:p)+" × 10"+sup(e); }
  function sup(e){ const m={"-":"⁻","0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"}; return String(e).split("").map(c=>m[c]).join(""); }
  function bytes(b){
    const u=["B","KiB","MiB","GiB","TiB"]; let i=0, v=b;
    while (v >= 1024 && i < u.length-1){ v/=1024; i++; }
    return v.toFixed(v>=100?0:(v>=10?1:2))+" "+u[i];
  }
  function softmax(z, T){
    const t = T || 1, m = Math.max(...z.map(v=>v/t));
    const e = z.map(v=>Math.exp(v/t - m)), s = e.reduce((a,b)=>a+b,0);
    return e.map(v=>v/s);
  }
  function entropyBits(p){ return -p.reduce((a,v)=> v>0 ? a + v*Math.log2(v) : a, 0); }
  function txt(g, x, y, s, o){
    o = o || {};
    return g.append("text").attr("x",x).attr("y",y).attr("font-size",o.size||11).attr("fill",o.fill||P.ink)
      .attr("text-anchor",o.anchor||"start").attr("font-family",o.mono?"SF Mono,Menlo,monospace":null)
      .attr("font-weight",o.bold?600:null).text(s);
  }
  function safe(name, fn){ try { fn(); } catch(e){ if (typeof console!=="undefined") console.error("[gpt.viz] "+name+" failed:", e); } }
  function lcg(seed){ let s = seed >>> 0; return () => { s = (1664525*s + 1013904223) >>> 0; return s/4294967296; }; }
  return {P, params, CFG, fmtN, sci, bytes, softmax, entropyBits, txt, safe, lcg};
})();

/* ───────────────────────── 03 · lineage timeline ───────────────────────── */
GV.safe("lineage", function(){
  const {P, params, CFG, fmtN, txt} = GV;
  const svg = d3.select("#lin-svg"), W = 640, H = 270, m = {l:64, r:24, t:22, b:40};
  const ms = [
    {name:"GPT-1", t:2018.45, n:params(CFG.gpt1).total, what:"generative pretraining on BooksCorpus + fine-tuning with input transformations; context 512"},
    {name:"GPT-2", t:2019.12, n:params(CFG.g2xl).total, lo:params(CFG.g2s).total, what:"WebText, byte-level BPE, pre-LN, zero-shot task transfer; four sizes; context 1024"},
    {name:"GPT-3", t:2020.41, n:params(CFG.g3).total, lo:params(CFG.g3s).total, what:"eight sizes on 300B tokens, few-shot in-context learning, alternating dense / banded sparse attention; context 2048"},
    {name:"InstructGPT", t:2022.17, n:params(CFG.g3).total, lo:params(CFG.g3xl).total, what:"SFT → reward model → PPO with KL penalty on GPT-3 architectures (1.3B, 6B, 175B)"},
    {name:"ChatGPT", t:2022.91, n:null, what:"sibling of InstructGPT fine-tuned from a GPT-3.5-series model on dialogue data; size not published"},
    {name:"GPT-4", t:2023.2, n:null, what:"image + text input, loss predicted from runs with up to 10,000× less compute; architecture and size withheld"}
  ];
  const x = d3.scaleLinear().domain([2018, 2023.6]).range([m.l, W-m.r]);
  const y = d3.scaleLog().domain([5e7, 1e12]).range([H-m.b, m.t]);
  let sel = 2;
  function draw(){
    svg.selectAll("*").remove();
    const g = svg.append("g");
    [1e8,1e9,1e10,1e11,1e12].forEach(v=>{
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3");
      txt(g, m.l-8, y(v)+4, fmtN(v), {anchor:"end", size:10, fill:P.muted});
    });
    [2018,2019,2020,2021,2022,2023].forEach(v=>{
      txt(g, x(v), H-m.b+16, String(v), {anchor:"middle", size:10, fill:P.muted});
      g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",H-m.b).attr("y2",H-m.b+4).attr("stroke",P.muted);
    });
    txt(g, 14, m.t+4, "params", {size:10, fill:P.muted});
    const known = ms.filter(d=>d.n);
    g.append("path").datum(known.slice(0,3)).attr("fill","none").attr("stroke",P.A).attr("stroke-opacity",.5)
      .attr("d", d3.line().x(d=>x(d.t)).y(d=>y(d.n)));
    ms.forEach((d,i)=>{
      const cx = x(d.t), cy = d.n ? y(d.n) : m.t+8, on = i===sel;
      if (d.lo) g.append("line").attr("x1",cx).attr("x2",cx).attr("y1",y(d.lo)).attr("y2",y(d.n)).attr("stroke",on?P.B:P.muted).attr("stroke-width",3).attr("stroke-opacity",.45);
      const node = g.append("g").style("cursor","pointer").on("click",()=>{ sel=i; draw(); });
      if (d.n) node.append("circle").attr("cx",cx).attr("cy",cy).attr("r",on?8:6).attr("fill",on?P.B:P.A);
      else node.append("rect").attr("x",cx-7).attr("y",cy-7).attr("width",14).attr("height",14).attr("fill","none").attr("stroke",on?P.B:P.muted).attr("stroke-dasharray","3,2");
      txt(node, cx, cy + (d.n ? -12 : 22), d.name, {anchor:"middle", size:11, fill:on?P.B:P.ink, bold:on});
    });
    const d = ms[sel];
    let extra;
    if (d.n){
      const prev = ms.slice(0,sel).filter(p=>p.n && p.name!=="InstructGPT").pop();
      extra = `largest size <b>${fmtN(d.n)}</b> (exact count from its configuration)` + (d.lo?`, smallest ${fmtN(d.lo)}`:"") +
        (prev && d.name!=="InstructGPT" ? ` · ${(d.n/prev.n).toFixed(0)}× the previous generation's largest` : "");
    } else extra = "parameter count <b>not published</b> — drawn as a dashed box, not a point";
    d3.select("#lin-read").html(`<b>${d.name}</b> (${Math.floor(d.t)}): ${d.what}. ${extra}.`);
  }
  draw();
});

/* ───────────────────────── 06 · causal mask ───────────────────────── */
GV.safe("mask", function(){
  const {P, txt} = GV;
  const svg = d3.select("#mask-svg"), toks = ["<s>","the","cat","sat","on","the","mat",".","it","slept"];
  let n = 5, row = 4, off = false;
  function draw(){
    svg.selectAll("*").remove();
    if (row > n-1) row = n-1;
    const cell = Math.min(40, Math.floor(250/n)), x0 = 110, y0 = 40, g = svg.append("g");
    txt(g, x0 + n*cell/2, 18, "keys kⱼ (attended to)", {anchor:"middle", size:10.5, fill:P.muted});
    txt(g, 18, y0 + n*cell/2, "queries qᵢ", {size:10.5, fill:P.muted});
    for (let j=0;j<n;j++) txt(g, x0 + j*cell + cell/2, y0-6, toks[j], {anchor:"middle", size:Math.min(11, cell*0.38), mono:true, fill:P.muted});
    let allowed = 0;
    for (let i=0;i<n;i++){
      const rg = g.append("g").style("cursor","pointer").on("click",()=>{ row=i; draw(); });
      txt(rg, x0-8, y0 + i*cell + cell/2 + 4, toks[i], {anchor:"end", size:Math.min(11, cell*0.38), mono:true, fill:i===row?P.B:P.ink});
      for (let j=0;j<n;j++){
        const ok = off || j <= i; if (ok) allowed++;
        const leak = off && j === i+1;
        rg.append("rect").attr("x",x0+j*cell+1).attr("y",y0+i*cell+1).attr("width",cell-2).attr("height",cell-2).attr("rx",3)
          .attr("fill", ok ? (leak ? P.bad : (i===row ? P.B : P.A)) : P.panel)
          .attr("fill-opacity", ok ? (i===row ? 0.85 : 0.45) : 1)
          .attr("stroke", i===row ? P.B : P.line);
        if (cell >= 26) txt(rg, x0+j*cell+cell/2, y0+i*cell+cell/2+4, ok ? "0" : "−∞", {anchor:"middle", size:10, mono:true, fill: ok ? P.ink : P.muted});
      }
    }
    /* right panel: softmax weights for the selected row with uniform scores (computed) */
    const px = x0 + n*cell + 40, keys = [];
    for (let j=0;j<n;j++) if (off || j<=row) keys.push(j);
    const w = 1/keys.length;
    txt(g, px, y0-6, `row q${row+1}: weights if scores were equal`, {size:10.5, fill:P.muted});
    for (let j=0;j<n;j++){
      const yy = y0 + j*22, ok = keys.includes(j), bw = 160*(ok?w:0);
      txt(g, px+40, yy+13, toks[j], {anchor:"end", size:10.5, mono:true, fill: ok?P.ink:P.muted});
      g.append("rect").attr("x",px+46).attr("y",yy+2).attr("width",160).attr("height",14).attr("rx",3).attr("fill",P.panel);
      g.append("rect").attr("x",px+46).attr("y",yy+2).attr("width",bw).attr("height",14).attr("rx",3).attr("fill", (off && j===row+1) ? P.bad : P.B).attr("fill-opacity",.8);
      txt(g, px+212, yy+13, ok ? w.toFixed(3) : "0", {size:10, mono:true, fill: ok?P.ink:P.muted});
    }
    const tri = n*(n+1)/2;
    d3.select("#mask-read").html(off
      ? `mask removed: all <b>${allowed}</b> of ${n*n} pairs allowed. Query ${row+1} can now read <b style="color:${P.bad}">${row+1<n ? toks[row+1] : "(no later token)"}</b> — the very token it is trained to predict. That is the leak the mask exists to prevent.`
      : `causal mask: <b>${allowed}</b> of ${n*n} pairs allowed (n(n+1)/2 = ${tri}, ${(100*allowed/(n*n)).toFixed(1)}%). Query ${row+1} ("${toks[row]}") attends to <b>${keys.length}</b> key${keys.length>1?"s":""}; its target is token ${row+2 <= toks.length ? '"'+toks[row+1]+'"' : "beyond the shown sequence"}, which it cannot see.`);
  }
  d3.select("#mask-n").on("input", function(){ n = +this.value; d3.select("#mask-nv").text(n); draw(); });
  d3.select("#mask-off").on("change", function(){ off = this.checked; draw(); });
  draw();
});

/* ───────────────────────── 07 · step-through generation ───────────────────────── */
GV.safe("generation", function(){
  const {P, softmax, txt} = GV;
  const svg = d3.select("#gen-svg");
  /* toy model: logits over a small candidate set, keyed by the full prefix */
  const M = {
    "The cat":                 [["sat",2.2],["is",1.6],["ran",1.1],["was",0.9],["jumped",0.2]],
    "The cat sat":             [["on",2.6],["down",1.4],["quietly",0.6],["by",0.4]],
    "The cat sat on":          [["the",3.0],["a",1.8],["my",0.9],["top",-0.2]],
    "The cat sat on the":      [["mat",2.4],["sofa",1.9],["floor",1.2],["roof",0.3]],
    "The cat sat on the mat":  [[".",2.8],["and",1.2],[",",1.0],["today",-0.5]],
    "The cat is":              [["asleep",2.0],["hungry",1.7],["black",1.2],["here",0.6]],
    "The cat is asleep":       [[".",3.0],["on",1.5],["again",0.8],["now",0.4]]
  };
  const prompt = ["The","cat"];
  let gen = [], logp = [];
  function dist(seq){ const c = M[seq.join(" ")]; if (!c) return null; const p = softmax(c.map(d=>d[1])); return c.map((d,i)=>({tok:d[0], z:d[1], p:p[i]})); }
  function choose(ds, step){ const mode = d3.select("#gen-mode").property("value"); return (mode==="second" && step===0) ? 1 : 0; }
  function draw(){
    svg.selectAll("*").remove();
    const g = svg.append("g"), seq = prompt.concat(gen);
    let x = 16; const y = 30;
    txt(g, 16, 16, "sequence so far (prompt in grey, generated in orange)", {size:10, fill:P.muted});
    seq.forEach((t,i)=>{
      const isGen = i >= prompt.length, w = Math.max(34, t.length*8.5+16);
      g.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",26).attr("rx",5)
        .attr("fill", isGen ? "rgba(255,180,84,.16)" : P.panel).attr("stroke", isGen ? P.B : P.line);
      txt(g, x+w/2, y+17, t, {anchor:"middle", mono:true, size:11.5, fill:isGen?P.B:P.ink});
      x += w+5;
    });
    const ds = dist(seq), top = 84, x0 = 150, bw = 280;
    if (ds){
      const pick = choose(ds, gen.length);
      txt(g, 16, top-8, `step ${gen.length+1}: P(next | "${seq.join(" ")}")`, {size:10.5, fill:P.muted});
      txt(g, x0+bw+70, top-8, "logit", {size:10, fill:P.muted, anchor:"end"});
      ds.forEach((d,i)=>{
        const yy = top + i*30, on = i===pick;
        txt(g, x0-10, yy+14, d.tok, {anchor:"end", mono:true, size:11.5, fill:on?P.good:P.ink});
        g.append("rect").attr("x",x0).attr("y",yy+2).attr("width",bw).attr("height",16).attr("rx",4).attr("fill",P.panel);
        g.append("rect").attr("x",x0).attr("y",yy+2).attr("width",d.p*bw).attr("height",16).attr("rx",4).attr("fill",on?P.good:P.A).attr("fill-opacity",on?.9:.55);
        txt(g, x0+d.p*bw+6, yy+14, (100*d.p).toFixed(1)+"%", {size:10, fill:on?P.good:P.muted});
        txt(g, x0+bw+70, yy+14, d.z.toFixed(1), {size:10, mono:true, fill:P.muted, anchor:"end"});
      });
    } else {
      txt(g, 16, top+10, "generation stopped: the toy model emitted \".\" (its stand-in for an end token)", {size:11, fill:P.muted});
    }
    const sum = logp.reduce((a,b)=>a+b,0), mean = logp.length ? -sum/logp.length : 0;
    const read = ds
      ? `next choice: <b style="color:${P.good}">${ds[choose(ds, gen.length)].tok}</b> with p = ${ds[choose(ds, gen.length)].p.toFixed(3)} (softmax over ${ds.length} candidate logits). `
      : "done. ";
    d3.select("#gen-read").html(read + (logp.length
      ? `generated ${logp.length} token${logp.length>1?"s":""}: Σ ln p = <b>${sum.toFixed(3)}</b>, mean NLL ${mean.toFixed(3)} nats, perplexity of the continuation <b>${Math.exp(mean).toFixed(2)}</b>.`
      : "nothing generated yet — press the button."));
  }
  d3.select("#gen-next").on("click", ()=>{
    const seq = prompt.concat(gen), ds = dist(seq); if (!ds) return;
    const k = choose(ds, gen.length); gen.push(ds[k].tok); logp.push(Math.log(ds[k].p)); draw();
  });
  d3.select("#gen-reset").on("click", ()=>{ gen=[]; logp=[]; draw(); });
  d3.select("#gen-mode").on("change", ()=>{ gen=[]; logp=[]; draw(); });
  draw();
});

/* ───────────────────────── 09 · pre-LN vs post-LN ───────────────────────── */
GV.safe("prepost", function(){
  const {P, txt} = GV;
  const svg = d3.select("#ln-svg");
  function box(g, x, y, w, label, col){
    g.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",24).attr("rx",5).attr("fill",P.panel).attr("stroke",col);
    txt(g, x+w/2, y+16, label, {anchor:"middle", size:10.5, fill:col});
  }
  function draw(){
    svg.selectAll("*").remove();
    const mode = d3.select("#ln-mode").property("value"), L = +d3.select("#ln-L").property("value"), scaled = d3.select("#ln-scale").property("checked");
    d3.select("#ln-Lv").text(L);
    const g = svg.append("g"), N = 2*L;
    /* left: block diagram, bottom to top */
    const cx = 40, seq = mode==="pre"
      ? [["x (residual in)",P.muted],["LN",P.B],["masked attn",P.A],["⊕ add",P.good],["LN",P.B],["FFN (GELU)",P.A],["⊕ add",P.good],["… × L, then final LN",P.B]]
      : [["x (residual in)",P.muted],["masked attn",P.A],["⊕ add",P.good],["LN",P.B],["FFN (GELU)",P.A],["⊕ add",P.good],["LN",P.B],["… × L (no final LN)",P.muted]];
    txt(g, cx, 16, mode==="pre" ? "pre-LN block (GPT-2, GPT-3)" : "post-LN block (GPT-1)", {size:11, bold:true});
    seq.forEach((s,i)=>{ const yy = 262 - i*31; box(g, cx, yy, 150, s[0], s[1]);
      if (i) g.append("line").attr("x1",cx+75).attr("x2",cx+75).attr("y1",yy+24).attr("y2",yy+31).attr("stroke",P.muted); });
    /* residual bypass line */
    const bx = cx+170;
    g.append("path").attr("d", `M${cx+150},${274} L${bx},${274} L${bx},${60} L${cx+150},${60}`).attr("fill","none")
      .attr("stroke", mode==="pre" ? P.good : P.muted).attr("stroke-dasharray", mode==="pre" ? null : "4,3");
    txt(g, bx+4, 170, mode==="pre" ? "clean identity path" : "path passes LNs", {size:9.5, fill: mode==="pre"?P.good:P.muted});
    /* right: residual stream std vs depth under a random-sum model */
    const x0 = 330, x1 = 620, y0 = 250, y1 = 40;
    const s2 = 1, add = scaled ? s2/N : s2;           /* per-sublayer added variance */
    const pts = d3.range(0, L+1).map(l => ({l, sd: mode==="pre" ? Math.sqrt(1 + 2*l*add) : 1}));
    const ymax = Math.max(2, d3.max(pts, d=>d.sd)*1.1);
    const xs = d3.scaleLinear().domain([0,L]).range([x0,x1]), ys = d3.scaleLinear().domain([0,ymax]).range([y0,y1]);
    g.append("line").attr("x1",x0).attr("x2",x1).attr("y1",y0).attr("y2",y0).attr("stroke",P.muted);
    g.append("line").attr("x1",x0).attr("x2",x0).attr("y1",y0).attr("y2",y1).attr("stroke",P.muted);
    ys.ticks(4).forEach(t=>{ txt(g, x0-6, ys(t)+4, String(t), {anchor:"end", size:9.5, fill:P.muted}); });
    txt(g, (x0+x1)/2, y0+26, "block index ℓ", {anchor:"middle", size:10, fill:P.muted});
    txt(g, x0, y1-12, "std of the residual stream (model: unit input, unit-variance sub-layer outputs)", {size:9.5, fill:P.muted});
    [0, Math.round(L/2), L].forEach(t=>txt(g, xs(t), y0+14, String(t), {anchor:"middle", size:9.5, fill:P.muted}));
    g.append("path").datum(pts).attr("fill","none").attr("stroke",P.B).attr("stroke-width",2).attr("d", d3.line().x(d=>xs(d.l)).y(d=>ys(d.sd)));
    const last = pts[pts.length-1];
    g.append("circle").attr("cx",xs(L)).attr("cy",ys(last.sd)).attr("r",4).attr("fill",P.B);
    const initStd = 0.02/Math.sqrt(N);
    d3.select("#ln-read").html(mode==="pre"
      ? `pre-LN, L = ${L}: N = 2L = <b>${N}</b> residual sub-layers. ${scaled ? `Scaled init: output-projection std 0.02/√${N} = <b>${initStd.toFixed(5)}</b>; stream std after the last block ≈ √(1 + 2L·1/N) = <b>${last.sd.toFixed(2)}</b>.` : `Unscaled: stream std after the last block ≈ √(1 + 2L) = <b>${last.sd.toFixed(2)}</b> — grows with depth, which the final LN must absorb.`}`
      : `post-LN, L = ${L}: the stream is renormalised after every sub-layer, so its std stays <b>1.00</b>, but the gradient to block 1 passes through <b>${N}</b> LayerNorms on the residual path — the reason deep post-LN stacks need warm-up.${scaled?" (Residual scaling has little role here.)":""}`);
  }
  d3.select("#ln-mode").on("change", draw);
  d3.select("#ln-L").on("input", draw);
  d3.select("#ln-scale").on("change", draw);
  draw();
});

/* ───────────────────────── 10 · BPE merges ───────────────────────── */
GV.safe("bpe", function(){
  const {P, txt} = GV;
  const svg = d3.select("#bpe-svg");
  const CORPUS = [["low",5],["lower",2],["newest",6],["widest",3]], EOW = "·";
  let words, merges;
  function reset(){ words = CORPUS.map(([w,c])=>({sym: w.split("").concat([EOW]), c})); merges = []; }
  function counts(){
    const m = new Map();                          /* insertion order = first occurrence */
    words.forEach(w=>{ for (let i=0;i<w.sym.length-1;i++){ const k = w.sym[i]+"\u0001"+w.sym[i+1]; m.set(k, (m.get(k)||0) + w.c); } });
    return [...m.entries()].map(([k,v])=>({a:k.split("\u0001")[0], b:k.split("\u0001")[1], n:v}));
  }
  function best(cs){ let b = null; cs.forEach(c=>{ if (!b || c.n > b.n) b = c; }); return b; }
  function step(){
    const b = best(counts()); if (!b || b.n < 2) return false;
    words.forEach(w=>{ const out=[]; for (let i=0;i<w.sym.length;i++){ if (i<w.sym.length-1 && w.sym[i]===b.a && w.sym[i+1]===b.b){ out.push(b.a+b.b); i++; } else out.push(w.sym[i]); } w.sym = out; });
    merges.push(b); return true;
  }
  const base = new Set(); CORPUS.forEach(([w])=>w.split("").forEach(ch=>base.add(ch))); base.add(EOW);
  function draw(msg){
    svg.selectAll("*").remove();
    const g = svg.append("g");
    txt(g, 16, 16, "corpus, current segmentation (count ×)", {size:10, fill:P.muted});
    words.forEach((w,i)=>{
      let x = 60; const y = 28 + i*34;
      txt(g, 50, y+17, w.c+"×", {anchor:"end", size:11, fill:P.muted});
      w.sym.forEach(s=>{
        const bw = Math.max(20, s.length*9+10), isNew = merges.length && s === merges[merges.length-1].a + merges[merges.length-1].b;
        g.append("rect").attr("x",x).attr("y",y).attr("width",bw).attr("height",24).attr("rx",4).attr("fill",isNew?"rgba(255,180,84,.2)":P.panel).attr("stroke",isNew?P.B:P.line);
        txt(g, x+bw/2, y+16, s, {anchor:"middle", mono:true, size:11.5, fill:isNew?P.B:P.ink}); x += bw+3;
      });
    });
    /* pair counts */
    const cs = counts().slice().sort((a,b)=>b.n-a.n).slice(0,6), bx = 380, top = 30;
    txt(g, bx, 16, "most frequent adjacent pairs now", {size:10, fill:P.muted});
    const mx = cs.length ? cs[0].n : 1;
    cs.forEach((c,i)=>{
      const yy = top + i*24;
      txt(g, bx+62, yy+13, c.a+" "+c.b, {anchor:"end", mono:true, size:10.5});
      g.append("rect").attr("x",bx+68).attr("y",yy+2).attr("width",150*c.n/mx).attr("height",14).attr("rx",3).attr("fill",i===0?P.good:P.A).attr("fill-opacity",.7);
      txt(g, bx+72+150*c.n/mx, yy+13, String(c.n), {size:10, fill:P.muted});
    });
    txt(g, 16, 172, "merges learned, in order", {size:10, fill:P.muted});
    merges.forEach((mm,i)=>{ const col = i%4, rw = Math.floor(i/4); txt(g, 16 + col*150, 192 + rw*18, `${i+1}. ${mm.a} + ${mm.b} → ${mm.a+mm.b}  (${mm.n})`, {mono:true, size:10.5, fill:i===merges.length-1?P.B:P.ink}); });
    const tokens = words.reduce((a,w)=>a + w.sym.length*w.c, 0);
    d3.select("#bpe-read").html((msg ? msg+" " : "") + `after <b>${merges.length}</b> merge${merges.length===1?"":"s"}: vocabulary = ${base.size} base symbols + ${merges.length} merges = <b>${base.size+merges.length}</b>; the corpus is <b>${tokens}</b> tokens long (it started at ${CORPUS.reduce((a,[w,c])=>a+(w.length+1)*c,0)}).` +
      (cs.length ? ` Next merge would be <b>${best(counts()).a} + ${best(counts()).b}</b> (count ${best(counts()).n}; ties go to the pair seen first).` : ""));
  }
  d3.select("#bpe-step").on("click", ()=>{ const ok = step(); draw(ok ? "" : "no pair occurs twice any more — stopping."); });
  d3.select("#bpe-reset").on("click", ()=>{ reset(); draw(); });
  reset(); draw();
});

/* ───────────────────────── 12 · parameter calculator ───────────────────────── */
GV.safe("params", function(){
  const {P, params, CFG, fmtN, txt} = GV;
  const svg = d3.select("#pc-svg");
  let st = Object.assign({}, CFG.g2s), presetKey = "g2s";
  function syncSliders(){
    [["L",st.L],["d",st.d],["V",st.V],["n",st.n]].forEach(([k,v])=>{ d3.select("#pc-"+k).property("value", v); d3.select("#pc-"+k+"v").text(v); });
  }
  function draw(){
    svg.selectAll("*").remove();
    const r = params(st), g = svg.append("g");
    const parts = [["attention",r.attn,P.A],["FFN",r.ffn,P.teal],["LayerNorms",r.ln,P.purple],["token emb (tied)",r.emb,P.B],["positions",r.pos,P.pink]];
    const x0 = 20, W = 600, y = 40;
    txt(g, x0, 22, `${presetKey ? CFG[presetKey].name : "custom"} — L ${st.L}, d ${st.d}, V ${st.V}, n_ctx ${st.n}${st.finalLN===false?", no final LN":""}`, {size:11, bold:true});
    let x = x0;
    parts.forEach(([lab,v,col])=>{
      const w = W*v/r.total;
      g.append("rect").attr("x",x).attr("y",y).attr("width",Math.max(0,w)).attr("height",34).attr("fill",col).attr("fill-opacity",.75);
      x += w;
    });
    parts.forEach(([lab,v,col],i)=>{
      const yy = 96 + i*24;
      g.append("rect").attr("x",x0).attr("y",yy).attr("width",12).attr("height",12).attr("fill",col).attr("fill-opacity",.75);
      txt(g, x0+20, yy+11, lab, {size:11});
      txt(g, x0+190, yy+11, fmtN(v), {size:11, mono:true, anchor:"end"});
      txt(g, x0+250, yy+11, (100*v/r.total).toFixed(1)+"%", {size:11, mono:true, anchor:"end", fill:P.muted});
    });
    const approxErr = 100*(r.approx - r.total)/r.total;
    txt(g, 330, 108, "exact total", {size:10.5, fill:P.muted}); txt(g, 600, 108, r.total.toLocaleString("en-US"), {size:12, mono:true, anchor:"end", bold:true});
    txt(g, 330, 132, "12·L·d² + V·d", {size:10.5, fill:P.muted}); txt(g, 600, 132, fmtN(r.approx)+` (${approxErr>=0?"+":""}${approxErr.toFixed(2)}%)`, {size:11, mono:true, anchor:"end"});
    txt(g, 330, 156, "non-embedding", {size:10.5, fill:P.muted}); txt(g, 600, 156, fmtN(r.nonEmb), {size:11, mono:true, anchor:"end"});
    const cfg = presetKey ? CFG[presetKey] : null;
    if (cfg){
      txt(g, 330, 180, "reported in the paper", {size:10.5, fill:P.muted}); txt(g, 600, 180, cfg.reportedLabel, {size:11, mono:true, anchor:"end", fill:P.B});
    }
    let rep = "";
    if (cfg && cfg.reported){ const diff = r.total - cfg.reported; rep = ` Against the reported ${cfg.reportedLabel}: formula is ${diff>=0?"+":""}${fmtN(diff)} (${(100*diff/cfg.reported).toFixed(1)}%).`; }
    d3.select("#pc-read").html(`exact count <b>${fmtN(r.total)}</b>; embeddings are <b>${(100*(r.emb+r.pos)/r.total).toFixed(1)}%</b> of it; FFN : attention = ${(r.ffn/r.attn).toFixed(2)} : 1.${rep}`);
  }
  d3.select("#pc-preset").on("change", function(){ presetKey = this.value; st = Object.assign({}, CFG[presetKey]); syncSliders(); draw(); });
  ["L","d","V","n"].forEach(k=>{
    d3.select("#pc-"+k).on("input", function(){ st[k] = +this.value; d3.select("#pc-"+k+"v").text(this.value);
      const c = CFG[presetKey]; if (c && (c.L!==st.L || c.d!==st.d || c.V!==st.V || c.n!==st.n)) presetKey = null; draw(); });
  });
  syncSliders(); draw();
});

/* ───────────────────────── 13 · training compute 6ND ───────────────────────── */
GV.safe("compute", function(){
  const {P, params, CFG, fmtN, sci, txt} = GV;
  const svg = d3.select("#cx-svg"), m = {l:58, r:20, t:16, b:36}, W = 640, H = 280;
  const PRE = {
    g3s: {N:params(CFG.g3s).total, D:300e9, label:"GPT-3 Small"},
    g3xl:{N:params(CFG.g3xl).total, D:300e9, label:"GPT-3 XL"},
    g313:{N:params(CFG.g313).total, D:300e9, label:"GPT-3 13B"},
    g3:  {N:params(CFG.g3).total, D:300e9, label:"GPT-3 175B"},
    chin:{N:70e9, D:1.4e12, label:"70B / 1.4T"}
  };
  const g3pts = ["g3s","g3m","g3lg","g3xl","g327","g367","g313","g3"].map(k=>({k, N:params(CFG[k]).total, D:300e9}));
  let N = PRE.g3.N, D = PRE.g3.D, label = PRE.g3.label;
  const x = d3.scaleLinear().domain([7,12]).range([m.l, W-m.r]), y = d3.scaleLinear().domain([9,13]).range([H-m.b, m.t]);
  function sync(){ d3.select("#cx-N").property("value", Math.log10(N)); d3.select("#cx-D").property("value", Math.log10(D)); }
  function draw(){
    svg.selectAll("*").remove();
    const g = svg.append("g"), T = +d3.select("#cx-T").property("value");
    d3.select("#cx-Nv").text(fmtN(N)); d3.select("#cx-Dv").text(fmtN(D)); d3.select("#cx-Tv").text(T);
    [7,8,9,10,11,12].forEach(e=>{ g.append("line").attr("x1",x(e)).attr("x2",x(e)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.line).attr("stroke-dasharray","2,3"); });
    ["10⁷","10⁸","10⁹","10¹⁰","10¹¹","10¹²"].forEach((s,i)=>txt(g, x(7+i), H-m.b+14, s, {anchor:"middle", size:9.5, fill:P.muted}));
    ["10⁹","10¹⁰","10¹¹","10¹²","10¹³"].forEach((s,i)=>{ txt(g, m.l-6, y(9+i)+4, s, {anchor:"end", size:9.5, fill:P.muted});
      g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(9+i)).attr("y2",y(9+i)).attr("stroke",P.line).attr("stroke-dasharray","2,3"); });
    txt(g, (m.l+W-m.r)/2, H-4, "parameters N", {anchor:"middle", size:10, fill:P.muted});
    txt(g, 12, m.t+4, "tokens D", {size:10, fill:P.muted});
    /* iso-compute lines: log D = log(C/6) − log N */
    [18,20,22,24,26].forEach(ce=>{
      const pts = [7,12].map(ln=>({ln, ld: Math.log10(Math.pow(10,ce)/6) - ln}));
      g.append("line").attr("x1",x(pts[0].ln)).attr("y1",y(pts[0].ld)).attr("x2",x(pts[1].ln)).attr("y2",y(pts[1].ld)).attr("stroke",P.purple).attr("stroke-opacity",.4);
      const lx = Math.log10(Math.pow(10,ce)/6) - 12.6; /* label where line exits top (D=10^12.6) */
      if (lx > 7.1 && lx < 11.9) txt(g, x(lx), y(12.75), "C = " + GV.sci(Math.pow(10,ce),0).replace("1 × ",""), {anchor:"middle", size:9, fill:P.purple});
    });
    /* 20 tokens per parameter */
    g.append("line").attr("x1",x(7)).attr("y1",y(7+Math.log10(20))).attr("x2",x(12)).attr("y2",y(Math.min(13,12+Math.log10(20)))).attr("stroke",P.good).attr("stroke-dasharray","5,3");
    txt(g, x(8.3), y(8.3+Math.log10(20))-6, "D = 20·N", {size:9.5, fill:P.good});
    g3pts.forEach(p=>g.append("circle").attr("cx",x(Math.log10(p.N))).attr("cy",y(Math.log10(p.D))).attr("r",3.5).attr("fill",P.A));
    txt(g, x(Math.log10(g3pts[0].N)), y(11.477)+16, "GPT-3 sweep (300B tokens)", {size:9.5, fill:P.A});
    g.append("circle").attr("cx",x(Math.log10(N))).attr("cy",y(Math.log10(D))).attr("r",7).attr("fill","none").attr("stroke",P.B).attr("stroke-width",2);
    const C = 6*N*D, pfd = C/(1e15*86400), days = C/(T*1e15)/86400, tpp = D/N;
    d3.select("#cx-read").html(`${label ? "<b>"+label+"</b>: " : ""}C = 6 · ${fmtN(N)} · ${fmtN(D)} = <b>${sci(C)}</b> FLOPs = <b>${pfd.toLocaleString("en-US",{maximumFractionDigits:pfd<10?2:0})}</b> petaflop/s-days. At a sustained ${T} PFLOP/s that is <b>${days < 1 ? (days*24).toFixed(1)+" hours" : days.toFixed(days<10?1:0)+" days"}</b>. Tokens per parameter: <b>${tpp.toFixed(tpp<10?2:1)}</b> (the compute-optimal rule of thumb is ≈20; a compute-optimal model for this C would have N ≈ ${fmtN(Math.sqrt(C/120))}, D ≈ ${fmtN(20*Math.sqrt(C/120))}).`);
  }
  d3.select("#cx-preset").on("change", function(){ const p = PRE[this.value]; N = p.N; D = p.D; label = p.label; sync(); draw(); });
  d3.select("#cx-N").on("input", function(){ N = Math.pow(10, +this.value); label = null; draw(); });
  d3.select("#cx-D").on("input", function(){ D = Math.pow(10, +this.value); label = null; draw(); });
  d3.select("#cx-T").on("input", draw);
  sync(); draw();
});

/* ───────────────────────── 14 · GPT-3 data mixture ───────────────────────── */
GV.safe("mixture", function(){
  const {P, txt} = GV;
  const svg = d3.select("#mix-svg");
  const ROWS = [ /* name, tokens (B), weight, reported epochs at 300B — Table 2.2 of arXiv 2005.14165 */
    ["Common Crawl", 410, 0.60, 0.44], ["WebText2", 19, 0.22, 2.9], ["Books1", 12, 0.08, 1.9], ["Books2", 55, 0.08, 0.43], ["Wikipedia", 3, 0.03, 3.4]
  ];
  function draw(){
    svg.selectAll("*").remove();
    const Dt = +d3.select("#mix-D").property("value"); d3.select("#mix-Dv").text(Dt);
    const g = svg.append("g"), tot = ROWS.reduce((a,r)=>a+r[1],0), wsum = ROWS.reduce((a,r)=>a+r[2],0);
    const x0 = 120, bw = 220, top = 40;
    txt(g, x0, 18, "share of available tokens", {size:10, fill:P.muted}); txt(g, x0, 30, "vs weight in training mix", {size:10, fill:P.B});
    txt(g, 420, 18, `implied epochs at ${Dt}B`, {size:10, fill:P.muted}); txt(g, 560, 18, "reported @300B", {size:10, fill:P.muted});
    const ex = d3.scaleLinear().domain([0, 4]).range([420, 540]);
    let worst = null;
    ROWS.forEach((r,i)=>{
      const yy = top + i*40, share = r[1]/tot, ep = r[2]*Dt/r[1];
      txt(g, x0-8, yy+16, r[0], {anchor:"end", size:11});
      g.append("rect").attr("x",x0).attr("y",yy).attr("width",bw*share).attr("height",12).attr("fill",P.A).attr("fill-opacity",.7);
      txt(g, x0+bw*share+4, yy+10, (100*share).toFixed(1)+"%", {size:9.5, fill:P.muted});
      g.append("rect").attr("x",x0).attr("y",yy+14).attr("width",bw*r[2]).attr("height",12).attr("fill",P.B).attr("fill-opacity",.8);
      txt(g, x0+bw*r[2]+4, yy+24, (100*r[2]).toFixed(0)+"%", {size:9.5, fill:P.B});
      g.append("line").attr("x1",ex(0)).attr("x2",ex(4)).attr("y1",yy+13).attr("y2",yy+13).attr("stroke",P.line);
      g.append("line").attr("x1",ex(1)).attr("x2",ex(1)).attr("y1",yy+5).attr("y2",yy+21).attr("stroke",P.muted).attr("stroke-dasharray","2,2");
      g.append("circle").attr("cx",ex(Math.min(4,ep))).attr("cy",yy+13).attr("r",5).attr("fill",ep>1?P.good:P.purple);
      txt(g, ex(Math.min(4,ep))+(ep>3.3?-8:8), yy+9, ep.toFixed(2)+(ep>4?"+":""), {size:9.5, anchor:ep>3.3?"end":"start"});
      txt(g, 600, yy+17, r[3].toFixed(2), {size:10.5, mono:true, anchor:"end", fill:P.muted});
      const dev = Math.abs(r[2]*300/r[1] - r[3]);
      if (!worst || dev > worst.dev) worst = {name:r[0], dev, calc:r[2]*300/r[1], rep:r[3]};
    });
    txt(g, ex(1), top+5*40+6, "1 epoch", {anchor:"middle", size:9, fill:P.muted});
    const cc = ROWS[0], cEp = cc[2]*Dt/cc[1];
    d3.select("#mix-read").html(`weights sum to <b>${(100*wsum).toFixed(0)}%</b> as printed. At ${Dt}B tokens, Common Crawl is seen ${cEp.toFixed(2)} times and Wikipedia ${(ROWS[4][2]*Dt/ROWS[4][1]).toFixed(2)} times — a <b>${((ROWS[4][2]/ROWS[4][1])/(cc[2]/cc[1])).toFixed(1)}×</b> oversampling of Wikipedia relative to Common Crawl. Largest gap between recomputed and reported epochs at 300B: <b>${worst.name}</b> (${worst.calc.toFixed(2)} vs ${worst.rep}).`);
  }
  d3.select("#mix-D").on("input", draw);
  draw();
});

/* ───────────────────────── 15 · GPT-1 input transformations ───────────────────────── */
GV.safe("finetune", function(){
  const {P, params, CFG, fmtN, txt} = GV;
  const svg = d3.select("#ft-svg"), d = CFG.gpt1.d, base = params(CFG.gpt1).total;
  const S = "⟨s⟩", E = "⟨e⟩", D = "$";
  const TASKS = {
    cls: {rows:[[S,"text",E]], combine:"linear → softmax over classes", out:2, note:"sentiment / acceptability: one sequence"},
    ent: {rows:[[S,"premise",D,"hypothesis",E]], combine:"linear → {entail, neutral, contradict}", out:3, note:"premise and hypothesis joined by a delimiter"},
    sim: {rows:[[S,"text 1",D,"text 2",E],[S,"text 2",D,"text 1",E]], combine:"add the two final states element-wise → linear", out:2, note:"no natural order, so both orders are processed"},
    mc:  {rows:[[S,"context + question",D,"answer 1",E],[S,"context + question",D,"answer 2",E],[S,"context + question",D,"answer 3",E]], combine:"linear score per row → softmax across rows", out:1, note:"one sequence per candidate answer"}
  };
  function draw(){
    svg.selectAll("*").remove();
    const key = d3.select("#ft-task").property("value"), t = TASKS[key], g = svg.append("g");
    const top = 30, rh = Math.min(64, 210/t.rows.length);
    txt(g, 16, 16, t.note, {size:10.5, fill:P.muted});
    t.rows.forEach((row,i)=>{
      let x = 16; const y = top + i*rh + 6;
      row.forEach(s=>{
        const special = s===S||s===E||s===D, w = special ? 30 : Math.max(58, s.length*7.2+14);
        g.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",26).attr("rx",5).attr("fill",special?"rgba(255,180,84,.18)":P.panel).attr("stroke",special?P.B:P.line);
        txt(g, x+w/2, y+17, s, {anchor:"middle", size:11, mono:special, fill:special?P.B:P.ink}); x += w+4;
      });
      g.append("line").attr("x1",x+2).attr("x2",370).attr("y1",y+13).attr("y2",y+13).attr("stroke",P.muted).attr("marker-end",null);
      g.append("rect").attr("x",370).attr("y",y).attr("width",110).attr("height",26).attr("rx",5).attr("fill",P.panel).attr("stroke",P.A);
      txt(g, 425, y+17, "GPT-1 (shared)", {anchor:"middle", size:10.5, fill:P.A});
      g.append("line").attr("x1",480).attr("x2",500).attr("y1",y+13).attr("y2",y+13).attr("stroke",P.muted);
      txt(g, 504, y+17, "h at ⟨e⟩", {size:10, mono:true, fill:P.muted});
    });
    const yb = top + t.rows.length*rh + 10;
    g.append("rect").attr("x",140).attr("y",yb).attr("width",360).attr("height",26).attr("rx",5).attr("fill","rgba(74,222,128,.12)").attr("stroke",P.good);
    txt(g, 320, yb+17, t.combine, {anchor:"middle", size:11, fill:P.good});
    const specials = new Set(); t.rows.forEach(row=>row.forEach(s=>{ if (s===S||s===E||s===D) specials.add(s); }));
    const nSp = specials.size, newParams = nSp*d + d*t.out;
    d3.select("#ft-read").html(`<b>${t.rows.length}</b> forward pass${t.rows.length>1?"es":""} per example. New parameters: ${nSp} special-token embeddings (${[...specials].join(" ")}; ${nSp} × ${d}) + W_y (${d} × ${t.out}) = <b>${newParams.toLocaleString("en-US")}</b>, i.e. ${(100*newParams/base).toFixed(4)}% of the ${fmtN(base)} pretrained model. Loss: L₃ = L₂ + 0.5 · L₁.`);
  }
  d3.select("#ft-task").on("change", draw);
  draw();
});

/* ───────────────────────── 18 · few-shot prompt builder ───────────────────────── */
GV.safe("fewshot", function(){
  const {P, txt, lcg} = GV;
  const svg = d3.select("#shot-svg"), BUDGET = 2048;
  const PRE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;   /* GPT-2-style pre-tokeniser */
  const pieces = s => (s.match(PRE) || []).length;
  const TR = [["sea otter","loutre de mer"],["cheese","fromage"],["peppermint","menthe poivrée"],["plush giraffe","girafe en peluche"],["the red house","la maison rouge"],["I am hungry","j'ai faim"],["good morning","bonjour"],["thank you very much","merci beaucoup"],["the black cat","le chat noir"],["a cup of coffee","une tasse de café"],["where is the station?","où est la gare ?"],["it is raining","il pleut"]];
  const SA = [["A delight from start to finish.","positive"],["Two hours I will never get back.","negative"],["The acting was superb.","positive"],["Dull, predictable and far too long.","negative"],["I laughed the whole way through.","positive"],["The plot made no sense at all.","negative"],["A moving, beautiful film.","positive"],["Even the soundtrack was annoying.","negative"]];
  const TASK = {
    tr: {desc:"Translate English to French:", demo:i=>`${TR[i%TR.length][0]} => ${TR[i%TR.length][1]}`, query:"cheese board =>", cyc:TR.length},
    ar: {desc:"Add the two numbers.", demo:i=>{ const r = lcg(1000+i); const a = Math.floor(r()*90)+10, b = Math.floor(r()*90)+10; return `Q: What is ${a} plus ${b}? A: ${a+b}`; }, query:"Q: What is 48 plus 76? A:", cyc:Infinity},
    sa: {desc:"Classify the sentiment of each review.", demo:i=>`Review: ${SA[i%SA.length][0]} Sentiment: ${SA[i%SA.length][1]}`, query:"Review: Not bad at all, I would watch it again. Sentiment:", cyc:SA.length}
  };
  function draw(){
    svg.selectAll("*").remove();
    const key = d3.select("#shot-task").property("value"), K = +d3.select("#shot-K").property("value"), useD = d3.select("#shot-desc").property("checked"), t = TASK[key];
    d3.select("#shot-Kv").text(K);
    const lines = []; if (useD) lines.push({s:t.desc, kind:"desc"});
    for (let i=0;i<K;i++) lines.push({s:t.demo(i), kind:"demo"});
    lines.push({s:t.query, kind:"query"});
    const text = lines.map(l=>l.s).join("\n"), used = pieces(text) + (lines.length-1); /* newlines count as pieces */
    const fixed = (useD ? pieces(t.desc)+1 : 0) + pieces(t.query);
    const perDemo = K ? (used - fixed)/K : pieces(t.demo(0))+1;
    const maxK = Math.floor((BUDGET - fixed)/perDemo);
    const g = svg.append("g");
    const setting = K===0 ? "zero-shot" : (K===1 ? "one-shot" : `few-shot (K = ${K})`);
    txt(g, 16, 18, `prompt as the model sees it — ${setting}`, {size:10.5, fill:P.muted});
    const show = lines.length <= 9 ? lines : lines.slice(0,4).concat([{s:`… ${lines.length-6} more demonstrations …`, kind:"gap"}]).concat(lines.slice(-2));
    show.forEach((l,i)=>{
      const col = l.kind==="desc" ? P.purple : (l.kind==="query" ? P.B : (l.kind==="gap" ? P.muted : P.ink));
      const s = l.s.length > 78 ? l.s.slice(0,77)+"…" : l.s;
      txt(g, 20, 40 + i*20, s, {size:11, mono:true, fill:col});
    });
    /* budget bar */
    const by = 232, bw = 600, frac = Math.min(1, used/BUDGET);
    g.append("rect").attr("x",20).attr("y",by).attr("width",bw).attr("height",18).attr("rx",4).attr("fill",P.panel).attr("stroke",P.line);
    g.append("rect").attr("x",20).attr("y",by).attr("width",bw*frac).attr("height",18).attr("rx",4).attr("fill",used>BUDGET?P.bad:P.good).attr("fill-opacity",.75);
    txt(g, 20, by+36, `${used} of ${BUDGET} context positions (lower bound)`, {size:10.5, fill:used>BUDGET?P.bad:P.muted});
    txt(g, 620, by+36, `${(100*used/BUDGET).toFixed(1)}%`, {size:10.5, anchor:"end", fill:P.muted});
    const cyc = K > t.cyc ? ` (the toy example list has ${t.cyc} items, so demonstrations beyond that repeat)` : "";
    d3.select("#shot-read").html(`${setting}: ≈<b>${used}</b> pre-tokeniser pieces, ≈${perDemo.toFixed(1)} per demonstration. With this format at most <b>${maxK}</b> demonstrations fit in GPT-3's ${BUDGET}-token window before the query is squeezed out${used>BUDGET?` — <b style="color:${P.bad}">this prompt already overflows</b>`:""}.${cyc}`);
  }
  d3.select("#shot-task").on("change", draw);
  d3.select("#shot-K").on("input", draw);
  d3.select("#shot-desc").on("change", draw);
  draw();
});

/* ───────────────────────── 19 · GPT-3 size sweep (Table H.1 values) ───────────────────────── */
GV.safe("sweep", function(){
  const {P, fmtN, txt} = GV;
  const svg = d3.select("#sw-svg"), m = {l:50, r:110, t:20, b:40}, W = 640, H = 300;
  const SIZES = [125e6, 350e6, 760e6, 1.3e9, 2.7e9, 6.7e9, 13e9, 175e9];
  const DATA = { /* zero, one, few — Table H.1, arXiv 2005.14165 */
    lambada:{name:"LAMBADA accuracy", zero:[42.7,54.3,60.4,63.6,67.1,70.3,72.5,76.2], one:[22.0,47.1,52.6,58.3,61.1,65.4,69.0,72.5], few:[22.0,40.4,63.2,57.0,78.1,79.1,81.3,86.4], chance:null},
    trivia: {name:"TriviaQA accuracy", zero:[4.15,7.61,14.0,19.7,31.3,38.7,41.8,64.3], one:[4.19,12.9,20.5,26.5,35.9,44.4,51.3,68.0], few:[6.96,16.3,26.5,32.1,42.3,51.6,57.5,71.2], chance:null},
    add2:   {name:"2-digit addition", zero:[0.70,0.65,0.70,0.85,1.10,2.54,15.4,76.9], one:[2.00,0.55,3.15,4.00,12.1,19.6,73.0,99.6], few:[2.00,4.10,3.50,4.50,8.90,11.9,55.5,100.0], chance:null},
    add3:   {name:"3-digit addition", zero:[0.10,0.10,0.05,0.10,0.10,0.25,1.40,34.2], one:[0.15,0.00,0.10,0.30,0.45,0.95,15.4,65.5], few:[0.15,0.45,0.30,0.55,0.75,0.90,8.40,80.4], chance:null},
    anli:   {name:"ANLI round 3 accuracy", zero:[33.6,34.0,33.8,33.4,35.3,34.8,34.4,34.5], one:[35.0,32.6,33.0,33.9,34.1,33.1,32.5,35.1], few:[35.0,34.4,35.1,36.0,32.7,33.9,34.5,40.2], chance:100/3}
  };
  const SET = [["zero","zero-shot",P.muted],["one","one-shot",P.A],["few","few-shot",P.B]];
  const x = d3.scaleLog().domain([1e8, 2.5e11]).range([m.l, W-m.r]);
  function draw(){
    svg.selectAll("*").remove();
    const key = d3.select("#sw-task").property("value"), d = DATA[key], g = svg.append("g");
    const y = d3.scaleLinear().domain([0,100]).range([H-m.b, m.t]);
    [0,20,40,60,80,100].forEach(v=>{ g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3"); txt(g, m.l-6, y(v)+4, v+"%", {anchor:"end", size:9.5, fill:P.muted}); });
    SIZES.forEach(s=>txt(g, x(s), H-m.b+14, fmtN(s), {anchor:"middle", size:9, fill:P.muted}));
    txt(g, (m.l+W-m.r)/2, H-6, "parameters (log scale) — all trained on 300B tokens", {anchor:"middle", size:10, fill:P.muted});
    if (d.chance) { g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(d.chance)).attr("y2",y(d.chance)).attr("stroke",P.bad).attr("stroke-dasharray","4,3"); txt(g, W-m.r+4, y(d.chance)+4, "chance", {size:9.5, fill:P.bad}); }
    SET.forEach(([k,lab,col],si)=>{
      const pts = d[k].map((v,i)=>({s:SIZES[i], v}));
      g.append("path").datum(pts).attr("fill","none").attr("stroke",col).attr("stroke-width",2).attr("d", d3.line().x(p=>x(p.s)).y(p=>y(p.v)));
      pts.forEach(p=>g.append("circle").attr("cx",x(p.s)).attr("cy",y(p.v)).attr("r",4).attr("fill",col)
        .on("mouseenter", ()=>d3.select("#sw-read").html(`${d.name}, ${lab}, ${fmtN(p.s)}: <b>${p.v}%</b> (reported).`)));
      txt(g, W-m.r+8, y(pts[7].v)+4 + (si-1)*11, `${lab} ${pts[7].v}`, {size:10, fill:col});
    });
    const f = d.few, z = d.zero, jumps = f.slice(1).map((v,i)=>v - f[i]), jmax = Math.max(...jumps), ji = jumps.indexOf(jmax);
    d3.select("#sw-read").html(`${d.name}, 175B: zero ${z[7]}%, one ${d.one[7]}%, few <b>${f[7]}%</b> — few-shot minus zero-shot = <b>${(f[7]-z[7]).toFixed(1)}</b> points (at 125M: ${(f[0]-z[0]).toFixed(1)}). Largest few-shot step between adjacent sizes: <b>+${jmax.toFixed(1)}</b> points, ${fmtN(SIZES[ji])} → ${fmtN(SIZES[ji+1])}. Values as reported in the paper.`);
  }
  d3.select("#sw-task").on("change", draw);
  draw();
});

/* ───────────────────────── 24 · sampling explorer ───────────────────────── */
GV.safe("sampling", function(){
  const {P, softmax, entropyBits, txt, lcg} = GV;
  const svg = d3.select("#sp-svg");
  const TOK = [["mat",2.0,2],["rug",1.0,1],["sofa",0.8,0],["bed",0.5,0],["floor",0.0,0],["roof",-0.4,0],["moon",-1.0,0],["car",-1.6,0]]; /* token, logit, times already generated */
  let samples = null;
  function pipeline(){
    const T = +d3.select("#sp-T").property("value"), k = +d3.select("#sp-k").property("value"), pp = +d3.select("#sp-p").property("value"), pen = +d3.select("#sp-pen").property("value");
    const z = TOK.map(t=>t[1] - pen*(t[2]>0?1:0));
    const base = softmax(TOK.map(t=>t[1]));
    const pT = softmax(z, T);
    const order = pT.map((p,i)=>i).sort((a,b)=>pT[b]-pT[a]);
    const keep = new Set(); let cum = 0;
    for (let r=0;r<order.length;r++){ const i = order[r]; if (r >= k) break; if (cum >= pp - 1e-12) break; keep.add(i); cum += pT[i]; }
    const mass = [...keep].reduce((a,i)=>a+pT[i],0);
    const fin = pT.map((p,i)=>keep.has(i) ? p/mass : 0);
    return {T,k,pp,pen,base,pT,fin,keep,mass};
  }
  function draw(){
    svg.selectAll("*").remove();
    const r = pipeline(), g = svg.append("g");
    d3.select("#sp-Tv").text(r.T.toFixed(2)); d3.select("#sp-kv").text(r.k); d3.select("#sp-pv").text(r.pp.toFixed(2)); d3.select("#sp-penv").text(r.pen.toFixed(1));
    const x0 = 110, bw = 360, top = 30;
    txt(g, x0, 16, "grey outline: plain softmax (T = 1)  ·  filled: after penalty, temperature, top-k, top-p, renormalised", {size:9.5, fill:P.muted});
    TOK.forEach((t,i)=>{
      const yy = top + i*30, kept = r.keep.has(i);
      txt(g, x0-10, yy+14, t[0] + (t[2] ? ` (used ${t[2]}×)` : ""), {anchor:"end", mono:true, size:10.5, fill:kept?P.ink:P.muted});
      g.append("rect").attr("x",x0).attr("y",yy+2).attr("width",bw*r.base[i]).attr("height",16).attr("rx",3).attr("fill","none").attr("stroke",P.muted);
      g.append("rect").attr("x",x0).attr("y",yy+2).attr("width",bw*r.fin[i]).attr("height",16).attr("rx",3).attr("fill",kept?P.B:P.panel).attr("fill-opacity",.8);
      txt(g, x0+Math.max(bw*r.fin[i], bw*r.base[i])+6, yy+14, kept ? r.fin[i].toFixed(3) : "cut", {size:10, mono:true, fill:kept?P.ink:P.bad});
      if (samples){ txt(g, 625, yy+14, String(samples[i]), {anchor:"end", size:10, mono:true, fill:P.good}); }
    });
    if (samples) txt(g, 625, top-8, "drawn", {anchor:"end", size:9.5, fill:P.good});
    const H0 = entropyBits(r.base), H1 = entropyBits(r.fin), topi = r.fin.indexOf(Math.max(...r.fin));
    d3.select("#sp-read").html(`kept <b>${r.keep.size}</b> of ${TOK.length} tokens holding ${(100*r.mass).toFixed(1)}% of the tempered mass; entropy ${H0.toFixed(2)} → <b>${H1.toFixed(2)}</b> bits; most likely token <b>${TOK[topi][0]}</b> at ${r.fin[topi].toFixed(3)}.` +
      (samples ? ` Of 1,000 seeded draws, ${samples[topi]} were "${TOK[topi][0]}" (expected ${(1000*r.fin[topi]).toFixed(0)}).` : ""));
  }
  function drawSamples(){
    const r = pipeline(), rnd = lcg(42), cnt = TOK.map(()=>0);
    for (let s=0;s<1000;s++){ let u = rnd(), acc = 0, pick = TOK.length-1; for (let i=0;i<TOK.length;i++){ acc += r.fin[i]; if (u < acc){ pick = i; break; } } if (r.fin[pick]===0){ pick = r.fin.indexOf(Math.max(...r.fin)); } cnt[pick]++; }
    samples = cnt; draw();
  }
  ["sp-T","sp-k","sp-p","sp-pen"].forEach(id=>d3.select("#"+id).on("input", ()=>{ samples = null; draw(); }));
  d3.select("#sp-draw").on("click", drawSamples);
  draw();
});

/* ───────────────────────── 25 · KV-cache memory ───────────────────────── */
GV.safe("kvcache", function(){
  const {P, params, CFG, bytes, fmtN, txt} = GV;
  const svg = d3.select("#kv-svg"), m = {l:62, r:24, t:20, b:36}, W = 640, H = 260;
  function draw(){
    svg.selectAll("*").remove();
    const key = d3.select("#kv-model").property("value"), c = CFG[key], nCtx = +d3.select("#kv-n").property("value"), b = +d3.select("#kv-b").property("value"), by = +d3.select("#kv-bytes").property("value");
    d3.select("#kv-nv").text(nCtx); d3.select("#kv-bv").text(b);
    const N = params(c).total, wB = N*by, perTok = 2*c.L*c.d*by, maxN = c.n;
    const cache = n => perTok*n*b;
    const g = svg.append("g");
    const ymax = Math.max(wB, cache(maxN))*1.1;
    const x = d3.scaleLinear().domain([0,maxN]).range([m.l, W-m.r]), y = d3.scaleLinear().domain([0,ymax]).range([H-m.b, m.t]);
    y.ticks(4).forEach(t=>{ g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(t)).attr("y2",y(t)).attr("stroke",P.line).attr("stroke-dasharray","2,3"); txt(g, m.l-6, y(t)+4, bytes(t), {anchor:"end", size:9, fill:P.muted}); });
    x.ticks(5).forEach(t=>txt(g, x(t), H-m.b+14, String(t), {anchor:"middle", size:9.5, fill:P.muted}));
    txt(g, (m.l+W-m.r)/2, H-4, `tokens in context (model maximum ${maxN})`, {anchor:"middle", size:10, fill:P.muted});
    g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(wB)).attr("y2",y(wB)).attr("stroke",P.A).attr("stroke-dasharray","6,3").attr("stroke-width",1.5);
    txt(g, m.l+6, y(wB)-5, `weights ${bytes(wB)}`, {size:10, fill:P.A});
    g.append("path").datum(d3.range(0, maxN+1, maxN/64)).attr("fill","none").attr("stroke",P.B).attr("stroke-width",2).attr("d", d3.line().x(n=>x(n)).y(n=>y(cache(n))));
    g.append("circle").attr("cx",x(nCtx)).attr("cy",y(cache(nCtx))).attr("r",5).attr("fill",P.B);
    txt(g, W-m.r-4, y(cache(maxN))-6, `KV cache, batch ${b}`, {anchor:"end", size:10, fill:P.B});
    const perSeq = perTok*nCtx, total = cache(nCtx), eq = wB/perSeq;
    d3.select("#kv-read").html(`${c.name}: 2 · ${c.L} · ${c.d} · ${by} B = <b>${bytes(perTok)}</b> per token. At ${nCtx} tokens: <b>${bytes(perSeq)}</b> per sequence, <b>${bytes(total)}</b> for batch ${b} — ${(100*total/wB).toFixed(1)}% of the ${fmtN(N)}-parameter weights (${bytes(wB)}). The cache equals the weights at a batch of ≈<b>${eq.toFixed(eq<10?1:0)}</b> full sequences of this length.`);
  }
  ["kv-model","kv-bytes"].forEach(id=>d3.select("#"+id).on("change", draw));
  ["kv-n","kv-b"].forEach(id=>d3.select("#"+id).on("input", draw));
  draw();
});

/* ───────────────────────── 26 · InstructGPT pipeline + RM loss ───────────────────────── */
GV.safe("rlhf", function(){
  const {P, txt} = GV;
  const svg = d3.select("#rl-svg");
  const STAGES = [ /* numbers from arXiv 2203.02155 */
    {name:"1 · SFT", prompts:13e3, detail:"labellers write demonstrations; GPT-3 fine-tuned 16 epochs on them", col:P.A},
    {name:"2 · Reward model", prompts:33e3, detail:"labellers rank K = 4–9 outputs; a 6B model learns a scalar reward from all pairs", col:P.purple},
    {name:"3 · PPO (+ptx)", prompts:31e3, detail:"policy maximises RM reward − β·KL(π ‖ π_SFT), plus γ × pretraining log-likelihood", col:P.B}
  ];
  let sel = 1;
  const sig = v => 1/(1+Math.exp(-v)), choose2 = k => k*(k-1)/2;
  function draw(){
    svg.selectAll("*").remove();
    const gap = +d3.select("#rl-gap").property("value"), K = +d3.select("#rl-K").property("value");
    d3.select("#rl-gapv").text(gap.toFixed(1)); d3.select("#rl-Kv").text(K);
    const g = svg.append("g");
    STAGES.forEach((s,i)=>{
      const x = 16 + i*206, on = i===sel, node = g.append("g").style("cursor","pointer").on("click",()=>{ sel=i; draw(); });
      node.append("rect").attr("x",x).attr("y",14).attr("width",186).attr("height",54).attr("rx",8).attr("fill",on?"rgba(255,255,255,.04)":P.panel).attr("stroke",s.col).attr("stroke-width",on?2:1);
      txt(node, x+93, 36, s.name, {anchor:"middle", size:12, bold:true, fill:s.col});
      txt(node, x+93, 55, `≈${(s.prompts/1000).toFixed(0)}k training prompts`, {anchor:"middle", size:10, fill:P.muted});
      if (i<2) g.append("text").attr("x",x+193).attr("y",46).attr("fill",P.muted).attr("font-size",14).text("→");
    });
    txt(g, 16, 88, STAGES[sel].detail, {size:10.5, fill:P.ink});
    /* sigmoid panel */
    const x0 = 60, x1 = 380, y0 = 280, y1 = 110;
    const xs = d3.scaleLinear().domain([-3,3]).range([x0,x1]), ys = d3.scaleLinear().domain([0,1]).range([y0,y1]);
    g.append("line").attr("x1",x0).attr("x2",x1).attr("y1",y0).attr("y2",y0).attr("stroke",P.muted);
    g.append("line").attr("x1",xs(0)).attr("x2",xs(0)).attr("y1",y0).attr("y2",y1).attr("stroke",P.line);
    [0,0.5,1].forEach(v=>txt(g, x0-6, ys(v)+4, v.toFixed(1), {anchor:"end", size:9.5, fill:P.muted}));
    [-3,0,3].forEach(v=>txt(g, xs(v), y0+13, String(v), {anchor:"middle", size:9.5, fill:P.muted}));
    txt(g, x0, y1-6, "P(labeller prefers y_w) = σ(r_w − r_l)", {size:10, fill:P.muted});
    g.append("path").datum(d3.range(-3,3.01,0.1)).attr("fill","none").attr("stroke",P.purple).attr("stroke-width",2).attr("d", d3.line().x(v=>xs(v)).y(v=>ys(sig(v))));
    g.append("circle").attr("cx",xs(gap)).attr("cy",ys(sig(gap))).attr("r",5).attr("fill",P.B);
    const pairs = choose2(K), loss = -Math.log(sig(gap));
    const bx = 420;
    txt(g, bx, 130, `ranking K = ${K} outputs gives`, {size:10.5, fill:P.muted});
    txt(g, bx, 152, `C(${K},2) = ${pairs} comparisons`, {size:12, mono:true, fill:P.ink});
    txt(g, bx, 180, `RM forward passes: ${K} (batched)`, {size:10.5, fill:P.muted});
    txt(g, bx, 198, `vs ${2*pairs} if pairs were separate`, {size:10.5, fill:P.muted});
    txt(g, bx, 232, `pair loss −ln σ(${gap.toFixed(1)}) = ${loss.toFixed(3)}`, {size:11, mono:true, fill:P.B});
    d3.select("#rl-read").html(`<b>${STAGES[sel].name}</b> selected. For a reward gap of ${gap.toFixed(1)}, the RM predicts the preferred answer with probability <b>${sig(gap).toFixed(3)}</b> and that pair's loss is <b>${loss.toFixed(3)}</b>. Ranking ${K} outputs yields <b>${pairs}</b> training pairs from one labelling task.`);
  }
  d3.select("#rl-gap").on("input", draw);
  d3.select("#rl-K").on("input", draw);
  draw();
});
