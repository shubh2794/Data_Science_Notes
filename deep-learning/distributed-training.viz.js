/* distributed-training.viz.js — every interactive figure on deep-learning/distributed-training.html.
   Loaded after data.js / notes.js. Each figure is its own IIFE wrapped in try/catch so one
   failure cannot blank the rest. House rule: every number a figure displays is computed here
   from its inputs (a model spec, a link rate, a schedule simulation), never typed into a label. */

const DT = (function(){
  const P = { A:"#5b9cff", B:"#ffb454", good:"#4ade80", bad:"#f87171", ink:"#e6e9ef", muted:"#9aa3b2",
              line:"#2a2f3a", panel:"#14171f", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6", grey:"#6b7280" };
  const $ = id => document.getElementById(id);
  const val = id => { const e = $(id); return e ? e.value : null; };
  const num = id => +val(id);
  function gb(bytes){ const g = bytes/1e9; return g >= 100 ? g.toFixed(0) : g >= 10 ? g.toFixed(1) : g >= 1 ? g.toFixed(2) : g.toFixed(3); }
  function secs(s){ if (!isFinite(s)) return "∞"; if (s >= 1) return s.toFixed(2)+" s"; if (s >= 1e-3) return (s*1e3).toFixed(1)+" ms"; return (s*1e6).toFixed(1)+" µs"; }
  function pct(x, d){ return (100*x).toFixed(d == null ? 1 : d)+"%"; }
  function big(x){ return x.toLocaleString("en-US"); }
  /* transformer configurations used by several figures (hidden size, layers, heads, parameters) */
  const MODELS = {
    "1.3": {psi:1.3e9, h:2048,  L:24,  a:16},
    "7":   {psi:7e9,   h:4096,  L:32,  a:32},
    "7.5": {psi:7.5e9, h:4096,  L:36,  a:32},
    "13":  {psi:13e9,  h:5120,  L:40,  a:40},
    "70":  {psi:70e9,  h:8192,  L:80,  a:64},
    "175": {psi:175e9, h:12288, L:96,  a:96},
    "405": {psi:405e9, h:16384, L:126, a:128}
  };
  function on(ids, fn){ ids.forEach(id => { const e = $(id); if (e){ e.addEventListener("input", fn); e.addEventListener("change", fn); } }); }
  return {P, $, val, num, gb, secs, pct, big, MODELS, on};
})();

/* ── Figure 1: how work is split across devices (schematic) ─────────────────────────── */
(function(){ try {
  const C={accent:"#f87171",a2:"#ffb454",good:"#4ade80",blue:"#5b9cff",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#dt-svg"),W=640, G=4;
  const layerCols=["#5b9cff","#4ade80","#fbbf24","#f87171"];
  function gpuBox(i){ const x=40+i*150; return {x,y:60,w:120,h:130}; }
  function draw(){
    svg.selectAll("*").remove();
    const k=d3.select("#dt-sel").property("value");
    let note="";
    for(let i=0;i<G;i++){
      const g=gpuBox(i);
      svg.append("rect").attr("x",g.x).attr("y",g.y).attr("width",g.w).attr("height",g.h).attr("rx",8).attr("fill","#14171f").attr("stroke",C.line);
      svg.append("text").attr("x",g.x+g.w/2).attr("y",g.y-8).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("GPU "+(i+1));
      if(k==="data"){
        for(let l=0;l<4;l++) svg.append("rect").attr("x",g.x+12).attr("y",g.y+12+l*26).attr("width",g.w-24).attr("height",20).attr("rx",3).attr("fill",layerCols[l]).attr("fill-opacity",.8);
        svg.append("text").attr("x",g.x+g.w/2).attr("y",g.y+g.h+16).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.a2).text("batch "+(i+1)+"/"+G);
      } else if(k==="tensor"){
        for(let l=0;l<4;l++) svg.append("rect").attr("x",g.x+12).attr("y",g.y+12+l*26).attr("width",g.w-24).attr("height",20).attr("rx",3).attr("fill",layerCols[l]).attr("fill-opacity",.35).attr("stroke",layerCols[l]).attr("stroke-dasharray","2 2");
        svg.append("text").attr("x",g.x+g.w/2).attr("y",g.y+g.h+16).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text("1/"+G+" of each layer");
      } else {
        svg.append("rect").attr("x",g.x+12).attr("y",g.y+50).attr("width",g.w-24).attr("height",30).attr("rx",4).attr("fill",layerCols[i]).attr("fill-opacity",.85);
        svg.append("text").attr("x",g.x+g.w/2).attr("y",g.y+70).attr("text-anchor","middle").attr("font-size",10).attr("fill","#0f1117").attr("font-weight",600).text("layer "+(i+1));
        svg.append("text").attr("x",g.x+g.w/2).attr("y",g.y+g.h+16).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text("stage "+(i+1));
      }
    }
    for(let i=0;i<G-1;i++){
      const g=gpuBox(i);
      svg.append("text").attr("x",g.x+g.w+15).attr("y",g.y+g.h/2+4).attr("text-anchor","middle").attr("font-size",14).attr("fill",C.muted).text(k==="pipeline"?"→":"↔");
    }
    const head={data:"full model on every GPU · gradients all-reduced ↔",tensor:"each layer's matrix sliced across GPUs · combine each layer ↔",pipeline:"different layers on different GPUs · micro-batches flow →"}[k];
    svg.append("text").attr("x",W/2).attr("y",30).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text(head);
    note={data:"Data parallel: replicate the model, split the batch, average gradients — for throughput.",
          tensor:"Tensor parallel: split each layer across GPUs — for a model too big per device (heavy comms).",
          pipeline:"Pipeline parallel: stage layers across GPUs like an assembly line (mind the bubble)."}[k];
    d3.select("#dt-read").html(note);
  }
  d3.select("#dt-sel").on("change",draw);
  draw();
} catch(e){ console.error("viz-dt", e); } })();

/* ── Figure 2: per-GPU model-state memory, DDP vs ZeRO stages ───────────────────────── */
(function(){ try {
  const {P, num, val, gb, on, MODELS} = DT;
  const svg = d3.select("#mem-svg"), W = 760, H = 330;
  const STRATS = [
    {k:"ddp", name:"DDP",           w:(K,N)=>[2,2,K]},
    {k:"z1",  name:"ZeRO-1",        w:(K,N)=>[2,2,K/N]},
    {k:"z2",  name:"ZeRO-2",        w:(K,N)=>[2,2/N,K/N]},
    {k:"z3",  name:"ZeRO-3 / FSDP", w:(K,N)=>[2/N,2/N,K/N]}
  ];
  const SEG = [{n:"16-bit weights",c:P.A},{n:"gradients",c:P.B},{n:"optimiser state",c:P.purple}];
  function draw(){
    const M = MODELS[val("mem-model")], psi = M.psi, N = num("mem-n"), K = num("mem-opt"),
          off = val("mem-off"), cap = num("mem-gpu")*1e9, sel = val("mem-strat");
    /* per GPU bytes for each strategy, then move offloaded states to the host */
    const rows = STRATS.map(s => {
      const per = s.w(K,N).map(x => x*psi);            // [weights, grads, opt] bytes per GPU
      const gpu = per.slice(), host = [0,0,0];
      let note = "";
      if (off !== "none"){
        if (s.k === "ddp") note = "offload needs a ZeRO stage";
        else {
          host[1] = gpu[1]; host[2] = gpu[2]; gpu[1] = 0; gpu[2] = 0;
          if (off === "all"){ if (s.k === "z3"){ host[0] = gpu[0]; gpu[0] = 0; } else note = "weight offload is ZeRO-3 only"; }
        }
      }
      return {s, per, gpu, host, tot: d3.sum(gpu), htot: d3.sum(host), note};
    });
    svg.selectAll("*").remove();
    const x0 = 130, x1 = W - 30;
    const maxV = Math.max(cap*1.15, d3.max(rows, r => r.tot), d3.max(rows, r => r.htot));
    const x = d3.scaleLinear().domain([0, maxV]).range([x0, x1]);
    const bh = 30, gap = 34, top = 40;
    svg.append("text").attr("x", x0).attr("y", 18).attr("fill", P.muted).attr("font-size", 11)
      .text(`${val("mem-model")} B parameters · N = ${N} · K = ${K} bytes/param of optimiser state · per-GPU model state, GB`);
    const ax = d3.axisBottom(x).ticks(6).tickFormat(d => (d/1e9).toFixed(0));
    const g = svg.append("g").attr("transform", `translate(0,${top + rows.length*(bh+gap) + 4})`).call(ax);
    g.selectAll("text").attr("fill", P.muted); g.selectAll("line,path").attr("stroke", P.line);
    rows.forEach((r, i) => {
      const y = top + i*(bh+gap);
      svg.append("text").attr("x", x0-8).attr("y", y+bh/2+4).attr("text-anchor","end").attr("fill", r.s.k===sel?P.ink:P.muted)
        .attr("font-size", 12).attr("font-weight", r.s.k===sel?700:400).text(r.s.name);
      let cx = x0;
      r.gpu.forEach((v, j) => {
        if (v <= 0) return;
        const w = Math.max(0.5, x(v) - x0);
        svg.append("rect").attr("x", cx).attr("y", y).attr("width", w).attr("height", bh).attr("fill", SEG[j].c).attr("fill-opacity", .8);
        cx += w;
      });
      if (r.s.k === sel) svg.append("rect").attr("x", x0-3).attr("y", y-3).attr("width", Math.max(6, x(r.tot)-x0+6)).attr("height", bh+6)
        .attr("fill","none").attr("stroke", P.ink).attr("stroke-width", 1.5).attr("rx", 3);
      const fits = r.tot <= cap;
      svg.append("text").attr("x", Math.min(cx + 6, x1 - 150)).attr("y", y+bh/2+4).attr("fill", fits?P.good:P.bad).attr("font-size", 11)
        .text(`${gb(r.tot)} GB ${fits?"fits":"does not fit"}${r.note?" · "+r.note:""}`);
      if (r.htot > 0){
        svg.append("rect").attr("x", x0).attr("y", y+bh+3).attr("width", Math.max(0.5, x(r.htot)-x0)).attr("height", 8).attr("fill", P.grey);
        svg.append("text").attr("x", x0 + Math.max(0.5, x(r.htot)-x0) + 6).attr("y", y+bh+11).attr("fill", P.muted).attr("font-size", 10)
          .text(`host ${gb(r.htot)} GB per GPU`);
      }
    });
    const xc = x(cap);
    svg.append("line").attr("x1", xc).attr("x2", xc).attr("y1", top-10).attr("y2", top + rows.length*(bh+gap)).attr("stroke", P.bad).attr("stroke-dasharray","4 3");
    svg.append("text").attr("x", xc).attr("y", top-14).attr("text-anchor","middle").attr("fill", P.bad).attr("font-size", 10).text(`GPU ${num("mem-gpu")} GB`);
    SEG.forEach((s, j) => {
      svg.append("rect").attr("x", x0 + j*140).attr("y", H-18).attr("width", 10).attr("height", 10).attr("fill", s.c);
      svg.append("text").attr("x", x0 + j*140 + 14).attr("y", H-9).attr("fill", P.muted).attr("font-size", 10).text(s.n);
    });
    svg.append("rect").attr("x", x0 + 3*140).attr("y", H-18).attr("width", 10).attr("height", 10).attr("fill", P.grey);
    svg.append("text").attr("x", x0 + 3*140 + 14).attr("y", H-9).attr("fill", P.muted).attr("font-size", 10).text("offloaded to host");
    const r = rows.find(q => q.s.k === sel), ddp = rows[0];
    const minN = Math.ceil((2+2+K)*psi / cap);
    d3.select("#mem-read").html(
      `<b>${r.s.name}</b>: ${gb(r.per[0])} GB weights + ${gb(r.per[1])} GB gradients + ${gb(r.per[2])} GB optimiser = <b>${gb(d3.sum(r.per))} GB</b> per GPU`+
      (r.htot>0?`, of which ${gb(r.htot)} GB lives in host memory (GPU keeps ${gb(r.tot)} GB)`:"")+
      `, ${ (ddp.tot/Math.max(1e-9,d3.sum(r.per))).toFixed(1)}× less than DDP's ${gb(ddp.tot)} GB. `+
      `${r.tot<=cap?"It fits the "+num("mem-gpu")+" GB budget before activations":"It does <b>not</b> fit "+num("mem-gpu")+" GB"}. `+
      `Only ZeRO-3 shards all ${2+2+K} bytes/param: it needs N ≥ ${minN} such GPUs just to hold the model state.`);
  }
  on(["mem-model","mem-n","mem-strat","mem-opt","mem-off","mem-gpu"], draw);
  draw();
} catch(e){ console.error("viz-mem", e); } })();

/* ── Figure 3: communication against compute per optimiser step ─────────────────────── */
(function(){ try {
  const {P, num, val, secs, pct, on, MODELS, $} = DT;
  const svg = d3.select("#cm-svg"), W = 760, H = 300;
  function draw(){
    const psi = MODELS[val("cm-model")].psi, n = num("cm-n"), beta = num("cm-link")*1e9;
    const tok = Math.pow(2, num("cm-tok")), tf = num("cm-tf")*1e12;
    $("cm-tokv").textContent = DT.big(tok); $("cm-tfv").textContent = String(num("cm-tf"));
    const S = 2*psi;                                   // 16-bit gradient bytes
    const f = (n-1)/n;
    const rows = [
      {n:"DDP · all-reduce",               b:2*f*S, c:P.A},
      {n:"ZeRO-2 · reduce-scatter + gather", b:2*f*S, c:P.B},
      {n:"ZeRO-3 · 2 gathers + scatter",     b:3*f*S, c:P.purple}
    ].map(r => Object.assign(r, {t: r.b/beta}));
    const comp = 6*psi*tok/tf;
    svg.selectAll("*").remove();
    const x0 = 210, x1 = W-40, top = 30, bh = 34, gap = 22;
    const all = rows.concat([{n:"compute · 6·Ψ·tokens", t:comp, c:P.good}]);
    const x = d3.scaleLinear().domain([0, d3.max(all, r => r.t)*1.1]).range([x0, x1]);
    all.forEach((r, i) => {
      const y = top + i*(bh+gap);
      svg.append("text").attr("x", x0-8).attr("y", y+bh/2+4).attr("text-anchor","end").attr("fill", P.ink).attr("font-size", 12).text(r.n);
      svg.append("rect").attr("x", x0).attr("y", y).attr("width", Math.max(1, x(r.t)-x0)).attr("height", bh).attr("fill", r.c).attr("fill-opacity", .8);
      const lbl = secs(r.t) + (r.b ? ` · ${(r.b/1e9).toFixed(1)} GB sent per rank` : "");
      const lx = x(r.t)+6 > x1-150 ? x0+6 : x(r.t)+6;
      svg.append("text").attr("x", lx).attr("y", y+bh/2+4).attr("fill", lx===x0+6?"#0f1117":P.muted).attr("font-size", 11).text(lbl);
    });
    const yb = top + all.length*(bh+gap);
    svg.append("line").attr("x1", x(comp)).attr("x2", x(comp)).attr("y1", top-10).attr("y2", yb-gap+4).attr("stroke", P.good).attr("stroke-dasharray","4 3");
    const ax = d3.axisBottom(x).ticks(6).tickFormat(d => secs(d));
    const g = svg.append("g").attr("transform", `translate(0,${yb-gap+8})`).call(ax);
    g.selectAll("text").attr("fill", P.muted); g.selectAll("line,path").attr("stroke", P.line);
    svg.append("text").attr("x", x0).attr("y", 16).attr("fill", P.muted).attr("font-size", 11)
      .text(`Ψ = ${val("cm-model")} B · n = ${n} · ${num("cm-link")} GB/s per GPU · ${DT.big(tok)} tokens per GPU · ${num("cm-tf")} TFLOP/s`);
    const r0 = rows[0].t/comp, r3 = rows[2].t/comp;
    /* tokens per GPU at which DDP traffic equals compute: 2·f·2Ψ/β = 6Ψ·tok/F  →  tok* = 2·f·F/(3β)·… */
    const tokEq = (2*f*S/beta) * tf / (6*psi);
    d3.select("#cm-read").html(
      `DDP moves <b>${(rows[0].b/1e9).toFixed(1)} GB</b> per rank in ${secs(rows[0].t)}; compute takes ${secs(comp)} — communication is <b>${pct(r0)}</b> of compute`+
      ` (ZeRO-3: ${pct(r3)}). ${r0<1?"Bucketed overlap can hide it":"<b>It cannot all be hidden</b>: the step is communication-bound"}. `+
      `Break-even at about ${DT.big(Math.round(tokEq))} tokens per GPU per step; below that, adding ranks stops paying.`);
  }
  on(["cm-model","cm-n","cm-link","cm-tok","cm-tf"], draw);
  draw();
} catch(e){ console.error("viz-comm", e); } })();

/* ── Figure 4: ring all-reduce, step by step ────────────────────────────────────────── */
(function(){ try {
  const {P, num, val, secs, on, $} = DT;
  const svg = d3.select("#rg-svg"), W = 760, H = 330;
  /* state[r][c] = bitmask of ranks whose contribution to chunk c rank r currently holds */
  function simulate(n, steps){
    const st = [];
    for (let r=0;r<n;r++){ st.push([]); for (let c=0;c<n;c++) st[r].push(1<<r); }
    const sends = [];
    for (let k=0;k<steps;k++){
      const msgs = [];
      for (let r=0;r<n;r++){
        const c = k < n-1 ? ((r - k) % n + n) % n : ((r + 1 - (k-(n-1))) % n + n) % n;
        msgs.push({from:r, to:(r+1)%n, c, mask:st[r][c]});
      }
      msgs.forEach(m => { st[m.to][m.c] = k < n-1 ? (st[m.to][m.c] | m.mask) : m.mask; });
      sends.push(msgs);
    }
    return {st, last: sends[sends.length-1] || null};
  }
  function bits(m){ let c=0; while(m){ c += m&1; m >>= 1; } return c; }
  function draw(){
    const n = num("rg-n"), stepEl = $("rg-step");
    stepEl.max = String(2*(n-1));
    let k = Math.min(num("rg-step"), 2*(n-1));
    if (+stepEl.value !== k) stepEl.value = String(k);
    $("rg-stepv").textContent = `${k} / ${2*(n-1)}`;
    const S = num("rg-s")*1e6, beta = num("rg-link")*1e9, alpha = num("rg-a")*1e-6;
    const {st, last} = simulate(n, k);
    svg.selectAll("*").remove();
    const cw = Math.min(46, 360/n), ch = Math.min(30, 230/n), gx = 90, gy = 50;
    svg.append("text").attr("x", gx).attr("y", 20).attr("fill", P.muted).attr("font-size", 11).text("rows = ranks, columns = the n chunks of each rank's buffer; fill = share of the n contributions summed in");
    for (let c=0;c<n;c++) svg.append("text").attr("x", gx + c*cw + cw/2).attr("y", gy-6).attr("text-anchor","middle").attr("fill", P.muted).attr("font-size", 10).text("chunk "+c);
    const col = d3.scaleLinear().domain([1, n]).range(["#1e2a44", P.A]);
    for (let r=0;r<n;r++){
      svg.append("text").attr("x", gx-8).attr("y", gy + r*ch + ch/2 + 4).attr("text-anchor","end").attr("fill", P.ink).attr("font-size", 11).text("rank "+r);
      for (let c=0;c<n;c++){
        const b = bits(st[r][c]);
        svg.append("rect").attr("x", gx + c*cw + 1).attr("y", gy + r*ch + 1).attr("width", cw-2).attr("height", ch-2).attr("rx", 3)
          .attr("fill", b===n ? P.good : col(b)).attr("fill-opacity", b===n?.85:1);
        svg.append("text").attr("x", gx + c*cw + cw/2).attr("y", gy + r*ch + ch/2 + 4).attr("text-anchor","middle").attr("font-size", 10)
          .attr("fill", b===n ? "#0f1117" : P.ink).text(b+"/"+n);
      }
    }
    if (last) last.forEach(m => {
      svg.append("rect").attr("x", gx + m.c*cw + 1).attr("y", gy + m.to*ch + 1).attr("width", cw-2).attr("height", ch-2).attr("rx", 3)
        .attr("fill","none").attr("stroke", P.B).attr("stroke-width", 2);
    });
    /* ring diagram */
    const cx = 610, cy = 160, R = 95;
    for (let r=0;r<n;r++){
      const a = -Math.PI/2 + 2*Math.PI*r/n, a2 = -Math.PI/2 + 2*Math.PI*((r+1)%n)/n;
      const px = cx + R*Math.cos(a), py = cy + R*Math.sin(a), qx = cx + R*Math.cos(a2), qy = cy + R*Math.sin(a2);
      svg.append("line").attr("x1", px).attr("y1", py).attr("x2", px + 0.8*(qx-px)).attr("y2", py + 0.8*(qy-py))
        .attr("stroke", last ? P.B : P.line).attr("stroke-width", last ? 2 : 1);
      svg.append("circle").attr("cx", px).attr("cy", py).attr("r", 15).attr("fill", P.panel).attr("stroke", P.A);
      svg.append("text").attr("x", px).attr("y", py+4).attr("text-anchor","middle").attr("fill", P.ink).attr("font-size", 11).text(r);
      if (last){ const mx = (px+qx)/2, my = (py+qy)/2; svg.append("text").attr("x", cx + 1.25*(mx-cx)).attr("y", cy + 1.25*(my-cy)+4).attr("text-anchor","middle").attr("fill", P.B).attr("font-size", 10).text("c"+last[r].c); }
    }
    const phase = k === 0 ? "start" : k <= n-1 ? `reduce-scatter step ${k} of ${n-1}` : `all-gather step ${k-(n-1)} of ${n-1}`;
    svg.append("text").attr("x", cx).attr("y", cy + R + 42).attr("text-anchor","middle").attr("fill", P.B).attr("font-size", 12).text(phase);
    const done = k*S/n, tot = 2*(n-1)/n*S;
    const tBw = tot/beta, tLat = 2*(n-1)*alpha;
    d3.select("#rg-read").html(
      `After ${k} step${k===1?"":"s"} each rank has sent ${ (done/1e6).toFixed(1)} MB of the ${ (tot/1e6).toFixed(1)} MB a full all-reduce needs (2·(n−1)/n = ${(2*(n-1)/n).toFixed(3)} × the ${num("rg-s")} MB buffer). `+
      `Time model: ${2*(n-1)} × ${num("rg-a")} µs latency = ${secs(tLat)} plus ${secs(tBw)} of bandwidth → <b>${secs(tLat+tBw)}</b>; latency is ${DT.pct(tLat/(tLat+tBw))} of it.`+
      (k===2*(n-1) ? " Every cell is green: every rank holds the full sum of every chunk." : k===n-1 ? " End of reduce-scatter: each rank owns one fully reduced chunk — exactly what ZeRO-2/3 need." : ""));
  }
  on(["rg-n","rg-step","rg-s","rg-link","rg-a"], draw);
  const nx = $("rg-next"); if (nx) nx.addEventListener("click", () => { const e = $("rg-step"); e.value = String((+e.value + 1) % (+e.max + 1)); draw(); });
  draw();
} catch(e){ console.error("viz-ring", e); } })();

/* ── Figure 5: Megatron tensor-parallel MLP and its communication ───────────────────── */
(function(){ try {
  const {P, num, val, secs, pct, on, MODELS} = DT;
  const svg = d3.select("#tp-svg"), W = 760, H = 340;
  function draw(){
    const t = num("tp-t"), M = MODELS[val("tp-model")], h = M.h, beta = num("tp-link")*1e9, F = num("tp-tf")*1e12, bs = num("tp-tok");
    svg.selectAll("*").remove();
    const cols = [P.A, P.B, P.good, P.purple, P.pink, P.teal, "#fbbf24", "#a3e635"];
    const shown = Math.min(t, 8);
    /* left: X → f → [X·A_i] → GeLU → [·B_i] → g → Z */
    const y0 = 40, lane = Math.min(34, 200/shown);
    svg.append("rect").attr("x", 20).attr("y", y0 + 80).attr("width", 50).attr("height", 60).attr("rx", 4).attr("fill", P.panel).attr("stroke", P.ink);
    svg.append("text").attr("x", 45).attr("y", y0 + 114).attr("text-anchor","middle").attr("fill", P.ink).attr("font-size", 12).text("X");
    svg.append("text").attr("x", 45).attr("y", y0 + 156).attr("text-anchor","middle").attr("fill", P.muted).attr("font-size", 9).text("b·s × h");
    svg.append("rect").attr("x", 85).attr("y", y0 + 95).attr("width", 26).attr("height", 30).attr("rx", 13).attr("fill", P.panel).attr("stroke", P.B);
    svg.append("text").attr("x", 98).attr("y", y0 + 114).attr("text-anchor","middle").attr("fill", P.B).attr("font-size", 12).text("f");
    for (let i=0;i<shown;i++){
      const y = y0 + i*lane + (220 - shown*lane)/2;
      svg.append("line").attr("x1", 111).attr("y1", y0+110).attr("x2", 140).attr("y2", y + lane/2).attr("stroke", P.line);
      svg.append("rect").attr("x", 140).attr("y", y+2).attr("width", 90).attr("height", lane-4).attr("rx", 3).attr("fill", cols[i%8]).attr("fill-opacity", .75);
      svg.append("text").attr("x", 185).attr("y", y + lane/2 + 4).attr("text-anchor","middle").attr("fill", "#0f1117").attr("font-size", 10).text(`GeLU(X·A${i+1})`);
      svg.append("rect").attr("x", 250).attr("y", y+2).attr("width", 80).attr("height", lane-4).attr("rx", 3).attr("fill", cols[i%8]).attr("fill-opacity", .45);
      svg.append("text").attr("x", 290).attr("y", y + lane/2 + 4).attr("text-anchor","middle").attr("fill", P.ink).attr("font-size", 10).text(`·B${i+1} = Z${i+1}`);
      svg.append("line").attr("x1", 330).attr("y1", y + lane/2).attr("x2", 360).attr("y2", y0+110).attr("stroke", P.line);
    }
    if (t > shown) svg.append("text").attr("x", 235).attr("y", y0 + 232).attr("text-anchor","middle").attr("fill", P.muted).attr("font-size", 10).text(`… ${t} lanes in total, ${shown} drawn`);
    svg.append("rect").attr("x", 360).attr("y", y0 + 95).attr("width", 26).attr("height", 30).attr("rx", 13).attr("fill", P.panel).attr("stroke", P.bad);
    svg.append("text").attr("x", 373).attr("y", y0 + 114).attr("text-anchor","middle").attr("fill", P.bad).attr("font-size", 12).text("g");
    svg.append("rect").attr("x", 400).attr("y", y0 + 80).attr("width", 50).attr("height", 60).attr("rx", 4).attr("fill", P.panel).attr("stroke", P.ink);
    svg.append("text").attr("x", 425).attr("y", y0 + 114).attr("text-anchor","middle").attr("fill", P.ink).attr("font-size", 12).text("Z");
    svg.append("text").attr("x", 185).attr("y", 24).attr("text-anchor","middle").attr("fill", P.muted).attr("font-size", 10).text(`A split by columns: h × ${DT.big(4*h/t)} each`);
    svg.append("text").attr("x", 290).attr("y", 36).attr("text-anchor","middle").attr("fill", P.muted).attr("font-size", 10).text(`B split by rows: ${DT.big(4*h/t)} × h each`);
    svg.append("text").attr("x", 235).attr("y", H-60).attr("text-anchor","middle").attr("fill", P.muted).attr("font-size", 10).text("f: identity forward, all-reduce backward · g: all-reduce forward, identity backward");
    /* right: comm/compute ratio against t for this h and link */
    const gx0 = 500, gx1 = W-20, gy0 = 50, gy1 = H-70;
    const ts = [1,2,4,8,16,32];
    const ratio = tt => 2*(tt-1)*F/(9*h*beta);
    const ymax = Math.max(1.2, d3.max(ts, ratio)*1.1);
    const x = d3.scaleBand().domain(ts).range([gx0, gx1]).padding(.25), y = d3.scaleLinear().domain([0, ymax]).range([gy1, gy0]);
    ts.forEach(tt => {
      const r = ratio(tt);
      svg.append("rect").attr("x", x(tt)).attr("y", y(Math.min(r, ymax))).attr("width", x.bandwidth()).attr("height", gy1 - y(Math.min(r, ymax)))
        .attr("fill", r > 1 ? P.bad : tt===t ? P.B : P.A).attr("fill-opacity", tt===t?1:.55);
      svg.append("text").attr("x", x(tt)+x.bandwidth()/2).attr("y", gy1+14).attr("text-anchor","middle").attr("fill", P.muted).attr("font-size", 10).text("t="+tt);
      svg.append("text").attr("x", x(tt)+x.bandwidth()/2).attr("y", y(Math.min(r, ymax))-4).attr("text-anchor","middle").attr("fill", P.ink).attr("font-size", 9).text(r.toFixed(2));
    });
    svg.append("line").attr("x1", gx0).attr("x2", gx1).attr("y1", y(1)).attr("y2", y(1)).attr("stroke", P.bad).attr("stroke-dasharray","4 3");
    svg.append("text").attr("x", gx1).attr("y", y(1)-4).attr("text-anchor","end").attr("fill", P.bad).attr("font-size", 9).text("comm = compute");
    svg.append("text").attr("x", (gx0+gx1)/2).attr("y", gy0-18).attr("text-anchor","middle").attr("fill", P.muted).attr("font-size", 10).text("TP communication ÷ compute, per layer, fwd + bwd");
    /* numbers for the chosen t */
    const arBytes = 2*bs*h;                                         // one all-reduce of the block output, 16-bit
    const perLayer = 4*2*(t-1)/t*arBytes;                           // 2 fwd + 2 bwd all-reduces, bytes sent per rank
    const flops = 72*bs*h*h/t;                                      // 6 × 12h² parameters × tokens ÷ t
    const tc = perLayer/beta, tf = flops/F;
    d3.select("#tp-read").html(
      `h = ${DT.big(h)}, t = ${t}, ${DT.big(bs)} tokens per micro-batch: each rank holds ${DT.big(12*h*h/t)} of the block's ${DT.big(12*h*h)} weights (${(24*h*h/t/1e6).toFixed(0)} MB in 16-bit). `+
      `Per layer it sends ${(perLayer/1e6).toFixed(0)} MB (4 all-reduces of a ${(arBytes/1e6).toFixed(0)} MB activation) — ${secs(tc)} — against ${secs(tf)} of compute: `+
      `<b>${pct(tc/Math.max(tf,1e-12))}</b>, which sits on the critical path. Over all ${M.L} layers: ${(perLayer*M.L/1e9).toFixed(1)} GB per micro-batch.`+
      (t===1 ? " With t = 1 nothing is split and nothing is sent." : ""));
  }
  on(["tp-t","tp-model","tp-link","tp-tf","tp-tok"], draw);
  draw();
} catch(e){ console.error("viz-tp", e); } })();

/* ── Figure 6: activation memory per layer (Megatron sequence parallelism) ─────────── */
(function(){ try {
  const {P, num, val, on, MODELS, $} = DT;
  const svg = d3.select("#ac-svg"), W = 760, H = 300;
  function draw(){
    const M = MODELS[val("ac-model")], h = M.h, a = M.a, L = M.L;
    const s = Math.pow(2, num("ac-s")), b = num("ac-b"), t = num("ac-t");
    $("ac-sv").textContent = DT.big(s);
    const sbh = s*b*h, att = 5*a*s/h;
    const rows = [
      {n:"no parallelism",              v: sbh*(34 + att)},
      {n:`tensor parallel t = ${t}`,     v: sbh*(10 + 24/t + att/t)},
      {n:"+ sequence parallel",         v: sbh*(34/t + att/t)},
      {n:"+ selective recompute",       v: sbh*34/t},
      {n:"full recompute (input only)", v: sbh*2/t}
    ];
    svg.selectAll("*").remove();
    const x0 = 210, x1 = W-120, top = 34, bh = 28, gap = 16;
    const x = d3.scaleLinear().domain([0, d3.max(rows, r => r.v)]).range([x0, x1]);
    svg.append("text").attr("x", x0).attr("y", 18).attr("fill", P.muted).attr("font-size", 11)
      .text(`bytes per layer per GPU, 16-bit: s = ${DT.big(s)}, b = ${b}, h = ${DT.big(h)}, a = ${a} heads`);
    rows.forEach((r, i) => {
      const y = top + i*(bh+gap);
      svg.append("text").attr("x", x0-8).attr("y", y+bh/2+4).attr("text-anchor","end").attr("fill", P.ink).attr("font-size", 12).text(r.n);
      svg.append("rect").attr("x", x0).attr("y", y).attr("width", Math.max(1, x(r.v)-x0)).attr("height", bh).attr("fill", [P.bad,P.B,P.A,P.good,P.grey][i]).attr("fill-opacity", .8);
      svg.append("text").attr("x", x(r.v)+6).attr("y", y+bh/2+4).attr("fill", P.muted).attr("font-size", 11)
        .text(`${DT.gb(r.v)} GB · ×${L} = ${DT.gb(r.v*L)} GB`);
    });
    const share = att/(34+att);
    d3.select("#ac-read").html(
      `Without parallelism the ${DT.big(s)}-token attention scores (the 5·a·s/h = ${att.toFixed(1)} term) are <b>${DT.pct(share)}</b> of the per-layer ${DT.gb(rows[0].v)} GB. `+
      `Tensor parallelism alone leaves ${DT.gb(sbh*10)} GB per layer unsplit (LayerNorm and dropout inputs); sequence parallelism divides that too, and with selective recompute the layer costs ${DT.gb(rows[3].v)} GB — `+
      `${(rows[0].v/rows[3].v).toFixed(1)}× less than no parallelism, ${(rows[1].v/rows[3].v).toFixed(1)}× less than tensor parallelism alone.`);
  }
  on(["ac-model","ac-s","ac-b","ac-t"], draw);
  draw();
} catch(e){ console.error("viz-act", e); } })();

/* ── Figure 7: ring attention blocks and causal load balance ────────────────────────── */
(function(){ try {
  const {P, num, val, on, $} = DT;
  const svg = d3.select("#cp-svg"), W = 760, H = 360;
  /* sequence cut into 2C chunks; rank r owns chunks own(r) */
  function own(r, C, zig){ return zig ? [r, 2*C-1-r] : [2*r, 2*r+1]; }
  /* work of query chunk qi against key chunk kj, in units of one full chunk×chunk block */
  function work(qi, kj, causal){ if (!causal) return 1; return kj < qi ? 1 : kj === qi ? 0.5 : 0; }
  function draw(){
    const C = num("cp-c"), zig = val("cp-layout")==="zig", causal = val("cp-mask")==="causal";
    const stEl = $("cp-step"); stEl.max = String(C-1);
    const k = Math.min(num("cp-step"), C-1); if (+stEl.value !== k) stEl.value = String(k);
    $("cp-stepv").textContent = `${k} / ${C-1}`;
    /* at step j, rank r holds the K/V chunks of rank (r − j) mod C */
    const W_ = []; // W_[j][r]
    for (let j=0;j<C;j++){ W_.push([]); for (let r=0;r<C;r++){ const src = ((r-j)%C+C)%C; let w=0; own(r,C,zig).forEach(q => own(src,C,zig).forEach(kk => { w += work(q,kk,causal); })); W_[j].push(w); } }
    const total = d3.sum(W_, row => d3.sum(row)), time = d3.sum(W_, row => d3.max(row));
    const eff = total/(C*time);
    svg.selectAll("*").remove();
    const n2 = 2*C, cell = Math.min(20, 300/n2), gx = 60, gy = 40;
    const cols = [P.A, P.B, P.good, P.purple, P.pink, P.teal, "#fbbf24", "#a3e635"];
    const ownerOf = ch => { for (let r=0;r<C;r++) if (own(r,C,zig).includes(ch)) return r; return 0; };
    svg.append("text").attr("x", gx).attr("y", 18).attr("fill", P.muted).attr("font-size", 11).text("query chunk ↓ × key/value chunk →; colour = rank computing it, bright = this step");
    for (let q=0;q<n2;q++){
      svg.append("rect").attr("x", gx-14).attr("y", gy + q*cell+2).attr("width", 8).attr("height", cell-4).attr("fill", cols[ownerOf(q)%8]);
      for (let kk=0;kk<n2;kk++){
        const w = work(q,kk,causal), rq = ownerOf(q), rk = ownerOf(kk);
        const stepOf = ((rq - rk)%C + C)%C;
        const x = gx + kk*cell, y = gy + q*cell;
        if (w === 0){ svg.append("rect").attr("x", x+1).attr("y", y+1).attr("width", cell-2).attr("height", cell-2).attr("fill", "#101218"); continue; }
        const r = svg.append("rect").attr("x", x+1).attr("y", y+1).attr("width", cell-2).attr("height", cell-2)
          .attr("fill", cols[rq%8]).attr("fill-opacity", stepOf===k ? .95 : .2);
        if (w === 0.5) svg.append("path").attr("d", `M${x+1},${y+1} L${x+cell-1},${y+cell-1} L${x+1},${y+cell-1} Z`).attr("fill", "#0f1117").attr("fill-opacity", .55);
        if (stepOf===k) r.attr("stroke", P.ink).attr("stroke-width", .6);
      }
    }
    /* right: per-step, per-rank work bars */
    const bx0 = 420, bx1 = W-20, by0 = 40, by1 = H-60;
    const x = d3.scaleBand().domain(d3.range(C)).range([bx0, bx1]).padding(.2);
    const xi = d3.scaleBand().domain(d3.range(C)).range([0, x.bandwidth()]).padding(.1);
    const y = d3.scaleLinear().domain([0, 4]).range([by1, by0]);
    for (let j=0;j<C;j++) for (let r=0;r<C;r++){
      const v = W_[j][r];
      svg.append("rect").attr("x", x(j)+xi(r)).attr("y", y(v)).attr("width", xi.bandwidth()).attr("height", by1-y(v)).attr("fill", cols[r%8]).attr("fill-opacity", j===k?1:.4);
    }
    d3.range(C).forEach(j => svg.append("text").attr("x", x(j)+x.bandwidth()/2).attr("y", by1+14).attr("text-anchor","middle").attr("fill", j===k?P.B:P.muted).attr("font-size", 10).text("step "+j));
    svg.append("text").attr("x", (bx0+bx1)/2).attr("y", by0-14).attr("text-anchor","middle").attr("fill", P.muted).attr("font-size", 10).text("work per rank per step (blocks); the slowest rank sets the step time");
    const g = svg.append("g").attr("transform", `translate(${bx0},0)`).call(d3.axisLeft(y).ticks(4));
    g.selectAll("text").attr("fill", P.muted); g.selectAll("line,path").attr("stroke", P.line);
    /* overlap check: block c = s/C tokens per rank vs F/B (paper's condition), doubled when the step does half the work */
    const s = Math.pow(2, num("cp-seq")); $("cp-seqv").textContent = DT.big(s);
    const F = 312e12, B = num("cp-bw")*1e9, cmin = F/B;
    const c = s/C, row = W_[k], avg = d3.max(row), need = cmin * 4/Math.max(avg, 1e-9);
    d3.select("#cp-read").html(
      `Step ${k}: rank work ${row.map(v => v.toFixed(1)).join(" / ")} blocks (max ${d3.max(row).toFixed(1)}). Whole pass: ${total.toFixed(1)} blocks of work in ${time.toFixed(1)} block-times on ${C} ranks → <b>${DT.pct(eff)}</b> utilisation`+
      (causal ? (zig ? " — the zig-zag layout gives every rank the same work at every step." : " — contiguous chunks leave early ranks idle while the last rank does the most.") : ".")+
      ` Overlap: each rank holds c = ${DT.big(c)} tokens; at 312 TFLOP/s and ${num("cp-bw")} GB/s the ring condition c ≥ F/B asks for ${DT.big(Math.round(cmin))}`+
      `${avg < 4 ? `, and ≈ ${DT.big(Math.round(need))} when the busiest rank does only ${avg.toFixed(1)} of 4 sub-blocks of work at this step` : ""} — `+
      `${c >= need ? "<b>communication hides behind compute</b>" : "<b>the K/V transfer is exposed</b>"}.`);
  }
  on(["cp-c","cp-layout","cp-mask","cp-step","cp-seq","cp-bw"], draw);
  draw();
} catch(e){ console.error("viz-cp", e); } })();

/* ── Figure 8: pipeline schedules, simulated ─────────────────────────────────────────── */
const DTPipe = (function(){
  /* list-schedule every stage's fixed op order; returns ops with start/end times.
     Times: F = 1 and B = 2 per full stage (a chunk of 1/v stage takes 1/v of that);
     zero-bubble splits B into B (input grad) = 1 and W (weight grad) = 1. */
  function staticOrder(p, m, sched, v){
    const seqs = [];
    for (let d=0; d<p; d++){
      const seq = [];
      if (sched === "gpipe"){ for (let j=0;j<m;j++) seq.push(["F",d,j]); for (let j=0;j<m;j++) seq.push(["B",d,j]); }
      else if (sched === "1f1b"){
        const w = Math.min(p-d-1, m); let f = 0, b = 0;
        for (; f<w; f++) seq.push(["F",d,f]);
        while (f < m){ seq.push(["F",d,f++]); seq.push(["B",d,b++]); }
        while (b < m) seq.push(["B",d,b++]);
      } else {
        const tot = m*v;
        const fw = kk => { const gi = Math.floor(kk/p), c = gi % v, mb = Math.floor(kk/(p*v))*p + kk%p; return [c*p+d, mb]; };
        const bw = kk => { const gi = Math.floor(kk/p), c = v-1-(gi % v), mb = Math.floor(kk/(p*v))*p + kk%p; return [c*p+d, mb]; };
        const w = Math.min((p-d-1)*2 + (v-1)*p, tot); let f = 0, b = 0;
        for (; f<w; f++){ const [vs,mb] = fw(f); seq.push(["F",vs,mb]); }
        while (f < tot){ let q = fw(f++); seq.push(["F",q[0],q[1]]); q = bw(b++); seq.push(["B",q[0],q[1]]); }
        while (b < tot){ const q = bw(b++); seq.push(["B",q[0],q[1]]); }
      }
      seqs.push(seq);
    }
    const V = sched === "inter" ? p*v : p, tF = sched === "inter" ? 1/v : 1, tB = 2*tF;
    const ptr = new Array(p).fill(0), free = new Array(p).fill(0), end = {}, ops = [];
    let left = d3.sum(seqs, s => s.length), prog = true;
    while (left > 0 && prog){
      prog = false;
      for (let d=0; d<p; d++){
        while (ptr[d] < seqs[d].length){
          const [k, vs, mb] = seqs[d][ptr[d]];
          const dep = k === "F" ? (vs === 0 ? 0 : end["F"+(vs-1)+"_"+mb]) : (vs === V-1 ? end["F"+vs+"_"+mb] : end["B"+(vs+1)+"_"+mb]);
          if (dep === undefined) break;
          const s = Math.max(free[d], dep), du = k === "F" ? tF : tB;
          end[k+vs+"_"+mb] = s+du; free[d] = s+du; ops.push({d, k, vs, mb, s, e:s+du, chunk: Math.floor(vs/p)}); ptr[d]++; left--; prog = true;
        }
      }
    }
    return {ops, deadlock: left > 0};
  }
  /* zero-bubble, ZB-H1 style: every stage may hold as much as 1F1B's first stage (p micro-batches);
     F stashes 1 unit, B frees half of it, the deferred W frees the rest. Priority B > F > W. */
  function zeroBubble(p, m){
    const endF = {}, endB = {}, ops = [], st = [];
    for (let d=0; d<p; d++) st.push({nf:0, nb:0, nw:0, free:0, mem:0});
    let t = 0, left = 3*m*p;
    while (left > 0 && t < 100000){
      for (let d=0; d<p; d++){
        const S = st[d]; if (S.free > t) continue;
        let pick = null;
        if (S.nb < S.nf){ const j = S.nb, dep = d === p-1 ? endF[d+"_"+j] : endB[(d+1)+"_"+j]; if (dep !== undefined && dep <= t && endF[d+"_"+j] <= t) pick = ["B", j]; }
        if (!pick && S.nf < m && S.mem + 1 <= p){ const j = S.nf, dep = d === 0 ? 0 : endF[(d-1)+"_"+j]; if (dep !== undefined && dep <= t) pick = ["F", j]; }
        if (!pick && S.nw < S.nb){ const j = S.nw; if (endB[d+"_"+j] <= t) pick = ["W", j]; }
        if (pick){ const [k, j] = pick, e = t+1; if (k === "F"){ endF[d+"_"+j] = e; S.nf++; S.mem += 1; } else if (k === "B"){ endB[d+"_"+j] = e; S.nb++; S.mem -= 0.5; } else { S.nw++; S.mem -= 0.5; } S.free = e; ops.push({d, k, vs:d, mb:j, s:t, e, chunk:0}); left--; }
      }
      t++;
    }
    return {ops, deadlock: left > 0};
  }
  function run(p, m, sched, v){
    const r = sched === "zb" ? zeroBubble(p, m) : staticOrder(p, m, sched, v);
    const T = d3.max(r.ops, o => o.e), busy = d3.sum(r.ops, o => o.e - o.s);
    /* peak activations held on each stage, in units of one micro-batch through one full stage:
       + 1/v at each F, released at the matching B (for zero-bubble, at W — the weight-gradient
       pass still needs the layer inputs, so the stash lives until W; B frees roughly half of it) */
    const peak = [];
    for (let d=0; d<p; d++){
      const ev = [];
      const sz = sched === "inter" ? 1/v : 1;
      r.ops.filter(o => o.d === d).forEach(o => {
        if (o.k === "F") ev.push([o.s, +sz]);
        else if (o.k === "B") ev.push([o.e, sched === "zb" ? -sz/2 : -sz]);
        else ev.push([o.e, -sz/2]);
      });
      ev.sort((a,b) => a[0]-b[0] || a[1]-b[1]);
      let cur = 0, mx = 0; ev.forEach(e => { cur += e[1]; mx = Math.max(mx, cur); }); peak.push(mx);
    }
    return Object.assign(r, {T, busy, bubble: 1 - busy/(p*T), peak});
  }
  return {run};
})();

(function(){ try {
  const {P, num, val, pct, on, $} = DT;
  const svg = d3.select("#pp-svg"), W = 760, H = 330;
  function draw(){
    const sched = val("pp-sched"), p = num("pp-p"), v = num("pp-v");
    let m = num("pp-m"); $("pp-mv").textContent = String(m);
    let note = "";
    if (sched === "inter" && m % p){ const m2 = Math.ceil(m/p)*p; note = ` The interleaved schedule needs m to be a multiple of p, so m was rounded up from ${m} to ${m2}.`; m = m2; }
    const r = DTPipe.run(p, m, sched, v);
    svg.selectAll("*").remove();
    if (r.deadlock){ d3.select("#pp-read").text("schedule deadlocked"); return; }
    const x0 = 70, x1 = W-20, top = 30, rowH = Math.min(40, (H-90)/p);
    const x = d3.scaleLinear().domain([0, r.T]).range([x0, x1]);
    const shade = d3.scaleLinear().domain([0, Math.max(1, m-1)]);
    for (let d=0; d<p; d++){
      svg.append("text").attr("x", x0-8).attr("y", top + d*rowH + rowH/2 + 4).attr("text-anchor","end").attr("fill", P.ink).attr("font-size", 11).text("stage "+d);
      svg.append("rect").attr("x", x0).attr("y", top + d*rowH + 2).attr("width", x1-x0).attr("height", rowH-4).attr("fill", "#101218");
    }
    r.ops.forEach(o => {
      const base = o.k === "F" ? P.A : o.k === "B" ? P.good : P.purple;
      const c = d3.interpolateRgb(base, "#ffffff")(0.45*shade(o.mb));
      const w = Math.max(1, x(o.e)-x(o.s));
      svg.append("rect").attr("x", x(o.s)).attr("y", top + o.d*rowH + 3).attr("width", w - 0.5).attr("height", rowH-6)
        .attr("fill", c).attr("fill-opacity", o.chunk ? .65 : .95).attr("stroke", o.chunk ? P.B : "none").attr("stroke-width", .8);
      if (w > 13 && rowH > 14) svg.append("text").attr("x", x(o.s)+w/2).attr("y", top + o.d*rowH + rowH/2 + 4).attr("text-anchor","middle").attr("fill", "#0f1117").attr("font-size", 9).text(o.mb+1);
    });
    const ax = d3.axisBottom(x).ticks(8);
    const g = svg.append("g").attr("transform", `translate(0,${top + p*rowH + 4})`).call(ax);
    g.selectAll("text").attr("fill", P.muted); g.selectAll("line,path").attr("stroke", P.line);
    svg.append("text").attr("x", x1).attr("y", top + p*rowH + 34).attr("text-anchor","end").attr("fill", P.muted).attr("font-size", 10).text("time, in units of one micro-batch forward through one stage");
    const lg = [["forward", P.A], [sched === "zb" ? "backward (input grad)" : "backward", P.good]].concat(sched === "zb" ? [["weight grad", P.purple]] : []);
    lg.forEach((l, i) => { svg.append("rect").attr("x", x0 + i*170).attr("y", 10).attr("width", 10).attr("height", 10).attr("fill", l[1]); svg.append("text").attr("x", x0 + i*170 + 14).attr("y", 19).attr("fill", P.muted).attr("font-size", 10).text(l[0]); });
    const vv = sched === "inter" ? v : 1;
    const formula = sched === "zb" ? (p-1)/(3*m + p - 1) : (p-1)/(vv*m + p - 1);
    const fText = sched === "zb" ? `(p−1)/(3m+p−1) = ${pct(formula)} (valid for m ≥ p)` : sched === "inter" ? `(p−1)/(v·m+p−1) = ${pct(formula)}` : `(p−1)/(m+p−1) = ${pct(formula)}`;
    const name = {gpipe:"GPipe (all forwards, then all backwards)", "1f1b":"1F1B", inter:`interleaved 1F1B, v = ${v}`, zb:"zero-bubble (ZB-H1-style: B and W split)"}[sched];
    d3.select("#pp-read").html(
      `<b>${name}</b>, p = ${p}, m = ${m}: simulated idle fraction <b>${pct(r.bubble)}</b>; formula ${fText}. `+
      `Stage 0 holds activations for up to <b>${r.peak[0].toFixed(r.peak[0] % 1 ? 1 : 0)}</b> micro-batches' worth of its layers at once (last stage: ${r.peak[p-1].toFixed(r.peak[p-1] % 1 ? 1 : 0)}).`+ note);
  }
  on(["pp-sched","pp-p","pp-m","pp-v"], draw);
  draw();
} catch(e){ console.error("viz-pipe", e); } })();

/* ── Figure 9: expert-parallel all-to-all, with capacity ────────────────────────────── */
(function(){ try {
  const {P, num, val, pct, on, $} = DT;
  const svg = d3.select("#ep-svg"), W = 760, H = 340;
  function rng(seed){ return function(){ seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function draw(){
    const X = num("ep-x"), E0 = num("ep-e"), k = num("ep-k"), cf = num("ep-cf"), skew = num("ep-skew"), T = num("ep-tok"), h = num("ep-h");
    $("ep-cfv").textContent = cf.toFixed(2); $("ep-skewv").textContent = skew.toFixed(1);
    const E = Math.max(X, Math.round(E0/X)*X), perRank = E/X;
    const kk = Math.min(k, E);
    /* router preference: expert e has weight 1/(e+1)^skew, experts shuffled deterministically across ranks */
    const R = rng(7 + 13*E + X);
    const order = d3.range(E); for (let i=E-1;i>0;i--){ const j = Math.floor(R()*(i+1)); [order[i], order[j]] = [order[j], order[i]]; }
    const wts = d3.range(E).map(e => 1/Math.pow(order[e]+1, skew)), tw = d3.sum(wts);
    const cap = Math.ceil(cf * T * X * kk / E);
    const want = new Array(E).fill(0);  // demanded load, kept + dropped
    const load = new Array(E).fill(0), sent = d3.range(X).map(() => new Array(X).fill(0));
    let dropped = 0, routed = 0;
    const R2 = rng(1234 + X*7 + E);
    for (let r=0; r<X; r++) for (let t=0; t<T; t++){
      const pick = new Set();
      while (pick.size < kk){ let u = R2()*tw, e = 0; while (e < E-1 && u > wts[e]){ u -= wts[e]; e++; } pick.add(e); }
      pick.forEach(e => { routed++; want[e]++; if (load[e] >= cap){ dropped++; return; } load[e]++; sent[r][Math.floor(e/perRank)]++; });
    }
    svg.selectAll("*").remove();
    /* left: X×X send matrix */
    const gx = 60, gy = 50, cell = Math.min(34, 240/X);
    const mx = d3.max(sent, row => d3.max(row));
    const col = d3.scaleLinear().domain([0, mx]).range(["#141824", P.A]);
    svg.append("text").attr("x", gx).attr("y", 22).attr("fill", P.muted).attr("font-size", 11).text("tokens sent: rank ↓ → rank holding the expert →");
    for (let i=0;i<X;i++){
      svg.append("text").attr("x", gx-6).attr("y", gy + i*cell + cell/2 + 4).attr("text-anchor","end").attr("fill", P.muted).attr("font-size", 9).text(i);
      svg.append("text").attr("x", gx + i*cell + cell/2).attr("y", gy-5).attr("text-anchor","middle").attr("fill", P.muted).attr("font-size", 9).text(i);
      for (let j=0;j<X;j++){
        svg.append("rect").attr("x", gx + j*cell + 1).attr("y", gy + i*cell + 1).attr("width", cell-2).attr("height", cell-2).attr("fill", col(sent[i][j])).attr("stroke", i===j ? P.B : "none");
        if (cell >= 26) svg.append("text").attr("x", gx + j*cell + cell/2).attr("y", gy + i*cell + cell/2 + 3).attr("text-anchor","middle").attr("fill", P.ink).attr("font-size", 8).text(sent[i][j]);
      }
    }
    svg.append("text").attr("x", gx).attr("y", gy + X*cell + 16).attr("fill", P.B).attr("font-size", 10).text("outlined diagonal stays on the rank (no network)");
    /* right: per-expert load vs capacity */
    const bx0 = 350, bx1 = W-20, by0 = 40, by1 = H-50;
    const x = d3.scaleBand().domain(d3.range(E)).range([bx0, bx1]).padding(.12);
    const need = load;
    const y = d3.scaleLinear().domain([0, Math.max(cap, d3.max(want))*1.1]).range([by1, by0]);
    for (let e=0;e<E;e++){
      const rk = Math.floor(e/perRank);
      svg.append("rect").attr("x", x(e)).attr("y", y(want[e])).attr("width", x.bandwidth()).attr("height", by1-y(want[e])).attr("fill", P.bad).attr("fill-opacity", .35);
      svg.append("rect").attr("x", x(e)).attr("y", y(need[e])).attr("width", x.bandwidth()).attr("height", by1-y(need[e])).attr("fill", rk % 2 ? P.teal : P.A).attr("fill-opacity", .85);
    }
    svg.append("line").attr("x1", bx0).attr("x2", bx1).attr("y1", y(cap)).attr("y2", y(cap)).attr("stroke", P.B).attr("stroke-dasharray", "4 3");
    svg.append("text").attr("x", bx1).attr("y", y(cap)-4).attr("text-anchor","end").attr("fill", P.B).attr("font-size", 10).text(`capacity ${cap} tokens per expert`);
    svg.append("text").attr("x", (bx0+bx1)/2).attr("y", by1+16).attr("text-anchor","middle").attr("fill", P.muted).attr("font-size", 10).text(`${E} experts, ${perRank} per rank (colour alternates by rank); red = dropped over capacity`);
    const g = svg.append("g").attr("transform", `translate(${bx0},0)`).call(d3.axisLeft(y).ticks(5));
    g.selectAll("text").attr("fill", P.muted); g.selectAll("line,path").attr("stroke", P.line);
    /* bytes: each kept assignment leaving its rank carries h values in 16-bit, once to dispatch and once back to combine */
    const offRank = d3.range(X).map(i => d3.sum(sent[i]) - sent[i][i]);
    const worst = d3.max(offRank), mean = d3.mean(offRank);
    const uniform = (X-1)/X * T * kk;
    d3.select("#ep-read").html(
      `${T} tokens per rank × top-${kk} → ${routed} assignments; capacity = ⌈${cf.toFixed(2)} × ${T}·${X}·${kk}/${E}⌉ = ${cap}; <b>${pct(dropped/routed)}</b> dropped. `+
      `Busiest expert ${d3.max(want)} vs mean ${(routed/E).toFixed(0)} requests (${(d3.max(want)/(routed/E)).toFixed(2)}×). `+
      `Dispatch sends up to ${worst} token-vectors off a rank (uniform routing: (X−1)/X·T·k = ${uniform.toFixed(0)}) = <b>${(worst*h*2/1e6).toFixed(1)} MB</b> at h = ${h}; combine returns the same, and backward repeats both — `+
      `${(4*worst*h*2/1e6).toFixed(1)} MB per MoE layer per step on the slowest rank.`);
  }
  on(["ep-x","ep-e","ep-k","ep-cf","ep-skew","ep-tok","ep-h"], draw);
  draw();
} catch(e){ console.error("viz-ep", e); } })();

/* ── Figure 10: N-D parallelism planner ─────────────────────────────────────────────── */
(function(){ try {
  const {P, num, val, gb, secs, pct, on, MODELS} = DT;
  const svg = d3.select("#pl-svg"), W = 760, H = 360;
  function draw(){
    const M = MODELS[val("pl-model")], psi = M.psi, h = M.h, L = M.L;
    const N = num("pl-gpus"), node = num("pl-node"), tp = num("pl-tp"), pp = num("pl-pp"), cp = num("pl-cp");
    const zero = val("pl-zero"), seq = num("pl-seq"), mbs = num("pl-mb"), gbt = num("pl-gbt")*1048576, cap = num("pl-mem")*1e9, ck = val("pl-ckpt");
    const mp = tp*pp*cp, dp = N/mp;
    svg.selectAll("*").remove();
    const problems = [];
    if (!Number.isInteger(dp) || dp < 1) problems.push(`TP·PP·CP = ${mp} does not divide ${N} GPUs`);
    if (pp > L) problems.push(`more stages (${pp}) than layers (${L})`);
    if (tp > M.a) problems.push(`TP ${tp} exceeds ${M.a} heads`);
    if (problems.length){
      svg.append("text").attr("x", W/2).attr("y", H/2).attr("text-anchor","middle").attr("fill", P.bad).attr("font-size", 13).text(problems.join(" · "));
      d3.select("#pl-read").html(`<b>Not a valid layout:</b> ${problems.join("; ")}.`); return;
    }
    /* memory per GPU */
    const psiL = psi/(tp*pp);                                   // parameters this GPU's model-parallel shard owns
    const wB = zero === "z3" ? 2*psiL/dp : 2*psiL, gB = zero === "z3" ? 2*psiL/dp : 2*psiL, oB = 12*psiL/dp;
    const sLoc = seq/cp, layersPerStage = L/pp;
    const perLayer = ck === "full" ? 2*sLoc*mbs*h/tp : 34*sLoc*mbs*h/tp; // SP + selective recompute, or full recompute
    const inflight = pp > 1 ? pp : 1;                          // 1F1B: stage 0 holds p micro-batches
    const aB = perLayer*layersPerStage*inflight;
    const tot = wB + gB + oB + aB;
    /* batch bookkeeping */
    const seqsPerStep = gbt/seq, m = seqsPerStep/(dp*mbs);
    const mOk = m >= 1 && Math.abs(m - Math.round(m)) < 1e-9;
    const bubble = pp > 1 ? (pp-1)/(Math.max(1, m)+pp-1) : 0;
    /* communication estimates */
    const F = 400e12, nvl = 450e9, ib = 50e9;
    const tpBeta = tp <= node ? nvl : ib;
    const tpRatio = tp > 1 ? 2*(tp-1)*F/(9*h*tpBeta) : 0;
    const dpBeta = N <= node ? nvl : ib;
    const dpBytes = (zero === "z3" ? 3 : 2)*(dp-1)/dp*2*psiL;
    const compute = 6*psi*gbt/N/F;
    const dpTime = dpBytes/dpBeta;
    /* draw: memory bar */
    const x0 = 150, x1 = W-40;
    const xm = d3.scaleLinear().domain([0, Math.max(cap*1.15, tot)]).range([x0, x1]);
    const segs = [["weights", wB, P.A], ["grads", gB, P.B], ["optimiser", oB, P.purple], ["activations", aB, P.teal]];
    let cx = x0;
    svg.append("text").attr("x", x0-8).attr("y", 36).attr("text-anchor","end").attr("fill", P.ink).attr("font-size", 12).text("memory per GPU");
    segs.forEach(s => { const w = Math.max(0.5, xm(s[1])-x0); svg.append("rect").attr("x", cx).attr("y", 22).attr("width", w).attr("height", 22).attr("fill", s[2]).attr("fill-opacity", .85); cx += w; });
    svg.append("line").attr("x1", xm(cap)).attr("x2", xm(cap)).attr("y1", 14).attr("y2", 52).attr("stroke", P.bad).attr("stroke-dasharray", "4 3");
    svg.append("text").attr("x", xm(cap)).attr("y", 64).attr("text-anchor","middle").attr("fill", P.bad).attr("font-size", 10).text(`${num("pl-mem")} GB`);
    segs.forEach((s, i) => { svg.append("rect").attr("x", x0 + i*120).attr("y", 72).attr("width", 9).attr("height", 9).attr("fill", s[2]); svg.append("text").attr("x", x0 + i*120 + 13).attr("y", 80).attr("fill", P.muted).attr("font-size", 10).text(`${s[0]} ${gb(s[1])}`); });
    /* draw: device mesh, first few nodes, rank = tp + TP·(cp + CP·(pp + PP·dp)) */
    const shownNodes = Math.min(Math.ceil(N/node), node > 8 ? 2 : 8), perRow = Math.min(node, 36);
    const cellW = Math.min(18, (W-160)/perRow), cellH = 16, my0 = 110;
    const cols = [P.A, P.B, P.good, P.purple, P.pink, P.teal, "#fbbf24", "#a3e635", "#fb7185", "#38bdf8", "#e879f9", "#facc15", "#34d399", "#f97316", "#818cf8", "#94a3b8"];
    svg.append("text").attr("x", 20).attr("y", my0-8).attr("fill", P.muted).attr("font-size", 10).text("device mesh, first nodes (colour = pipeline stage, number = TP rank, outline = first DP replica)");
    let yy = my0;
    for (let nd=0; nd<shownNodes; nd++){
      svg.append("text").attr("x", 20).attr("y", yy + cellH - 4).attr("fill", P.muted).attr("font-size", 10).text("node "+nd);
      for (let gi=0; gi<node; gi++){
        const rank = nd*node + gi; if (rank >= N) break;
        const tpr = rank % tp, rest = Math.floor(rank/tp), cpr = rest % cp, rest2 = Math.floor(rest/cp), ppr = rest2 % pp, dpr = Math.floor(rest2/pp);
        const px = 70 + (gi % perRow)*cellW, py = yy + Math.floor(gi/perRow)*(cellH+2);
        svg.append("rect").attr("x", px).attr("y", py).attr("width", cellW-2).attr("height", cellH).attr("rx", 2).attr("fill", cols[ppr % 16]).attr("fill-opacity", cp > 1 ? .45 + .55*(cpr+1)/cp : .8)
          .attr("stroke", dpr === 0 ? P.ink : "none").attr("stroke-width", .7);
        if (cellW >= 14) svg.append("text").attr("x", px + (cellW-2)/2).attr("y", py + cellH - 4).attr("text-anchor","middle").attr("fill", "#0f1117").attr("font-size", 8).text(tpr);
      }
      yy += Math.ceil(node/perRow)*(cellH+2) + 6;
      if (yy > H - 30) break;
    }
    const fits = tot <= cap;
    d3.select("#pl-read").html(
      `<b>DP = ${N}/(${tp}·${pp}·${cp}) = ${dp}</b>. Per GPU: ${gb(wB)} + ${gb(gB)} + ${gb(oB)} GB model state (${zero === "z3" ? "ZeRO-3 over DP" : "ZeRO-1 over DP"}) + ${gb(aB)} GB activations `+
      `(${ck === "full" ? "full recompute" : "sequence parallel + selective recompute"}, ${inflight} micro-batch${inflight>1?"es":""} in flight) = <b>${gb(tot)} GB</b> — ${fits ? "<span style='color:#4ade80'>fits</span>" : "<span style='color:#f87171'>does not fit</span>"}. `+
      `Batch: ${DT.big(Math.round(seqsPerStep))} sequences → m = ${m >= 1 ? (Math.round(m*100)/100) : "<1"} micro-batches per pipeline${pp>1?`, bubble ≈ ${pct(bubble)}`:""}${m < 1 ? " — <b>too few sequences for this DP × micro-batch</b>" : !mOk ? " (not an integer)" : ""}. `+
      `TP ${tp > 1 ? (tp <= node ? "inside the NVLink domain" : "<b>crosses nodes</b>") + `, comm ≈ ${pct(tpRatio)} of layer compute` : "off"}${tp*cp > node ? "; TP·CP exceeds a node" : ""}. `+
      `DP gradient sync ${secs(dpTime)} (${(dpBytes/1e9).toFixed(1)} GB per GPU, ${dpBeta === nvl ? "NVLink" : "InfiniBand"}) vs ${secs(compute)} of compute per step at 400 TFLOP/s.`);
  }
  on(["pl-model","pl-gpus","pl-node","pl-tp","pl-pp","pl-cp","pl-zero","pl-seq","pl-mb","pl-gbt","pl-mem","pl-ckpt"], draw);
  draw();
} catch(e){ console.error("viz-plan", e); } })();
