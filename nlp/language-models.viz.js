/* language-models.viz.js — extracted from language-models.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const PC={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};

/* shared toy next-token distributions, keyed by last token */
const DISTS={
  "<start>":{"The":.4,"A":.2,"Once":.15,"In":.15,"She":.1},
  "the":{"cat":.25,"model":.2,"data":.15,"world":.15,"system":.15,"answer":.1},
  "The":{"cat":.25,"model":.2,"data":.15,"network":.15,"system":.15,"future":.1},
  "cat":{"sat":.35,"ran":.2,"slept":.15,"jumped":.15,"purred":.15},
  "sat":{"on":.6,"quietly":.2,"down":.2},
  "on":{"the":.7,"a":.2,"its":.1},
  "model":{"learns":.3,"predicts":.25,"generates":.2,"sees":.15,"fails":.1},
  "data":{"is":.4,"flows":.2,"drives":.2,"matters":.2},
  "is":{"the":.3,"a":.25,"very":.2,"all":.15,"not":.1},
  "learns":{"to":.5,"from":.3,"patterns":.2},
  "to":{"predict":.4,"generate":.3,"learn":.3},
};
const GENERIC={".":.35,"and":.2,"the":.2,"a":.15,"that":.1};
function distFor(ctx){ const last=ctx[ctx.length-1]??"<start>"; return DISTS[last]||DISTS[(last||"").toLowerCase()]||GENERIC; }
function applyTemp(d,T){
  const ks=Object.keys(d), logits=ks.map(k=>Math.log(d[k])), sc=logits.map(l=>l/T), mx=Math.max(...sc);
  const ex=sc.map(l=>Math.exp(l-mx)), s=ex.reduce((a,b)=>a+b,0), out={};
  ks.forEach((k,i)=>out[k]=ex[i]/s); return out;
}

/* ───────────────────────── 01 · next-token distribution + sampling ───────────────────────── */
(function(){
  let ctx=[]; const svg=d3.select("#lm-svg"),W=640,H=220,m={t:10,r:40,b:24,l:120};
  function render(){
    const T=+d3.select("#lm-temp").property("value");
    const d=applyTemp(distFor(ctx),T);
    const arr=Object.entries(d).map(([t,p])=>({t,p})).sort((a,b)=>b.p-a.p);
    const y=d3.scaleBand().domain(arr.map(a=>a.t)).range([m.t,H-m.b]).padding(.25);
    const x=d3.scaleLinear().domain([0,Math.max(.1,d3.max(arr,a=>a.p))]).range([m.l,W-m.r]);
    const sel=svg.selectAll("g.bar").data(arr,a=>a.t);
    const ent=sel.enter().append("g").attr("class","bar");
    ent.append("rect").attr("x",m.l).attr("height",y.bandwidth()).attr("rx",4);
    ent.append("text").attr("class","lbl").attr("x",m.l-10).attr("text-anchor","end").attr("font-size",12).attr("fill",PC.ink);
    ent.append("text").attr("class","val").attr("font-size",11).attr("fill",PC.muted);
    const all=ent.merge(sel);
    all.select("rect").transition().duration(250).attr("y",a=>y(a.t)).attr("width",a=>x(a.p)-m.l).attr("fill",(a,i)=>i===0?PC.accent:"#3b4252").attr("height",y.bandwidth());
    all.select("text.lbl").transition().duration(250).attr("y",a=>y(a.t)+y.bandwidth()/2+4).text(a=>a.t);
    all.select("text.val").transition().duration(250).attr("x",a=>x(a.p)+6).attr("y",a=>y(a.t)+y.bandwidth()/2+4).text(a=>(a.p*100).toFixed(0)+"%");
    sel.exit().remove();
    d3.select("#lm-gen").text(ctx.length?ctx.join(" "):"(empty — sample to begin)");
  }
  function sample(){
    const T=+d3.select("#lm-temp").property("value");
    const d=applyTemp(distFor(ctx),T);
    const r=Math.random(); let acc=0,chosen=Object.keys(d)[0];
    for(const [k,p] of Object.entries(d)){ acc+=p; if(r<=acc){chosen=k;break;} }
    ctx.push(chosen); render(); return chosen;
  }
  let auto=null;
  d3.select("#lm-temp").on("input",function(){ d3.select("#lm-tval").text((+this.value).toFixed(1)); render(); });
  d3.select("#lm-sample").on("click",sample);
  d3.select("#lm-auto").on("click",function(){
    if(auto){clearInterval(auto);auto=null;this.textContent="Auto-generate";return;}
    this.textContent="⏸ Stop";
    auto=setInterval(()=>{ const c=sample(); if(c==="."||ctx.length>16){clearInterval(auto);auto=null;d3.select("#lm-auto").text("Auto-generate");} },600);
  });
  d3.select("#lm-reset").on("click",()=>{ if(auto){clearInterval(auto);auto=null;d3.select("#lm-auto").text("Auto-generate");} ctx=[]; render(); });
  render();
})();

/* ───────────────────────── 04 · perplexity meter ───────────────────────── */
(function(){
  const svg=d3.select("#ppl-svg"),W=640,H=170,m={t:14,r:16,b:20,l:16};
  const choicesEl=d3.select("#ppl-choices"), out=document.getElementById("ppl-readout"), ctxEl=document.getElementById("ppl-ctx");
  let ctx=[], logps=[];   // logps = accumulated log P of chosen tokens
  function choices(){
    const d=distFor(ctx);
    return Object.entries(d).map(([t,p])=>({t,p})).sort((a,b)=>b.p-a.p);
  }
  function renderChoices(){
    const arr=choices();
    const sel=choicesEl.selectAll("button.btn").data(arr,d=>d.t);
    sel.exit().remove();
    sel.enter().append("button").attr("class","btn ghost").merge(sel)
      .text(d=>`${d.t} · ${(d.p*100).toFixed(0)}%`)
      .on("click",(e,d)=>pick(d));
  }
  function pick(d){
    logps.push(Math.log(d.p));
    ctx.push(d.t);
    redraw();
  }
  function redraw(){
    ctxEl.innerHTML = ctx.length ? "sequence: <b>"+ctx.join(" ")+"</b>" : "sequence: <i>(pick a first token)</i>";
    renderChoices();
    // bars: per-token probability of each chosen token
    const data=ctx.map((t,i)=>({t,p:Math.exp(logps[i])}));
    const x=d3.scaleBand().domain(d3.range(Math.max(1,data.length))).range([m.l,W-m.r]).padding(.3);
    const y=d3.scaleLinear().domain([0,1]).range([H-m.b,m.t]);
    svg.selectAll("*").remove();
    // baseline
    svg.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",H-m.b).attr("y2",H-m.b).attr("stroke",PC.line);
    data.forEach((d,i)=>{
      const col = d.p>0.4?PC.good : d.p>0.18?PC.a2 : PC.bad;
      svg.append("rect").attr("x",x(i)).attr("width",x.bandwidth()).attr("y",y(d.p)).attr("height",H-m.b-y(d.p)).attr("rx",3).attr("fill",col).attr("fill-opacity",.85);
      svg.append("text").attr("x",x(i)+x.bandwidth()/2).attr("y",y(d.p)-5).attr("text-anchor","middle").attr("font-size",10).attr("fill",PC.muted).text((d.p*100).toFixed(0)+"%");
      svg.append("text").attr("x",x(i)+x.bandwidth()/2).attr("y",H-m.b+14).attr("text-anchor","middle").attr("font-size",10).attr("fill",PC.ink).text(d.t.length>7?d.t.slice(0,6)+"…":d.t);
    });
    if(!data.length){ svg.append("text").attr("x",W/2).attr("y",H/2).attr("text-anchor","middle").attr("font-size",12).attr("fill",PC.muted).text("pick tokens above to score them"); }
    // running perplexity = exp( -mean logp )
    if(logps.length){
      const meanNLL = -logps.reduce((a,b)=>a+b,0)/logps.length;
      const ppl=Math.exp(meanNLL);
      const c = ppl<2?PC.good : ppl<4?PC.a2 : PC.bad;
      out.innerHTML=`avg NLL <b>${meanNLL.toFixed(3)}</b> nats &nbsp;·&nbsp; running perplexity <b style="color:${c}">${ppl.toFixed(2)}</b>`;
    } else out.innerHTML="";
  }
  document.getElementById("ppl-reset").onclick=()=>{ ctx=[]; logps=[]; redraw(); };
  redraw();
})();

/* ───────────────────────── 07 · decoding-strategy explorer ───────────────────────── */
(function(){
  const svg=d3.select("#dec-svg"),W=640,H=240,m={t:14,r:16,b:24,l:16};
  const out=document.getElementById("dec-readout");
  // a fixed peaky distribution to make truncation visible
  const base=[["model",.30],["cat",.22],["data",.15],["world",.11],["system",.08],["future",.06],["answer",.05],["machine",.03]];
  let mode="greedy";
  const btns={greedy:"dec-greedy",temp:"dec-temp",topk:"dec-topk",topp:"dec-topp"};
  function setMode(mm){
    mode=mm;
    Object.entries(btns).forEach(([k,id])=>d3.select("#"+id).attr("class","btn"+(k===mm?"":" ghost")));
    d3.select("#dec-knob-temp").style("display",mm==="temp"?"":"none");
    d3.select("#dec-knob-k").style("display",mm==="topk"?"":"none");
    d3.select("#dec-knob-p").style("display",mm==="topp"?"":"none");
    render();
  }
  function probs(){
    // apply temperature to the base logits (log of base prob), default T=1
    const T = mode==="temp" ? +d3.select("#dec-T").property("value") : 1;
    const ks=base.map(b=>b[0]), logits=base.map(b=>Math.log(b[1])), sc=logits.map(l=>l/T), mx=Math.max(...sc);
    const ex=sc.map(l=>Math.exp(l-mx)), s=ex.reduce((a,b)=>a+b,0);
    return ks.map((t,i)=>({t,p:ex[i]/s}));
  }
  function keptSet(arr){
    const sorted=[...arr].sort((a,b)=>b.p-a.p);
    if(mode==="greedy") return new Set([sorted[0].t]);
    if(mode==="topk"){ const k=+d3.select("#dec-k").property("value"); return new Set(sorted.slice(0,k).map(d=>d.t)); }
    if(mode==="topp"){ const p=+d3.select("#dec-p").property("value"); const keep=new Set(); let cum=0;
      for(const d of sorted){ keep.add(d.t); cum+=d.p; if(cum>=p) break; } return keep; }
    return new Set(sorted.map(d=>d.t)); // temp keeps all, just reshapes
  }
  function render(){
    const arr=probs().sort((a,b)=>b.p-a.p), kept=keptSet(arr);
    const x=d3.scaleBand().domain(arr.map(a=>a.t)).range([m.l,W-m.r]).padding(.28);
    const y=d3.scaleLinear().domain([0,Math.max(.1,d3.max(arr,a=>a.p))]).range([H-m.b,m.t]);
    svg.selectAll("*").remove();
    svg.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",H-m.b).attr("y2",H-m.b).attr("stroke",PC.line);
    arr.forEach(d=>{
      const on=kept.has(d.t);
      svg.append("rect").attr("x",x(d.t)).attr("width",x.bandwidth()).attr("y",y(d.p)).attr("height",H-m.b-y(d.p)).attr("rx",3)
        .attr("fill",on?PC.accent:"#2a2f3a").attr("fill-opacity",on?.9:.4).attr("stroke",on?PC.accent:PC.line).attr("stroke-opacity",on?.7:.4);
      svg.append("text").attr("x",x(d.t)+x.bandwidth()/2).attr("y",y(d.p)-5).attr("text-anchor","middle").attr("font-size",10).attr("fill",on?PC.ink:PC.muted).text((d.p*100).toFixed(0)+"%");
      svg.append("text").attr("x",x(d.t)+x.bandwidth()/2).attr("y",H-m.b+15).attr("text-anchor","middle").attr("font-size",10).attr("fill",on?PC.ink:PC.muted).text(d.t.length>7?d.t.slice(0,6)+"…":d.t);
    });
    const nkept=[...kept].length, pmass=arr.filter(a=>kept.has(a.t)).reduce((s,a)=>s+a.p,0);
    const labels={greedy:"Greedy — take the single argmax",temp:"Temperature — reshape, keep all",topk:"Top-k — keep k most probable",topp:"Top-p — keep smallest set with ∑P ≥ p"};
    out.innerHTML=`${labels[mode]} &nbsp;·&nbsp; candidate set: <b style="color:${PC.accent}">${nkept}</b> token${nkept===1?"":"s"} &nbsp;·&nbsp; covering <b>${(pmass*100).toFixed(0)}%</b> mass`;
  }
  d3.select("#dec-T").on("input",function(){ d3.select("#dec-Tv").text((+this.value).toFixed(1)); render(); });
  d3.select("#dec-k").on("input",function(){ d3.select("#dec-kv").text(this.value); render(); });
  d3.select("#dec-p").on("input",function(){ d3.select("#dec-pv").text((+this.value).toFixed(2)); render(); });
  d3.select("#dec-greedy").on("click",()=>setMode("greedy"));
  d3.select("#dec-temp").on("click",()=>setMode("temp"));
  d3.select("#dec-topk").on("click",()=>setMode("topk"));
  d3.select("#dec-topp").on("click",()=>setMode("topp"));
  setMode("greedy");
})();
