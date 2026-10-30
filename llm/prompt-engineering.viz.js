/* prompt-engineering.viz.js — figures for prompt-engineering.html.
   Loaded after data.js / notes.js. One IIFE per figure; each returns early if its
   elements are missing. All three figures are illustrative models, not measurements. */

const PE_C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#11141b",blue:"#5b9cff",violet:"#c084fc"};

/* ---------- (a) few-shot sensitivity explorer ---------- */
(function(){
  const svgEl=document.getElementById("fs-svg");
  const ids=["fs-k","fs-sel","fs-bal","fs-ord","fs-cal"];
  if(!svgEl||typeof d3==="undefined"||ids.some(i=>!document.getElementById(i))) return;
  const C=PE_C, svg=d3.select(svgEl), W=640, H=280;
  const m={l:48,r:18,t:22,b:40};
  const KMAX=16, NDRAW=25;

  // seeded RNG so a given setting always shows the same draws
  function rng(seed){let a=seed>>>0;return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
  function gauss(r){let u=0,v=0;while(u===0)u=r();while(v===0)v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}

  // toy model: mean and spread of accuracy on a balanced binary task (chance = 0.5)
  function model(k,o){
    const G=o.sel==="similar"?0.27:0.20;
    let mean=0.62+G*(1-Math.exp(-k/2.5));
    if(k>0){
      const imb=Math.abs(o.bal-0.5)*2;                    // 0 balanced … 1 all one label
      mean-=0.14*imb*(o.cal?0.35:1);                      // majority-label bias
      if(o.ord==="block") mean-=0.05*(1-Math.exp(-k/2))*(o.cal?0.4:1);  // recency bias
    }
    let sd;
    if(k===0) sd=0.015;                                   // only wording noise
    else{
      const ordF=o.ord==="block"?1.3:(o.ord==="interleaved"?0.8:1);
      sd=(0.10*Math.exp(-k/5)+0.02)*(o.sel==="similar"?0.8:1)*(o.cal?0.6:1)*ordF;
    }
    return {mean:Math.max(0.45,Math.min(0.97,mean)),sd};
  }
  const clip=v=>Math.max(0.40,Math.min(0.99,v));

  function opts(){
    return {k:+d3.select("#fs-k").property("value"),
            sel:d3.select("#fs-sel").property("value"),
            bal:+d3.select("#fs-bal").property("value")/100,
            ord:d3.select("#fs-ord").property("value"),
            cal:d3.select("#fs-cal").property("checked")};
  }

  function draw(){
    svg.selectAll("*").remove();
    const o=opts();
    const x=d3.scaleLinear().domain([0,KMAX]).range([m.l,W-m.r]);
    const y=d3.scaleLinear().domain([0.40,1.0]).range([H-m.b,m.t]);
    // axes
    svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).ticks(8).tickFormat(d3.format("d")))
      .call(g=>{g.selectAll("text").attr("fill",C.muted);g.selectAll("line,path").attr("stroke",C.line);});
    svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(d3.format(".0%")))
      .call(g=>{g.selectAll("text").attr("fill",C.muted);g.selectAll("line,path").attr("stroke",C.line);});
    svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-6).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("number of in-context examples k");
    svg.append("text").attr("transform",`translate(12,${(m.t+H-m.b)/2}) rotate(-90)`).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("simulated accuracy");
    // chance line
    svg.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(0.5)).attr("y2",y(0.5)).attr("stroke",C.bad).attr("stroke-dasharray","3 4").attr("opacity",.6);
    svg.append("text").attr("x",m.l+4).attr("y",y(0.5)+12).attr("text-anchor","start").attr("fill",C.bad).attr("font-size",9.5).text("chance");

    const ks=d3.range(0,KMAX+1);
    const ref={sel:"random",bal:0.5,ord:"shuffled",cal:false};
    const cur=ks.map(k=>({k,...model(k,o)}));
    const base=ks.map(k=>({k,...model(k,ref)}));
    // band for current settings
    svg.append("path").datum(cur).attr("fill",C.accent).attr("fill-opacity",.14)
      .attr("d",d3.area().x(d=>x(d.k)).y0(d=>y(clip(d.mean-d.sd))).y1(d=>y(clip(d.mean+d.sd))).curve(d3.curveMonotoneX));
    // reference mean (balanced, shuffled, random pool, uncalibrated)
    svg.append("path").datum(base).attr("fill","none").attr("stroke",C.muted).attr("stroke-dasharray","5 4").attr("stroke-width",1.2)
      .attr("d",d3.line().x(d=>x(d.k)).y(d=>y(d.mean)).curve(d3.curveMonotoneX));
    // current mean
    svg.append("path").datum(cur).attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2)
      .attr("d",d3.line().x(d=>x(d.k)).y(d=>y(d.mean)).curve(d3.curveMonotoneX));
    // draws at the chosen k
    const st=model(o.k,o);
    const seed=o.k*7919+(o.sel==="similar"?13:0)+Math.round(o.bal*100)*31+({shuffled:1,interleaved:2,block:3}[o.ord])*101+(o.cal?577:0);
    const r=rng(seed+1);
    const draws=d3.range(NDRAW).map(()=>clip(st.mean+st.sd*gauss(r)));
    svg.append("line").attr("x1",x(o.k)).attr("x2",x(o.k)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",C.a2).attr("opacity",.35);
    svg.selectAll("circle.draw").data(draws).enter().append("circle").attr("class","draw")
      .attr("cx",(d,i)=>x(o.k)+((i%5)-2)*3.2).attr("cy",d=>y(d)).attr("r",3)
      .attr("fill",C.a2).attr("fill-opacity",.75).attr("stroke","#0f1117").attr("stroke-width",.5);
    // legend
    const lg=svg.append("g").attr("transform",`translate(${W-m.r-240},${y(0.585)})`);
    [[C.accent,"mean ± 1 sd (current settings)",false],[C.muted,"reference: random pool, balanced, shuffled",true],[C.a2,"25 random draws at chosen k",false]].forEach((e,i)=>{
      const yy=i*14;
      if(i===2) lg.append("circle").attr("cx",9).attr("cy",yy).attr("r",3).attr("fill",e[0]);
      else lg.append("line").attr("x1",0).attr("x2",18).attr("y1",yy).attr("y2",yy).attr("stroke",e[0]).attr("stroke-width",2).attr("stroke-dasharray",e[2]?"5 4":null);
      lg.append("text").attr("x",24).attr("y",yy+3.5).attr("fill",C.muted).attr("font-size",10).text(e[1]);
    });
    svg.append("text").attr("x",W-m.r).attr("y",m.t-6).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",9.5).attr("font-style","italic").text("illustrative toy model — not measured data");

    const lo=d3.min(draws), hi=d3.max(draws);
    const notes=[];
    if(o.k===0) notes.push("zero-shot: no examples, so selection, balance and order do nothing");
    else{
      if(Math.abs(o.bal-0.5)>=0.3) notes.push("label imbalance pulls predictions toward the majority label");
      if(o.ord==="block") notes.push("same-label block last → recency bias");
      if(o.k<=4) notes.push("small k: the spread across draws is wide");
      if(o.cal) notes.push("calibration shrinks bias and spread");
      if(o.k===1) notes.push("with one example the label share is necessarily 0% or 100% — the slider is a pool-level idealisation");
    }
    d3.select("#fs-read").html(`k = <b>${o.k}</b> · mean <b>${(st.mean*100).toFixed(1)}%</b> · sd ${(st.sd*100).toFixed(1)} pts · draws ${(lo*100).toFixed(0)}–${(hi*100).toFixed(0)}%${notes.length?" · "+notes.join(" · "):""}`);
  }
  ["#fs-k","#fs-bal"].forEach(s=>d3.select(s).on("input",draw));
  ["#fs-sel","#fs-ord","#fs-cal"].forEach(s=>d3.select(s).on("change",draw));
  draw();
})();

/* ---------- (b) prompt cost calculator ---------- */
(function(){
  const svgEl=document.getElementById("pc-svg");
  const ids=["pc-static","pc-dyn","pc-out","pc-calls","pc-comp","pc-cache"];
  if(!svgEl||typeof d3==="undefined"||ids.some(i=>!document.getElementById(i))) return;
  const C=PE_C, svg=d3.select(svgEl), W=640, H=250;
  // placeholder prices (illustrative): $ per token
  const CIN=3/1e6, COUT=5*CIN, RHO_R=0.1, RHO_W=1.25, WRITES=288;
  const money=v=>v>=100?"$"+d3.format(",.0f")(v):(v>=1?"$"+d3.format(",.2f")(v):"$"+d3.format(".3f")(v));

  function costs(ns,nd,no,N,keep,cache){
    const s=ns*keep;
    const parts={
      prefix: cache? N*s*RHO_R*CIN : N*s*CIN,
      input:  N*nd*CIN,
      output: N*no*COUT,
      writes: cache&&s>0? Math.min(WRITES,N)*s*(RHO_W-RHO_R)*CIN : 0
    };
    parts.total=parts.prefix+parts.input+parts.output+parts.writes;
    return parts;
  }

  function draw(){
    svg.selectAll("*").remove();
    const v=id=>+d3.select("#"+id).property("value");
    const ns=v("pc-static"), nd=v("pc-dyn"), no=v("pc-out"), N=v("pc-calls"), keep=v("pc-comp");
    const cache=d3.select("#pc-cache").property("checked");
    const base=costs(ns,nd,no,N,1,false);
    const opt=costs(ns,nd,no,N,keep,cache);
    const rows=[{name:"Baseline",sub:"full prompt, no cache",c:base},{name:"Optimised",sub:`keep ${Math.round(keep*100)}% of prefix${cache?", cached":""}`,c:opt}];
    const segs=[["prefix","static prefix",C.blue],["input","per-call input",C.accent],["output","output",C.a2],["writes","cache writes",C.violet]];
    const L=118, R=96, x=d3.scaleLinear().domain([0,Math.max(base.total,opt.total)||1]).range([L,W-R]);
    rows.forEach((row,i)=>{
      const y0=36+i*74, bh=38;
      svg.append("text").attr("x",8).attr("y",y0+16).attr("fill",C.ink).attr("font-size",12).attr("font-weight",700).text(row.name);
      svg.append("text").attr("x",8).attr("y",y0+31).attr("fill",C.muted).attr("font-size",9.5).text(row.sub);
      svg.append("rect").attr("x",L).attr("y",y0).attr("width",W-R-L).attr("height",bh).attr("rx",5).attr("fill",C.panel).attr("stroke",C.line);
      let acc=0;
      segs.forEach(([k,label,col])=>{
        const val=row.c[k]; if(val<=0) return;
        const xa=x(acc), xb=x(acc+val); acc+=val;
        svg.append("rect").attr("x",xa).attr("y",y0).attr("width",Math.max(0,xb-xa)).attr("height",bh).attr("fill",col).attr("fill-opacity",.85).attr("stroke","#0f1117");
        if(xb-xa>52) svg.append("text").attr("x",(xa+xb)/2).attr("y",y0+bh/2+4).attr("text-anchor","middle").attr("font-size",10).attr("font-weight",700).attr("fill","#06101f").text(money(val));
      });
      svg.append("text").attr("x",x(row.c.total)+6).attr("y",y0+bh/2+4).attr("fill",C.ink).attr("font-size",11.5).attr("font-weight",700).text(money(row.c.total)+"/day");
    });
    // legend
    let lx=L;
    segs.forEach(([k,label,col])=>{
      svg.append("rect").attr("x",lx).attr("y",H-34).attr("width",10).attr("height",10).attr("rx",2).attr("fill",col);
      svg.append("text").attr("x",lx+15).attr("y",H-25).attr("fill",C.muted).attr("font-size",10).text(label);
      lx+=label.length*6+34;
    });
    svg.append("text").attr("x",8).attr("y",H-8).attr("fill",C.muted).attr("font-size",9.5).attr("font-style","italic")
      .text("placeholder prices: input $3/M tok · output 5× · cached read 0.1× · cache write 1.25× · 288 writes/day");
    const save=base.total>0?1-opt.total/base.total:0;
    const outShare=base.total>0?base.output/base.total:0;
    const warn=keep<0.5?' · <span style="color:#f87171">aggressive compression: expect accuracy loss — re-run the eval</span>':"";
    d3.select("#pc-read").html(`baseline <b>${money(base.total)}</b>/day (≈ ${money(base.total*30)}/month) · optimised <b>${money(opt.total)}</b>/day · saving <b>${(save*100).toFixed(0)}%</b> · output is ${(outShare*100).toFixed(0)}% of baseline cost, ${(opt.total>0?opt.output/opt.total*100:0).toFixed(0)}% of optimised${warn}`);
  }
  ["#pc-static","#pc-dyn","#pc-out","#pc-calls","#pc-comp"].forEach(s=>d3.select(s).on("input",draw));
  d3.select("#pc-cache").on("change",draw);
  draw();
})();

/* ---------- (c) prompting / RAG / fine-tuning decision triangle ---------- */
(function(){
  const svgEl=document.getElementById("tri-svg");
  const ids=["tri-know","tri-beh","tri-vol"];
  if(!svgEl||typeof d3==="undefined"||ids.some(i=>!document.getElementById(i))) return;
  const C=PE_C, svg=d3.select(svgEl), W=640, H=320;
  const V=[
    {key:"P",name:"Prompting",fix:"task statement · format · tone",x:W/2,y:72,col:C.accent},
    {key:"R",name:"RAG",fix:"missing / fresh knowledge",x:120,y:240,col:C.blue},
    {key:"F",name:"Fine-tuning",fix:"persistent behaviour · cost at volume",x:W-120,y:240,col:C.a2}
  ];
  const smooth=(a,b,t)=>{const u=Math.max(0,Math.min(1,(t-a)/(b-a)));return u*u*(3-2*u);};
  // heuristic weights: prompting is the baseline and absorbs mild behaviour gaps;
  // RAG tracks the knowledge gap; fine-tuning tracks large behaviour gaps and volume pressure
  function weights(k,b,v){
    const s=smooth(0.35,0.9,b);
    const w={P:0.4+0.6*b*(1-s), R:1.1*k, F:0.9*s+0.45*v};
    const t=w.P+w.R+w.F; return {P:w.P/t,R:w.R/t,F:w.F/t};
  }
  const trail=[];

  function draw(){
    svg.selectAll("*").remove();
    const k=+d3.select("#tri-know").property("value")/100;
    const b=+d3.select("#tri-beh").property("value")/100;
    const v=+d3.select("#tri-vol").property("value")/100;
    const w=weights(k,b,v);
    const px=V.reduce((s,q)=>s+w[q.key]*q.x,0), py=V.reduce((s,q)=>s+w[q.key]*q.y,0);
    trail.push([px,py]); if(trail.length>14) trail.shift();
    const ranked=V.slice().sort((a,c)=>w[c.key]-w[a.key]);
    const top=ranked[0], second=ranked[1];
    const combo=w[second.key]>=0.75*w[top.key];
    // triangle
    svg.append("path").attr("d",`M${V[0].x},${V[0].y}L${V[1].x},${V[1].y}L${V[2].x},${V[2].y}Z`).attr("fill",C.panel).attr("stroke",C.line).attr("stroke-width",1.5);
    // centroid-to-edge guides
    const cx=d3.mean(V,q=>q.x), cy=d3.mean(V,q=>q.y);
    [[0,1],[1,2],[2,0]].forEach(([i,j])=>{
      svg.append("line").attr("x1",cx).attr("y1",cy).attr("x2",(V[i].x+V[j].x)/2).attr("y2",(V[i].y+V[j].y)/2).attr("stroke",C.line).attr("stroke-dasharray","2 4");
    });
    // vertices
    V.forEach(q=>{
      const on=q===top||(combo&&q===second);
      svg.append("circle").attr("cx",q.x).attr("cy",q.y).attr("r",10+26*w[q.key]).attr("fill",q.col).attr("fill-opacity",on?.32:.10).attr("stroke",q.col).attr("stroke-width",on?2:1);
      const below=q.key!=="P";
      svg.append("text").attr("x",q.x).attr("y",below?q.y+46:q.y-44).attr("text-anchor","middle").attr("fill",on?q.col:C.ink).attr("font-size",13).attr("font-weight",700).text(`${q.name} ${(w[q.key]*100).toFixed(0)}%`);
      svg.append("text").attr("x",q.x+(q.key==="P"?150:0)).attr("y",below?q.y+60:q.y+4).attr("text-anchor",q.key==="P"?"start":"middle").attr("fill",C.muted).attr("font-size",10).text("fixes: "+q.fix);
    });
    // trail + point
    if(trail.length>1) svg.append("path").attr("d",d3.line()(trail)).attr("fill","none").attr("stroke",C.ink).attr("stroke-opacity",.25).attr("stroke-dasharray","2 3");
    svg.append("circle").attr("cx",px).attr("cy",py).attr("r",7).attr("fill",C.ink).attr("stroke","#0f1117").attr("stroke-width",2);
    svg.append("text").attr("x",8).attr("y",H-4).attr("fill",C.muted).attr("font-size",9.5).attr("font-style","italic").text("illustrative heuristic, not a calibrated model");

    let rec;
    if(combo){
      const pair=[top.key,second.key].sort().join("");
      rec={FP:"prompt well, then fine-tune once the behaviour is pinned down by evals",
           PR:"prompting + RAG: ground the answer in retrieved facts with a clear instruction",
           FR:"RAG for the facts plus a fine-tuned model for format and register"}[pair];
    } else {
      rec={P:"stay with prompting — iterate on instructions and examples against an eval set",
           R:"add retrieval — the failures trace to facts the model doesn't have",
           F:"fine-tune (e.g. a parameter-efficient adapter) — the behaviour must be persistent, or the prompt is too long to pay for at this volume"}[top.key];
    }
    d3.select("#tri-read").html(`recommendation: <b>${combo?top.name+" + "+second.name:top.name}</b> — ${rec}`);
  }
  ids.forEach(i=>d3.select("#"+i).on("input",draw));
  draw();
})();
