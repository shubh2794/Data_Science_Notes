/* vector-databases.viz.js — extracted from vector-databases.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ---- Viz 1: HNSW layered greedy search ---- */
(function(){
  const C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",q:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",edge:"#2a2f3a"};
  const svg=d3.select("#hn-svg"),W=640,H=360;
  // base positions in a 0-100 space (deterministic)
  const P=[
    [12,30],[26,62],[20,16],[38,40],[33,78],[48,22],[52,58],[60,84],
    [66,34],[74,66],[80,20],[86,50],[92,80],[58,12]
  ].map((p,i)=>({id:i,x:p[0],y:p[1]}));
  // layer membership (subset chain): L0=all, L1 subset, L2 subset of L1
  const LYR=[ P.map(p=>p.id), [0,3,6,9,11,2], [0,9] ];   // index 0 = base
  const ENTRY=0;                                         // top-layer entry node
  const M=[3,2,2];                                       // links per node per layer (base..top)
  function dist(a,b){const dx=a.x-b.x,dy=a.y-b.y;return Math.hypot(dx,dy);}
  // build kNN edges per layer
  function edgesFor(ids,m){
    const ns=ids.map(i=>P[i]); const E=new Map(ids.map(i=>[i,[]]));
    ns.forEach(a=>{
      const near=ns.filter(b=>b.id!==a.id).sort((u,v)=>dist(a,u)-dist(a,v)).slice(0,m);
      near.forEach(b=>{ if(!E.get(a.id).includes(b.id)) E.get(a.id).push(b.id);
                        if(!E.get(b.id).includes(a.id)) E.get(b.id).push(a.id); });
    });
    return E;
  }
  const EDGE=LYR.map((ids,l)=>edgesFor(ids,M[l]));
  // greedy search within a layer from `start` toward q; returns path of node ids
  function greedy(l,start,q){
    let cur=start, path=[cur], improved=true, guard=0;
    while(improved && guard++<50){ improved=false; let best=cur,bd=dist(P[cur],q);
      (EDGE[l].get(cur)||[]).forEach(nb=>{ const d=dist(P[nb],q); if(d<bd-1e-9){bd=d;best=nb;improved=true;} });
      if(improved){cur=best;path.push(cur);} }
    return path;
  }
  // full multi-layer plan → ordered steps [{layer, node, drop?}]
  function plan(q){
    const steps=[]; let entry=ENTRY;
    for(let l=LYR.length-1;l>=0;l--){
      const pth=greedy(l,entry,q);
      pth.forEach((n,i)=>steps.push({layer:l,node:n,first:i===0}));
      entry=pth[pth.length-1];           // descend at the local-best node
      if(l>0) steps.push({layer:l,node:entry,drop:true});
    }
    return {steps, answer:entry};
  }
  // geometry: three stacked panels (top layer drawn highest)
  const px=70,pw=520, ph=88, gap=14, top=14;
  function panelY(l){ return top + (LYR.length-1-l)*(ph+gap); }   // l=2 at top
  const X=x=>px + x/100*pw;
  const Y=(l,y)=>panelY(l) + 10 + y/100*(ph-20);
  let cur, PLAN, qx=76;
  function q(){ return {x:qx, y:46}; }
  function rebuild(){ PLAN=plan(q()); cur=0; draw(); }
  function draw(){
    svg.selectAll("*").remove();
    const visited=new Set(PLAN.steps.slice(0,cur+1).filter(s=>!s.drop).map(s=>s.layer+":"+s.node));
    const pathPts=[];
    for(let l=LYR.length-1;l>=0;l--){
      const py=panelY(l);
      // panel
      svg.append("rect").attr("x",px-18).attr("y",py).attr("width",pw+36).attr("height",ph).attr("rx",9)
        .attr("fill", l===0?"rgba(34,211,238,.04)":"#0e1016").attr("stroke","#222833");
      svg.append("text").attr("x",px-12).attr("y",py+15).attr("fill",C.muted).attr("font-size",10)
        .text("layer "+l+(l===LYR.length-1?"  (entry · sparsest)":l===0?"  (base · densest)":""));
      // edges
      const ids=LYR[l];
      ids.forEach(i=>{ (EDGE[l].get(i)||[]).forEach(j=>{ if(i<j){
        svg.append("line").attr("x1",X(P[i].x)).attr("y1",Y(l,P[i].y)).attr("x2",X(P[j].x)).attr("y2",Y(l,P[j].y))
          .attr("stroke",C.edge).attr("stroke-width",1); }});});
      // query marker (projected into this panel)
      svg.append("line").attr("x1",X(q().x)).attr("y1",py+6).attr("x2",X(q().x)).attr("y2",py+ph-6).attr("stroke",C.q).attr("stroke-dasharray","2 4").attr("stroke-opacity",.5);
      // nodes
      ids.forEach(i=>{ const on=visited.has(l+":"+i), p=P[i];
        svg.append("circle").attr("cx",X(p.x)).attr("cy",Y(l,p.y)).attr("r",on?6:4.2)
          .attr("fill",on?C.good:C.node).attr("fill-opacity",on?1:0.65).attr("stroke",on?"#0e1016":"none").attr("stroke-width",1.5); });
    }
    // draw the traversed path (within + across layers) up to cur
    const seq=PLAN.steps.slice(0,cur+1);
    let prev=null;
    seq.forEach(s=>{ const p=P[s.node], pt=[X(p.x),Y(s.layer,p.y)];
      if(prev){ svg.append("line").attr("x1",prev[0]).attr("y1",prev[1]).attr("x2",pt[0]).attr("y2",pt[1])
        .attr("stroke",s.drop?C.accent:C.good).attr("stroke-width",2.2).attr("stroke-dasharray",s.drop?"4 3":null)
        .attr("marker-end",null); }
      prev=pt; });
    // current node pulse
    const c=PLAN.steps[cur]; if(c){ const p=P[c.node];
      svg.append("circle").attr("cx",X(p.x)).attr("cy",Y(c.layer,p.y)).attr("r",9).attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2); }
    // answer marker when done
    const done = cur>=PLAN.steps.length-1;
    if(done){ const p=P[PLAN.answer]; const py=panelY(0);
      svg.append("circle").attr("cx",X(p.x)).attr("cy",Y(0,p.y)).attr("r",8).attr("fill",C.good);
      svg.append("text").attr("x",X(p.x)).attr("y",py+ph+ -2).attr("text-anchor","middle").attr("fill",C.good).attr("font-size",9).text("nearest ✓"); }
    const s=PLAN.steps[cur];
    const msg = s.drop ? `drop to layer ${s.layer-1} at the local-best node`
      : s.first ? `layer ${s.layer}: enter at the entry point`
      : `layer ${s.layer}: hop to a closer neighbor`;
    d3.select("#hn-read").html(done
      ? `<span style="color:${C.good}">▶</span> Reached the nearest neighbor in <b>${PLAN.steps.filter(x=>!x.drop).length} hops</b> across ${LYR.length} layers — vs scanning all ${P.length} vectors.`
      : `<span style="color:${C.accent}">▶</span> ${msg} &nbsp;<span style="color:${C.muted}">(step ${cur+1}/${PLAN.steps.length})</span>`);
  }
  let timer=null;
  function step(){ if(cur<PLAN.steps.length-1){cur++;draw();} else if(timer){clearInterval(timer);timer=null;} }
  d3.select("#hn-step").on("click",()=>{ if(timer){clearInterval(timer);timer=null;} step(); });
  d3.select("#hn-auto").on("click",()=>{ if(timer){clearInterval(timer);timer=null;return;} timer=setInterval(step,700); });
  d3.select("#hn-reset").on("click",()=>{ if(timer){clearInterval(timer);timer=null;} cur=0; draw(); });
  d3.select("#hn-qx").on("input",function(){ if(timer){clearInterval(timer);timer=null;} qx=+this.value; rebuild(); });
  rebuild();
})();

/* ---- Viz 2: efSearch recall vs latency ---- */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",muted:"#9aa3b2",ink:"#e6e9ef"};
  const svg=d3.select("#ef-svg"),W=640,H=200;
  const px=50,py=22,pw=W-150,ph=150, efmin=8,efmax=400;
  const X=e=>px+(e-efmin)/(efmax-efmin)*pw;
  // recall saturates with ef; latency grows ~linearly
  const recall=e=>1-0.55*Math.exp(-e/55);          // → ~1.0
  const latency=e=>0.12+e/efmax*0.95;               // normalized 0..~1
  function draw(){
    svg.selectAll("*").remove();
    const e=+d3.select("#ef-v").property("value");
    svg.append("line").attr("x1",px).attr("y1",py+ph).attr("x2",px+pw).attr("y2",py+ph).attr("stroke","#3a4150");
    svg.append("text").attr("x",px+pw/2).attr("y",py+ph+18).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("efSearch →");
    const es=d3.range(efmin,efmax+1,4);
    const Yr=v=>py+ph-v*ph, Yl=v=>py+ph-Math.min(v,1)*ph;
    svg.append("path").attr("d",d3.line().x(X).y(d=>Yr(recall(d)))(es)).attr("fill","none").attr("stroke",C.good).attr("stroke-width",2.4);
    svg.append("path").attr("d",d3.line().x(X).y(d=>Yl(latency(d)))(es)).attr("fill","none").attr("stroke",C.bad).attr("stroke-width",2.4).attr("stroke-dasharray","5 3");
    svg.append("text").attr("x",px+pw+8).attr("y",Yr(recall(efmax))+4).attr("fill",C.good).attr("font-size",10).text("recall");
    svg.append("text").attr("x",px+pw+8).attr("y",Yl(latency(efmax))+4).attr("fill",C.bad).attr("font-size",10).text("latency");
    svg.append("line").attr("x1",X(e)).attr("y1",py).attr("x2",X(e)).attr("y2",py+ph).attr("stroke",C.accent).attr("stroke-dasharray","3 3");
    svg.append("circle").attr("cx",X(e)).attr("cy",Yr(recall(e))).attr("r",4.5).attr("fill",C.good);
    svg.append("circle").attr("cx",X(e)).attr("cy",Yl(latency(e))).attr("r",4.5).attr("fill",C.bad);
    d3.select("#ef-read").html(`efSearch <b>${e}</b> → recall <b style="color:${C.good}">${(recall(e)*100).toFixed(1)}%</b> · relative latency <b style="color:${C.bad}">${(latency(e)).toFixed(2)}×</b> — past the knee, recall barely moves but latency keeps climbing.`);
  }
  d3.select("#ef-v").on("input",draw);
  draw();
})();
