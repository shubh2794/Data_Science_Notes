/* graph-neural-networks.viz.js — extracted from graph-neural-networks.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ─────────────────── 02 · receptive field = L hops ─────────────────── */
(function(){
  const svg=d3.select("#gnn-svg");
  const P=[[60,170],[140,90],[140,250],[215,170],[225,60],[230,290],[300,110],[305,225],
           [380,60],[385,170],[390,285],[470,110],[475,235],[550,170],[600,80],[600,265]];
  const nodes=P.map((p,i)=>({id:i,x:p[0],y:p[1]}));
  const edges=[[0,1],[0,2],[1,3],[2,3],[1,4],[2,5],[3,6],[3,7],[4,6],[5,7],[6,8],[6,9],
               [7,9],[7,10],[8,9],[9,10],[8,11],[9,11],[9,12],[10,12],[11,13],[12,13],
               [11,14],[13,14],[13,15],[12,15]];
  const adj=nodes.map(()=>[]);
  edges.forEach(([a,b])=>{ adj[a].push(b); adj[b].push(a); });

  const gE=svg.append("g"), gN=svg.append("g"), gL=svg.append("g");
  const lines=gE.selectAll("line").data(edges).enter().append("line")
    .attr("x1",d=>nodes[d[0]].x).attr("y1",d=>nodes[d[0]].y)
    .attr("x2",d=>nodes[d[1]].x).attr("y2",d=>nodes[d[1]].y).attr("stroke-width",1.6);

  const hopCol=k=> k===0 ? C.B : d3.interpolateRgb(C.A,"#2f3a52")(Math.min(1,(k-1)/4));
  let centre=3, L=1;

  function hops(src){
    const dist=nodes.map(()=>Infinity); dist[src]=0; const q=[src];
    while(q.length){ const v=q.shift(); adj[v].forEach(u=>{ if(dist[u]===Infinity){ dist[u]=dist[v]+1; q.push(u); } }); }
    return dist;
  }

  const g=gN.selectAll("g.nd").data(nodes).enter().append("g").attr("class","nd")
    .attr("transform",d=>`translate(${d.x},${d.y})`).attr("cursor","pointer")
    .on("click",(ev,d)=>{ centre=d.id; render(); });
  g.append("circle").attr("r",14).attr("stroke","#0f1117").attr("stroke-width",2);
  g.append("text").attr("text-anchor","middle").attr("dy",4).attr("font-size",11).attr("font-weight",600);

  // legend
  const lg=gL.append("g").attr("transform","translate(16,318)");
  ["centre","1 hop","2 hops","3+ hops","unreached"].forEach((t,i)=>{
    const col=i===4?"#232a36":hopCol(i);
    const x=i*112;
    lg.append("circle").attr("cx",x).attr("cy",-4).attr("r",6).attr("fill",col).attr("stroke","#0f1117").attr("stroke-width",1.4);
    lg.append("text").attr("x",x+11).attr("y",0).attr("font-size",10.5).attr("fill",C.muted).text(t);
  });

  function render(){
    const dist=hops(centre);
    lines.transition().duration(250)
      .attr("stroke",d=>(dist[d[0]]<=L&&dist[d[1]]<=L)?C.A:C.line)
      .attr("stroke-opacity",d=>(dist[d[0]]<=L&&dist[d[1]]<=L)?0.85:0.35)
      .attr("stroke-width",d=>(dist[d[0]]<=L&&dist[d[1]]<=L)?2.4:1.4);
    g.select("circle").transition().duration(250)
      .attr("r",d=>dist[d.id]===0?17:14)
      .attr("fill",d=>dist[d.id]<=L?hopCol(dist[d.id]):"#232a36")
      .attr("stroke",d=>dist[d.id]===0?"#fff":"#0f1117");
    g.select("text").attr("fill",d=>dist[d.id]<=L?"#101520":C.muted).text(d=>d.id);
    const counts=[]; let total=0;
    for(let k=0;k<=L;k++){ const c=dist.filter(v=>v===k).length; total+=c; if(k>0) counts.push(`${k}-hop: <b>${c}</b>`); }
    const tail=L===0?"it sees only its own features (h⁽⁰⁾ = x<sub>v</sub>)":counts.join(" · ");
    d3.select("#gnn-readout").html(
      `centre <b>${centre}</b> · L = <b>${L}</b> → receptive field <b>${total}</b>/${nodes.length} nodes &nbsp;·&nbsp; ${tail}`);
  }
  const sl=document.getElementById("gnn-layers"), lv=document.getElementById("gnn-lval");
  sl.addEventListener("input",()=>{ L=+sl.value; lv.textContent=L; render(); });
  d3.select("#gnn-reset").on("click",()=>{ centre=3; L=1; sl.value=1; lv.textContent="1"; render(); });
  render();
})();

/* ─────────────────── 05 · aggregation weights: GCN / mean / GAT ─────────────────── */
(function(){
  const svg=d3.select("#agg-svg"), cx=205, cy=150;
  const nb=[{y:40,d:2,s:1.4},{y:95,d:3,s:-0.6},{y:150,d:6,s:0.9},{y:205,d:10,s:-1.2},{y:260,d:4,s:0.3}];
  const nx=430, selfScore=0.7, dv=nb.length;               // centre degree = 5
  const lrelu=z=> z>0 ? z : 0.2*z;
  const gE=svg.append("g"), gN=svg.append("g"), gT=svg.append("g");

  function weights(mode){
    if(mode==="mean") return {w:nb.map(()=>1/dv), self:null, sum:1};
    if(mode==="gcn"){ const w=nb.map(n=>1/Math.sqrt((dv+1)*(n.d+1))); return {w, self:1/(dv+1), sum:w.reduce((a,b)=>a+b,0)+1/(dv+1)}; }
    const ex=nb.map(n=>Math.exp(lrelu(n.s))), es=Math.exp(lrelu(selfScore));
    const Z=ex.reduce((a,b)=>a+b,0)+es;
    return {w:ex.map(e=>e/Z), self:es/Z, sum:1};
  }
  const copy={
    gcn:"GCN — weight <b>1/√(d̃<sub>v</sub> d̃<sub>u</sub>)</b>: purely structural. High-degree neighbours are damped; features are irrelevant. Weights do <b>not</b> sum to 1.",
    mean:"Mean (GraphSAGE) — every sampled neighbour counts <b>equally</b>; the self-vector is kept aside and concatenated, not averaged in.",
    gat:"GAT — <b>α<sub>vu</sub> = softmax(LeakyReLU(aᵀ[Wh<sub>v</sub>‖Wh<sub>u</sub>]))</b>: learned from features, a convex combination over the neighbourhood plus self."
  };

  const eg=gE.selectAll("line").data(nb).enter().append("line")
    .attr("x1",cx).attr("y1",cy).attr("x2",nx).attr("y2",d=>d.y).attr("stroke",C.A).attr("stroke-linecap","round");
  const wl=gT.selectAll("text.w").data(nb).enter().append("text").attr("class","w")
    .attr("x",(cx+nx)/2).attr("y",d=>(cy+d.y)/2-6).attr("text-anchor","middle").attr("font-size",11)
    .attr("font-family","Menlo,monospace").attr("fill",C.B)
    .attr("paint-order","stroke").attr("stroke","#0f1117").attr("stroke-width",3).attr("stroke-linejoin","round");

  const ng=gN.selectAll("g").data(nb).enter().append("g").attr("transform",d=>`translate(${nx},${d.y})`);
  ng.append("circle").attr("r",15).attr("fill","#2a3340").attr("stroke",C.A).attr("stroke-width",1.6);
  ng.append("text").attr("text-anchor","middle").attr("dy",4).attr("font-size",11).attr("fill",C.ink).text((d,i)=>"u"+(i+1));
  ng.append("text").attr("x",22).attr("dy",4).attr("font-size",10.5).attr("fill",C.muted).text(d=>"deg "+d.d);

  const cnode=gN.append("g").attr("transform",`translate(${cx},${cy})`);
  cnode.append("circle").attr("r",20).attr("fill",C.B).attr("stroke","#fff").attr("stroke-width",2);
  cnode.append("text").attr("text-anchor","middle").attr("dy",5).attr("font-size",13).attr("font-weight",700).attr("fill","#101520").text("v");
  const selfTxt=gT.append("text").attr("x",cx).attr("y",cy+40).attr("text-anchor","middle").attr("font-size",11)
    .attr("font-family","Menlo,monospace").attr("fill",C.good);
  gT.append("text").attr("x",cx).attr("y",cy-32).attr("text-anchor","middle").attr("font-size",10.5).attr("fill",C.muted).text("deg 5");

  function draw(mode){
    const {w,self,sum}=weights(mode);
    eg.data(nb).transition().duration(350).attr("stroke-width",(d,i)=>1.5+26*w[i]).attr("stroke-opacity",(d,i)=>0.35+0.65*Math.min(1,w[i]*3));
    wl.data(nb).text((d,i)=>w[i].toFixed(3));
    selfTxt.text(self===null?"self: concatenated":"self weight "+self.toFixed(3));
    d3.select("#agg-readout").html(copy[mode]+` &nbsp;·&nbsp; Σ weights = <b>${sum.toFixed(3)}</b>`);
    ["gcn","mean","gat"].forEach(m=>d3.select("#agg-"+m).attr("class",m===mode?"btn":"btn ghost"));
  }
  ["gcn","mean","gat"].forEach(m=>d3.select("#agg-"+m).on("click",()=>draw(m)));
  draw("gcn");
})();

/* ─────────────────── 09 · over-smoothing ─────────────────── */
(function(){
  const svg=d3.select("#os-svg"), N=14, STEPS=14, ALPHA=0.3;
  const pos=[], comm=[];
  for(let i=0;i<7;i++){ const a=i*2*Math.PI/7-0.4; pos.push([105+62*Math.cos(a),120+62*Math.sin(a)]); comm.push(0); }
  for(let i=0;i<7;i++){ const a=i*2*Math.PI/7+0.9; pos.push([240+62*Math.cos(a),250+62*Math.sin(a)]); comm.push(1); }
  const E=[];
  for(let b=0;b<2;b++){ const o=b*7;
    for(let i=0;i<7;i++) E.push([o+i,o+(i+1)%7]);
    E.push([o+0,o+3]); E.push([o+1,o+4]); E.push([o+2,o+5]); }
  [[2,10],[1,11],[2,9],[0,11]].forEach(e=>E.push(e));   // bridges between the two communities

  // normalized adjacency with self-loops:  Â = D̃^-1/2 (A+I) D̃^-1/2
  const deg=new Array(N).fill(1);
  E.forEach(([a,b])=>{ deg[a]++; deg[b]++; });
  function prop(h){
    const acc=new Array(N).fill(0);
    for(let i=0;i<N;i++) acc[i]+=h[i]/deg[i];   // self-loop term: 1/√(d̃ᵢd̃ᵢ)
    E.forEach(([a,b])=>{ const s=1/Math.sqrt(deg[a]*deg[b]); acc[a]+=s*h[b]; acc[b]+=s*h[a]; });
    return acc;
  }
  const h0=[]; for(let i=0;i<N;i++) h0.push((comm[i]?-1:1)*(1+0.12*Math.sin(i*2.3)));
  function run(res){ const seq=[h0.slice()];
    for(let l=0;l<STEPS;l++){ const p=prop(seq[l]);
      seq.push(res? p.map((v,i)=>(1-ALPHA)*v+ALPHA*h0[i]) : p); }
    return seq; }
  const seqP=run(false), seqR=run(true);
  const gap=s=>{ const A=d3.mean(s.filter((v,i)=>!comm[i])), B=d3.mean(s.filter((v,i)=>comm[i])); return Math.abs(A-B); };
  const g0=gap(h0);
  const curveP=seqP.map(s=>gap(s)/g0), curveR=seqR.map(s=>gap(s)/g0);

  const col=d3.scaleLinear().domain([-1,0,1]).range([C.B,"#39414f",C.A]).clamp(true);
  const gE=svg.append("g"), gN=svg.append("g"), gC=svg.append("g");
  svg.append("text").attr("x",20).attr("y",26).attr("font-size",11).attr("fill",C.muted).text("two communities · node colour = feature value");
  gE.selectAll("line").data(E).enter().append("line")
    .attr("x1",d=>pos[d[0]][0]).attr("y1",d=>pos[d[0]][1])
    .attr("x2",d=>pos[d[1]][0]).attr("y2",d=>pos[d[1]][1])
    .attr("stroke",d=>comm[d[0]]!==comm[d[1]]?C.good:C.line).attr("stroke-width",d=>comm[d[0]]!==comm[d[1]]?2:1.4);
  const nodesSel=gN.selectAll("circle").data(d3.range(N)).enter().append("circle")
    .attr("cx",i=>pos[i][0]).attr("cy",i=>pos[i][1]).attr("r",11)
    .attr("stroke","#0f1117").attr("stroke-width",1.8);

  // chart
  const x0=375,x1=620,y0=300,y1=60;
  const xs=d3.scaleLinear().domain([0,STEPS]).range([x0,x1]);
  const ys=d3.scaleLinear().domain([0,1]).range([y0,y1]);
  gC.append("line").attr("x1",x0).attr("y1",y0).attr("x2",x1).attr("y2",y0).attr("stroke","#3a4150");
  gC.append("line").attr("x1",x0).attr("y1",y0).attr("x2",x0).attr("y2",y1).attr("stroke","#3a4150");
  [0,0.5,1].forEach(t=>{ gC.append("line").attr("x1",x0).attr("y1",ys(t)).attr("x2",x1).attr("y2",ys(t)).attr("stroke","#1b2130");
    gC.append("text").attr("x",x0-6).attr("y",ys(t)+4).attr("text-anchor","end").attr("font-size",10).attr("fill",C.muted).text(t); });
  [0,7,14].forEach(t=>gC.append("text").attr("x",xs(t)).attr("y",y0+16).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text(t));
  gC.append("text").attr("x",(x0+x1)/2).attr("y",y0+34).attr("text-anchor","middle").attr("font-size",10.5).attr("fill",C.muted).text("layers");
  gC.append("text").attr("x",x0).attr("y",y1-16).attr("font-size",10.5).attr("fill",C.muted).text("class separation (relative to layer 0)");
  const ln=d3.line().x((d,i)=>xs(i)).y(d=>ys(Math.min(1,d)));
  gC.append("path").datum(curveP).attr("d",ln).attr("fill","none").attr("stroke",C.bad).attr("stroke-width",2);
  gC.append("path").datum(curveR).attr("d",ln).attr("fill","none").attr("stroke",C.good).attr("stroke-width",2).attr("stroke-dasharray","5 3");
  gC.append("text").attr("x",x1).attr("y",ys(curveP[STEPS])+16).attr("text-anchor","end").attr("font-size",10).attr("fill",C.bad).text("plain");
  gC.append("text").attr("x",x1).attr("y",ys(curveR[STEPS])-8).attr("text-anchor","end").attr("font-size",10).attr("fill",C.good).text("initial residual");
  const marker=gC.append("line").attr("y1",y1).attr("y2",y0).attr("stroke",C.muted).attr("stroke-dasharray","3 3");
  const dotP=gC.append("circle").attr("r",4).attr("fill",C.bad);
  const dotR=gC.append("circle").attr("r",4).attr("fill",C.good);

  const sl=document.getElementById("os-layers"), lv=document.getElementById("os-lval"), cb=document.getElementById("os-res");
  function render(){
    const L=+sl.value, res=cb.checked; lv.textContent=L;
    const s=(res?seqR:seqP)[L];
    nodesSel.transition().duration(200).attr("fill",i=>col(s[i]));
    marker.attr("x1",xs(L)).attr("x2",xs(L));
    dotP.attr("cx",xs(L)).attr("cy",ys(Math.min(1,curveP[L])));
    dotR.attr("cx",xs(L)).attr("cy",ys(Math.min(1,curveR[L])));
    const sep=(res?curveR:curveP)[L], spread=d3.deviation(s)||0;
    const verdict = sep>0.55 ? `<b style="color:${C.good}">separable</b>` : sep>0.2 ? `<b style="color:${C.B}">fading</b>` : `<b style="color:${C.bad}">collapsed</b>`;
    d3.select("#os-readout").html(
      `L = <b>${L}</b> · ${res?"initial residual":"plain propagation"} &nbsp;·&nbsp; class separation <b>${(100*sep).toFixed(0)}%</b> of layer 0 &nbsp;·&nbsp; node spread <b>${spread.toFixed(3)}</b> &nbsp;·&nbsp; ${verdict}`);
  }
  sl.addEventListener("input",render); cb.addEventListener("change",render); render();
})();
