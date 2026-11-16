/* apache-spark.viz.js — interactive figures for apache-spark.html.
   Loaded after data.js / notes.js. Uses d3 from ../vendor. */

/* ---- Viz 1: transformations → stages (narrow vs wide, broadcast, lazy action) ---- */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",warm:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#2a2f3a",narrow:"#5b9cff",wide:"#c084fc"};
  const svg=d3.select("#dg-svg");
  if(svg.empty()) return;
  const BW=72, BH=34, Y0=74, YD=168, X0=14, STEP=88;
  let running=-1, doneUpTo=-1, timer=null;
  const opt=id=>d3.select("#"+id).property("checked");

  function build(){
    const ops=[{name:"read orders",kind:"source"}];
    if(opt("dg-filter")) ops.push({name:"filter",kind:"narrow"});
    if(opt("dg-with")) ops.push({name:"withColumn",kind:"narrow"});
    if(opt("dg-join")) ops.push({name:opt("dg-bcast")?"broadcast join":"sort-merge join",kind:opt("dg-bcast")?"narrow":"wide",join:true});
    if(opt("dg-group")) ops.push({name:"groupBy.agg",kind:"wide"});
    if(opt("dg-order")) ops.push({name:"orderBy",kind:"wide"});
    ops.push({name:"write (action)",kind:"action"});
    let st=0;
    ops.forEach((o,i)=>{ if(o.kind==="wide") st++; o.stage=st; o.x=X0+i*STEP; });
    const join=ops.find(o=>o.join);
    let dim=null, nStages=st+1, shuffles=ops.filter(o=>o.kind==="wide").length+ (join&&!opt("dg-bcast")?1:0);
    if(join){ dim={name:"read dim",x:join.x,bcast:opt("dg-bcast")};
      if(!dim.bcast){ dim.stage=nStages; nStages++; } }
    return {ops,dim,nStages,shuffles,join};
  }

  function stageColor(s,P){
    // execution order: the dim stage (if any) runs alongside stage 0; others in sequence
    const order = s===(P.dim&&!P.dim.bcast?P.dim.stage:-99) ? 0 : s;
    if(order<=doneUpTo) return "rgba(74,222,128,.10)";
    if(order===running) return "rgba(34,211,238,.16)";
    return "rgba(255,255,255,.025)";
  }

  function draw(){
    svg.selectAll("*").remove();
    const P=build();
    svg.append("defs").append("marker").attr("id","dg-arr").attr("viewBox","0 0 10 10").attr("refX",9).attr("refY",5)
      .attr("markerWidth",6).attr("markerHeight",6).attr("orient","auto").append("path").attr("d","M0,0L10,5L0,10z").attr("fill",C.muted);
    // stage panels on the main row
    const groups=d3.groups(P.ops,o=>o.stage);
    groups.forEach(([s,list])=>{
      const x0=d3.min(list,o=>o.x)-6, x1=d3.max(list,o=>o.x)+BW+6;
      svg.append("rect").attr("x",x0).attr("y",Y0-30).attr("width",x1-x0).attr("height",BH+44).attr("rx",9)
        .attr("fill",stageColor(s,P)).attr("stroke",C.line);
      svg.append("text").attr("x",x0+6).attr("y",Y0-16).attr("fill",C.muted).attr("font-size",10).text("stage "+s);
    });
    // dim side input
    if(P.dim){
      const d=P.dim;
      if(!d.bcast){
        svg.append("rect").attr("x",d.x-6).attr("y",YD-12).attr("width",BW+12).attr("height",BH+30).attr("rx",9)
          .attr("fill",stageColor(d.stage,P)).attr("stroke",C.line);
        svg.append("text").attr("x",d.x).attr("y",YD+BH+12).attr("fill",C.muted).attr("font-size",10).text("stage "+d.stage);
      }
      svg.append("rect").attr("x",d.x).attr("y",YD).attr("width",BW).attr("height",BH).attr("rx",6)
        .attr("fill",C.panel).attr("stroke",C.narrow);
      svg.append("text").attr("x",d.x+BW/2).attr("y",YD+21).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",10.5).text(d.name);
      svg.append("line").attr("x1",d.x+BW/2).attr("y1",YD-2).attr("x2",d.x+BW/2).attr("y2",Y0+BH+3)
        .attr("stroke",d.bcast?C.warm:C.wide).attr("stroke-width",1.6).attr("stroke-dasharray",d.bcast?"4 3":null).attr("marker-end","url(#dg-arr)");
      svg.append("text").attr("x",d.x+BW+(d.bcast?8:14)).attr("y",YD+21).attr("fill",d.bcast?C.warm:C.wide).attr("font-size",9.5)
        .text(d.bcast?"broadcast (copied to every executor)":"shuffle (Exchange)");
    }
    // boxes + arrows
    P.ops.forEach((o,i)=>{
      if(i>0){
        const px=P.ops[i-1].x+BW;
        svg.append("line").attr("x1",px+1).attr("y1",Y0+BH/2).attr("x2",o.x-2).attr("y2",Y0+BH/2)
          .attr("stroke",o.kind==="wide"?C.wide:C.muted).attr("stroke-width",o.kind==="wide"?2:1.2).attr("marker-end","url(#dg-arr)");
        if(o.kind==="wide"){
          svg.append("text").attr("x",(px+o.x)/2).attr("y",Y0+BH+30).attr("text-anchor","middle").attr("fill",C.wide).attr("font-size",9).text("shuffle");
        }
      }
      const col=o.kind==="wide"?C.wide:o.kind==="action"?C.warm:C.narrow;
      svg.append("rect").attr("x",o.x).attr("y",Y0).attr("width",BW).attr("height",BH).attr("rx",6)
        .attr("fill",C.panel).attr("stroke",col).attr("stroke-width",1.4);
      const words=o.name.split(" ");
      if(words.length>1 && o.name.length>11){
        svg.append("text").attr("x",o.x+BW/2).attr("y",Y0+15).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",10).text(words[0]);
        svg.append("text").attr("x",o.x+BW/2).attr("y",Y0+27).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",10).text(words.slice(1).join(" "));
      } else {
        svg.append("text").attr("x",o.x+BW/2).attr("y",Y0+21).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",10.5).text(o.name);
      }
      svg.append("text").attr("x",o.x+BW/2).attr("y",Y0-36).attr("text-anchor","middle").attr("fill",col).attr("font-size",9)
        .text(o.kind==="source"?"source":o.kind==="action"?"action":o.kind);
    });
    // legend
    const lg=[["narrow (pipelined)",C.narrow],["wide (shuffle)",C.wide],["action",C.warm]];
    lg.forEach(([t,c],i)=>{ svg.append("circle").attr("cx",380+i*95).attr("cy",214).attr("r",4).attr("fill",c);
      svg.append("text").attr("x",388+i*95).attr("y",218).attr("fill",C.muted).attr("font-size",9.5).text(t); });
    const status = running<0 && doneUpTo<0 ? `<span style="color:${C.muted}">plan only — nothing has executed yet (transformations are lazy)</span>`
      : running>=0 ? `<span style="color:${C.accent}">running stage ${running}…</span>`
      : `<span style="color:${C.good}">job complete</span>`;
    d3.select("#dg-read").html(`<b>${P.nStages}</b> stage${P.nStages>1?"s":""} · <b>${P.shuffles}</b> shuffle${P.shuffles===1?"":"s"} · ${status}`+
      (P.join&&!opt("dg-bcast")?` · the dimension side is shuffled too — try broadcasting it`:""));
    return P;
  }
  function stopRun(){ if(timer){clearInterval(timer);timer=null;} running=-1; doneUpTo=-1; }
  d3.selectAll("#viz-dag input").on("change",()=>{ stopRun(); draw(); });
  d3.select("#dg-run").on("click",()=>{
    stopRun(); const P=build(); const last=d3.max(P.ops,o=>o.stage);
    running=0; draw();
    timer=setInterval(()=>{ doneUpTo=running; running++; if(running>last){ clearInterval(timer); timer=null; running=-1; } draw(); },750);
  });
  draw();
})();

/* ---- Viz 2: skewed shuffle partitions and salting ---- */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",warm:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",bar:"#5b9cff"};
  const svg=d3.select("#sk-svg");
  if(svg.empty()) return;
  const P=8, N=80000, HOT=3, W=640, px=60, pw=420, py=16, ph=160;
  function rows(h,S){
    const r=d3.range(P).map(()=>({base:(1-h)*N/P,hot:0}));
    for(let j=0;j<S;j++) r[(HOT+j*3)%P].hot+=h*N/S;    // salted sub-keys hash to different partitions
    return r;
  }
  function draw(){
    svg.selectAll("*").remove();
    const h=+d3.select("#sk-hot").property("value"), S=+d3.select("#sk-salt").property("value");
    d3.select("#sk-hotv").text(Math.round(h*100)+"%"); d3.select("#sk-saltv").text(S);
    const R=rows(h,S), tot=R.map(d=>d.base+d.hot);
    const ymax=Math.max(N*0.85, d3.max(tot));
    const X=d3.scaleBand().domain(d3.range(P)).range([px,px+pw]).padding(0.22);
    const Y=d3.scaleLinear().domain([0,ymax]).range([py+ph,py]);
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${py+ph})`).call(d3.axisBottom(X).tickFormat(i=>"p"+i));
    svg.append("g").attr("class","axis").attr("transform",`translate(${px},0)`).call(d3.axisLeft(Y).ticks(5).tickFormat(d3.format("~s")));
    R.forEach((d,i)=>{
      svg.append("rect").attr("x",X(i)).attr("width",X.bandwidth()).attr("y",Y(d.base)).attr("height",Y(0)-Y(d.base)).attr("fill",C.bar).attr("fill-opacity",.8);
      if(d.hot>0) svg.append("rect").attr("x",X(i)).attr("width",X.bandwidth()).attr("y",Y(d.base+d.hot)).attr("height",Y(d.base)-Y(d.base+d.hot)).attr("fill",C.bad).attr("fill-opacity",.85);
    });
    const mean=N/P, mx=d3.max(tot);
    svg.append("line").attr("x1",px).attr("x2",px+pw).attr("y1",Y(mean)).attr("y2",Y(mean)).attr("stroke",C.good).attr("stroke-dasharray","4 3");
    svg.append("text").attr("x",px+pw+6).attr("y",Y(mean)+4).attr("fill",C.good).attr("font-size",10).text("mean (ideal)");
    svg.append("line").attr("x1",px).attr("x2",px+pw).attr("y1",Y(mx)).attr("y2",Y(mx)).attr("stroke",C.bad).attr("stroke-dasharray","2 3");
    svg.append("text").attr("x",px+pw+6).attr("y",Y(mx)+4).attr("fill",C.bad).attr("font-size",10).text("slowest task");
    svg.append("rect").attr("x",px+pw+6).attr("y",py+ph-34).attr("width",9).attr("height",9).attr("fill",C.bad);
    svg.append("text").attr("x",px+pw+19).attr("y",py+ph-26).attr("fill",C.muted).attr("font-size",10).text("hot-key rows");
    svg.append("rect").attr("x",px+pw+6).attr("y",py+ph-18).attr("width",9).attr("height",9).attr("fill",C.bar);
    svg.append("text").attr("x",px+pw+19).attr("y",py+ph-10).attr("fill",C.muted).attr("font-size",10).text("other keys");
    const unsalted=d3.max(rows(h,1).map(d=>d.base+d.hot));
    const extra = S>1 ? ` · salting adds a tiny second aggregation over ${S} partials per key` : "";
    d3.select("#sk-read").html(`skew ratio max/mean = <b>${(mx/mean).toFixed(2)}</b> · stage time ∝ slowest task = <b>${d3.format(",")(Math.round(mx))}</b> rows`+
      (S>1?` · <span style="color:${C.good}">${(unsalted/mx).toFixed(2)}× faster than unsalted</span>`:"")+extra);
  }
  d3.selectAll("#sk-hot,#sk-salt").on("input",draw);
  draw();
})();
