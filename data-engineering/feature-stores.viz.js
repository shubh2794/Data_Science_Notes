/* feature-stores.viz.js — interactive figures for feature-stores.html.
   Loaded after data.js / notes.js. Uses d3 from ../vendor. */

/* ---- Viz 1: point-in-time (as-of) join vs naive latest-value join ---- */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",warm:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#2a2f3a",feat:"#5b9cff"};
  const svg=d3.select("#pit-svg");
  if(svg.empty()) return;
  // spend_7d for customer 42: [event time (day), value]
  const F=[[0,120],[3,95],[6,210],[9,180],[12,40],[15,260],[18,300]];
  const labels=[{t:4.5,y:"no churn"},{t:10,y:"no churn"},{t:16.5,y:"churned"}];
  const D0=0, D1=20;
  const X=d3.scaleLinear().domain([D0,D1]).range([50,610]);
  const Y=d3.scaleLinear().domain([0,340]).range([140,24]);
  const LY=200;   // label lane
  function pick(t,mode,delta,ttl){
    if(mode==="naive"){ const f=F[F.length-1]; return {f}; }
    const ok=F.filter(([tau])=>tau+delta<=t && tau>=t-ttl);
    return ok.length?{f:ok[ok.length-1]}:{f:null};
  }
  function truth(t,delta){ const ok=F.filter(([tau])=>tau+delta<=t); return ok.length?ok[ok.length-1]:null; }
  function draw(){
    svg.selectAll("*").remove();
    const mode=d3.select("#pit-mode").property("value"), delta=+d3.select("#pit-delay").property("value"), ttl=+d3.select("#pit-ttl").property("value");
    d3.select("#pit-delayv").text(delta+" d"); d3.select("#pit-ttlv").text(ttl+" d");
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${LY+22})`).call(d3.axisBottom(X).ticks(10).tickFormat(d=>"day "+d));
    svg.append("g").attr("class","axis").attr("transform","translate(50,0)").call(d3.axisLeft(Y).ticks(4));
    svg.append("text").attr("x",56).attr("y",16).attr("fill",C.muted).attr("font-size",10).text("feature: customer_42.spend_7d (event time of each update)");
    // step line of the feature's true value over event time
    const pts=F.map(([t,v])=>[X(t),Y(v)]).concat([[X(D1),Y(F[F.length-1][1])]]);
    svg.append("path").attr("d",d3.line().curve(d3.curveStepAfter)(pts)).attr("fill","none").attr("stroke",C.feat).attr("stroke-width",1.6).attr("stroke-opacity",.7);
    F.forEach(([t,v])=>{
      svg.append("circle").attr("cx",X(t)).attr("cy",Y(v)).attr("r",4.5).attr("fill",C.feat);
      svg.append("text").attr("x",X(t)+5).attr("y",Y(v)-7).attr("fill",C.ink).attr("font-size",9.5).text(v);
      if(delta>0){
        svg.append("line").attr("x1",X(t)).attr("x2",X(Math.min(D1,t+delta))).attr("y1",150).attr("y2",150).attr("stroke",C.warm).attr("stroke-width",2).attr("stroke-opacity",.6);
        svg.append("circle").attr("cx",X(Math.min(D1,t+delta))).attr("cy",150).attr("r",2.5).attr("fill",C.warm);
      }
    });
    if(delta>0) svg.append("text").attr("x",610).attr("y",162).attr("text-anchor","end").attr("fill",C.warm).attr("font-size",9).text("amber = ingestion delay: value only queryable at τ + δ");
    // labels
    const res=[];
    labels.forEach((L,i)=>{
      const p=pick(L.t,mode,delta,ttl), tr=truth(L.t,delta);
      const leak=p.f&&p.f[0]+delta>L.t, isNull=!p.f;
      const ttlNull=isNull&&tr;
      const col=leak?C.bad:isNull?C.muted:C.good;
      res.push({L,p,leak,isNull,ttlNull,tr});
      if(mode==="asof") svg.append("rect").attr("x",X(Math.max(D0,L.t-ttl))).attr("y",LY-6).attr("width",X(L.t)-X(Math.max(D0,L.t-ttl))).attr("height",12).attr("rx",3).attr("fill","rgba(34,211,238,.10)");
      svg.append("line").attr("x1",X(L.t)).attr("x2",X(L.t)).attr("y1",24).attr("y2",LY).attr("stroke",col).attr("stroke-dasharray","4 3");
      if(p.f){
        svg.append("path").attr("d",`M${X(L.t)},${LY-8} Q${(X(L.t)+X(p.f[0]))/2},${LY-40} ${X(p.f[0])},${Y(p.f[1])+6}`)
          .attr("fill","none").attr("stroke",col).attr("stroke-width",1.8);
        svg.append("circle").attr("cx",X(p.f[0])).attr("cy",Y(p.f[1])).attr("r",7).attr("fill","none").attr("stroke",col).attr("stroke-width",2);
      }
      const g=svg.append("g").attr("class","dragpt").style("cursor","ew-resize");
      g.append("circle").attr("cx",X(L.t)).attr("cy",LY).attr("r",9).attr("fill",col).attr("stroke","#0e1016").attr("stroke-width",2);
      g.append("text").attr("x",X(L.t)).attr("y",LY+4).attr("text-anchor","middle").attr("fill","#06101f").attr("font-size",9).attr("font-weight",700).text("t"+(i+1));
      g.call(d3.drag().on("drag",ev=>{ L.t=Math.round(Math.max(0.5,Math.min(D1-0.5,X.invert(ev.x)))*2)/2; draw(); }));
      svg.append("text").attr("x",X(L.t)).attr("y",LY+42).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9).text(`label: ${L.y}`);
    });
    const parts=res.map((r,i)=>{
      const v=r.p.f?r.p.f[1]:"null";
      const tag=r.leak?`<span style="color:${C.bad}">LEAK (value from day ${r.p.f[0]}${delta?" / lands "+(r.p.f[0]+delta):""})</span>`
        : r.ttlNull?`<span style="color:${C.muted}">null — latest value older than TTL</span>`
        : r.isNull?`<span style="color:${C.muted}">null — nothing known yet</span>`
        : `<span style="color:${C.good}">ok</span>`;
      return `t${i+1}=day ${r.L.t} → <b>${v}</b> ${tag}`;
    });
    const leaks=res.filter(r=>r.leak).length;
    d3.select("#pit-read").html(parts.join(" · ")+(leaks?` · <b style="color:${C.bad}">${leaks} of ${res.length} training rows see the future</b>`:""));
  }
  d3.selectAll("#pit-mode").on("change",draw);
  d3.selectAll("#pit-delay,#pit-ttl").on("input",draw);
  draw();
})();

/* ---- Viz 2: freshness sawtooth (batch) vs streaming ---- */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",warm:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef"};
  const svg=d3.select("#fr-svg");
  if(svg.empty()) return;
  const H=48, STREAM=0.03;   // hours shown; streaming staleness ≈ 2 min
  const X=d3.scaleLinear().domain([0,H]).range([50,560]);
  function draw(){
    svg.selectAll("*").remove();
    const D=+d3.select("#fr-int").property("value"), L=+d3.select("#fr-lat").property("value"), S=+d3.select("#fr-sla").property("value");
    d3.select("#fr-intv").text(D+" h"); d3.select("#fr-latv").text(L+" h"); d3.select("#fr-slav").text(S+" h");
    const ymax=Math.max(30,L+D+2);
    const Y=d3.scaleLinear().domain([0,ymax]).range([186,16]);
    svg.append("g").attr("class","axis").attr("transform","translate(0,186)").call(d3.axisBottom(X).ticks(8).tickFormat(d=>d+"h"));
    svg.append("g").attr("class","axis").attr("transform","translate(50,0)").call(d3.axisLeft(Y).ticks(5).tickFormat(d=>d+"h"));
    svg.append("text").attr("x",56).attr("y",12).attr("fill",C.muted).attr("font-size",9.5).text("staleness of the value served online");
    // batch sawtooth: latest landed event time = max kD with kD + L ≤ t (steady state, so k may be negative)
    const ts=d3.range(0,H+0.001,0.02);
    const stal=t=>{ const k=Math.floor((t-L)/D); return t-k*D; };
    svg.append("path").attr("d",d3.line().x(t=>X(t)).y(t=>Y(stal(t)))(ts)).attr("fill","none").attr("stroke",C.warm).attr("stroke-width",2);
    svg.append("path").attr("d",d3.line().x(t=>X(t)).y(()=>Y(STREAM))(ts)).attr("fill","none").attr("stroke",C.good).attr("stroke-width",2);
    // run markers
    for(let k=0;k*D<=H;k++){
      svg.append("line").attr("x1",X(k*D)).attr("x2",X(k*D)).attr("y1",186).attr("y2",181).attr("stroke",C.warm);
      if(k*D+L<=H) svg.append("circle").attr("cx",X(k*D+L)).attr("cy",Y(L)).attr("r",2.5).attr("fill",C.warm);
    }
    svg.append("line").attr("x1",X(0)).attr("x2",X(H)).attr("y1",Y(S)).attr("y2",Y(S)).attr("stroke",C.bad).attr("stroke-dasharray","5 3");
    svg.append("text").attr("x",X(H)+6).attr("y",Y(S)+4).attr("fill",C.bad).attr("font-size",10).text("SLA "+S+"h");
    svg.append("text").attr("x",X(H)+6).attr("y",Y(L+D/2)+4).attr("fill",C.warm).attr("font-size",10).text("batch");
    svg.append("text").attr("x",X(H)+6).attr("y",Y(STREAM)-2).attr("fill",C.good).attr("font-size",10).text("streaming");
    const viol=Math.max(0,Math.min(1,(L+D-S)/D));
    const mean=L+D/2;
    d3.select("#fr-read").html(`batch staleness ranges <b>${L}h → ${(L+D).toFixed(2).replace(/\.00$/,"")}h</b>, mean ≈ <b>${mean.toFixed(2).replace(/\.00$/,"")}h</b> · `+
      (viol>0?`<span style="color:${C.bad}">violates the ${S}h SLA ${(viol*100).toFixed(0)}% of the time</span>`:`<span style="color:${C.good}">always within the ${S}h SLA</span>`)+
      ` · to meet it, need Δ ≤ ${Math.max(0,S-L).toFixed(2).replace(/\.00$/,"")}h`+(S-L<=0?" (impossible in batch: latency alone exceeds the SLA — stream it)":"")+` · streaming ≈ ${Math.round(STREAM*60)} min, at the cost of an always-on job`);
  }
  d3.selectAll("#fr-int,#fr-lat,#fr-sla").on("input",draw);
  draw();
})();
